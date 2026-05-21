import torch
import numpy as np
import cv2
import base64
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from typing import List, Union, Optional
import config

class FeatureExtractor:
    """
    A class to extract features from images using Meta's DINOv3 model.
    """
    def __init__(self, model_id: str = config.MODEL_ID):
        """
        Initializes the DINOv3 model and image processor.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, output_attentions=True).to(self.device)
        self.model.eval()

    def extract_features(self, images: Union[Image.Image, List[Image.Image]], return_attention: bool = False):
        """
        Extracts L2-normalized CLS token embeddings. 
        Optionally returns the last layer attention weight for heatmap.
        """
        if isinstance(images, Image.Image):
            images = [images]

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            if return_attention:
                # Get attentions from the last layer
                # outputs.attentions is a tuple of layers: (batch, num_heads, sequence_len, sequence_len)
                last_layer_attention = outputs.attentions[-1]
                # We want attention from CLS token (index 0) to all other patches
                cls_attention = last_layer_attention[:, :, 0, 1:] 
                # Mean across all attention heads
                cls_attention = cls_attention.mean(dim=1)
                return embeddings.cpu().numpy(), cls_attention.cpu().numpy()
            
        return embeddings.cpu().numpy()

    def generate_heatmap(self, image: Image.Image, attention: np.ndarray) -> str:
        """
        Generates a heatmap overlay and returns it as a base64 string.
        """
        # attention shape: (num_patches,)
        w, h = image.size
        # For dinov3-vits16: patch size 16x16, 224/16 = 14. So 14x14 = 196 patches.
        grid_size = int(np.sqrt(attention.shape[0]))
        attn_map = attention.reshape(grid_size, grid_size)
        
        # Rescale to 0-255
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        attn_map = (attn_map * 255).astype(np.uint8)
        
        # Resize to match original image
        attn_map_resized = cv2.resize(attn_map, (w, h))
        heatmap = cv2.applyColorMap(attn_map_resized, cv2.COLORMAP_JET)
        
        # Convert PIL image to CV2
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Overlay
        overlay = cv2.addWeighted(img_cv, 1 - config.HEATMAP_OPACITY, heatmap, config.HEATMAP_OPACITY, 0)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', overlay)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"

    def process_image_path(self, image_path: str) -> Optional[np.ndarray]:
        """
        Loads an image from path and extracts its features.
        """
        try:
            with Image.open(image_path).convert("RGB") as img:
                return self.extract_features(img)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
