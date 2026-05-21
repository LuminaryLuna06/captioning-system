import os
import faiss
import json
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
import config
from typing import List, Dict

class ImageIndexer:
    """
    A class to index images from a directory using DINOv3 features and FAISS.
    """
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.index = None
        self.id_to_path: Dict[int, str] = {}

    def _get_image_paths(self, directory: str) -> List[str]:
        """
        Recursively finds all image paths in a directory.
        """
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def build_index(self, directory: str, batch_size: int = config.BATCH_SIZE):
        """
        Scans a directory, extracts features in batches, and builds a FAISS index.
        """
        image_paths = self._get_image_paths(directory)
        if not image_paths:
            print("No images found in the specified directory.")
            return

        print(f"Found {len(image_paths)} images. Starting indexing...")

        # Since embeddings are L2 normalized, Inner Product (IP) is equivalent to Cosine Similarity.
        dummy_img = Image.new('RGB', config.IMAGE_SIZE)
        dummy_feat = self.extractor.extract_features(dummy_img)
        dimension = dummy_feat.shape[1]
        
        if config.USE_HNSW:
            # HNSW with IP (Inner Product)
            # M is the number of neighbors, efConstruction is the construction time parameter
            self.index = faiss.IndexHNSWFlat(dimension, config.HNSW_M, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
            print(f"Using HNSW index (M={config.HNSW_M}, efConstruction={config.HNSW_EF_CONSTRUCTION})")
        elif config.USE_QUANTIZATION:
            # Scalar Quantization (8-bit)
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexScalarQuantizer(dimension, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
            self.index.train(dummy_feat.astype('float32')) # Small training with dummy or first batch
            print("Using Scalar Quantization (8-bit) index")
        else:
            self.index = faiss.IndexFlatIP(dimension)
            print("Using Flat IP index")

        self.id_to_path = {}

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_paths = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Skipping corrupted/invalid image {path}: {e}")

            if batch_images:
                embeddings = self.extractor.extract_features(batch_images).astype('float32')
                
                # If using Quantization, we might need a better training set than just dummy
                if config.USE_QUANTIZATION and not self.index.is_trained:
                    self.index.train(embeddings)

                # Map current index IDs to paths
                start_id = self.index.ntotal
                for j, path in enumerate(valid_paths):
                    self.id_to_path[start_id + j] = path
                
                self.index.add(embeddings)
                print(f"Indexed {self.index.ntotal}/{len(image_paths)} images...")

        self.save_index()
        print("Indexing completed successfully.")

    def save_index(self, index_path: str = config.FAISS_INDEX_PATH, map_path: str = config.MAP_PATH):
        """
        Saves the FAISS index and the ID-to-path mapping to disk.
        """
        if self.index is not None:
            faiss.write_index(self.index, index_path)
            with open(map_path, 'w') as f:
                json.dump(self.id_to_path, f)
            print(f"Index saved to {index_path}")
            print(f"Mapping saved to {map_path}")

    def load_index(self, index_path: str = config.FAISS_INDEX_PATH, map_path: str = config.MAP_PATH):
        """
        Loads the FAISS index and the ID-to-path mapping from disk.
        """
        if os.path.exists(index_path) and os.path.exists(map_path):
            self.index = faiss.read_index(index_path)
            with open(map_path, 'r') as f:
                data = json.load(f)
                # JSON keys are always strings, convert back to int
                self.id_to_path = {int(k): v for k, v in data.items()}
            print("Index and mapping loaded successfully.")
            return True
        else:
            print("Index or mapping file not found.")
            return False

    def search(self, query_img: Image.Image, k: int = 5):
        """
        Searches for the top-k most similar images.
        """
        if self.index is None:
            if not self.load_index():
                raise ValueError("Index not loaded. Please index a folder first.")

        query_feat = self.extractor.extract_features(query_img).astype('float32')
        scores, indices = self.index.search(query_feat, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "path": self.id_to_path.get(idx, "Unknown"),
                    "score": float(score)
                })
        return results

if __name__ == "__main__":
    # For testing from command line
    import sys
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        indexer = ImageIndexer()
        indexer.build_index(target_dir)
    else:
        print("Usage: python indexer.py <dataset_directory>")
