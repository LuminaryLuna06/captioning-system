import os

# Model Configuration
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Reranking Configuration
USE_RERANKING = True
HEAVY_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
RERANKING_TOP_K = 15

# Path Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(BASE_DIR, "data", "kb_images")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "data", "cache", "dino_faiss.index")
MAP_PATH = os.path.join(BASE_DIR, "data", "cache", "id_map.json")
KB_JSON_PATH = os.path.join(BASE_DIR, "data", "kb.json")

# Advanced Search Configuration
USE_HNSW = True
HNSW_M = 16  # Max number of outgoing connections in the graph
HNSW_EF_CONSTRUCTION = 100  # Size of the dynamic list for the nearest neighbors (higher = better quality, slower build)
USE_QUANTIZATION = True

# Heatmap Configuration
HEATMAP_OPACITY = 0.4
HEATMAP_SIZE = (224, 224)

# Create necessary directories
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
