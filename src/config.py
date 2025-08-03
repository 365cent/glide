import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define base directory as the project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Define paths for data and models
LOGS_DIR = BASE_DIR / "logs"
LABELS_DIR = BASE_DIR / "labels"
PROCESSED_DIR = BASE_DIR / "processed"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# Ensure directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Performance configuration
VECTOR_SIZE = 300  # Standard embedding size
MAX_SEQ_LENGTH = 128  # For BERT
BATCH_SIZE = 8  # Default batch size
NUM_WORKERS = 2  # Default number of workers

# Performance thresholds for auto-optimization
SMALL_DATASET_THRESHOLD = 10000    # < 10K entries
MEDIUM_DATASET_THRESHOLD = 100000  # < 100K entries
LARGE_DATASET_THRESHOLD = 500000   # < 500K entries

# Performance configurations based on dataset size
PERF_CONFIG = {
    'small': {'batch_size': 16, 'workers': 4, 'clear_freq': 100},
    'medium': {'batch_size': 12, 'workers': 3, 'clear_freq': 50},
    'large': {'batch_size': 8, 'workers': 2, 'clear_freq': 25},
    'very_large': {'batch_size': 4, 'workers': 1, 'clear_freq': 10}
}

logger.info("Project directories and configuration initialized.")
