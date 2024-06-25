import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'Data')
KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, 'knowledge_base.json')
DATA_SOURCES = [
    os.path.join(DATA_DIR, 'data_source1.json'),
    os.path.join(DATA_DIR, 'data_source2.json')
]

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, 'Models')
LEMMATIZER_MODEL = os.path.join(MODELS_DIR, 'lemmatizer_model.pkl')
VECTORIZER_MODEL = os.path.join(MODELS_DIR, 'vectorizer_model.pkl')
NAIVE_BAYES_MODEL = os.path.join(MODELS_DIR, 'naive_bayes_model.pkl')

# Utils paths
UTILS_DIR = os.path.join(BASE_DIR, 'Utils')
STOP_WORDS_FILE = os.path.join(UTILS_DIR, 'stop_words.txt')

# Output paths
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')
PROGRESS_FILE = os.path.join(OUTPUT_DIR, 'progress.json')
TOPIC_MODEL_VISUALIZATION = os.path.join(OUTPUT_DIR, 'topic_model_visualization.png')

# Model parameters
NUM_TOPICS = 5  # Increased from 2 for more diverse topic modeling

# NLP settings
MAX_SUMMARY_LENGTH = 200
MIN_SUMMARY_LENGTH = 50

# Processing settings
BATCH_SIZE = 100
MAX_WORKERS = 4  # For parallel processing

# API keys (if needed)
API_KEY = os.environ.get('SUM_API_KEY', '')

# Debug mode
DEBUG = True

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, UTILS_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Validate critical paths
critical_paths = [KNOWLEDGE_BASE_PATH, STOP_WORDS_FILE, LEMMATIZER_MODEL, VECTORIZER_MODEL, NAIVE_BAYES_MODEL]
for path in critical_paths:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Critical file not found: {path}")
