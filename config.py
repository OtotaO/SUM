import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
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
    NUM_TOPICS = 5

    # NLP settings
    MAX_SUMMARY_LENGTH = 200
    MIN_SUMMARY_LENGTH = 50

    # Processing settings
    BATCH_SIZE = 100
    MAX_WORKERS = 4

    # API keys
    API_KEY = os.getenv('SUM_API_KEY', '')

    # Flask settings
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() in ('true', '1', 't')

    # Logging settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    @classmethod
    def init_app(cls):
        # Ensure directories exist
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.UTILS_DIR, cls.OUTPUT_DIR]:
            os.makedirs(directory, exist_ok=True)

        # Validate critical paths
        critical_paths = [cls.KNOWLEDGE_BASE_PATH, cls.STOP_WORDS_FILE, cls.LEMMATIZER_MODEL, cls.VECTORIZER_MODEL, cls.NAIVE_BAYES_MODEL]
        for path in critical_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Critical file not found: {path}")

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# Set the active configuration
active_config = config[os.getenv('FLASK_ENV', 'default')]
