"""
download_nltk_resources.py - Download required NLTK resources

This script downloads all the required NLTK resources for the SUM platform
using the centralized NLTK utilities.

Author: ototao
License: Apache License 2.0
"""

import os
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import centralized NLTK utilities
from Utils.nltk_utils import initialize_nltk, download_specific_resource, REQUIRED_RESOURCES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_nltk_resources():
    """Download all required NLTK resources using centralized utilities."""
    print("Downloading NLTK resources...")
    
    # Create nltk_data directory if it doesn't exist
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Initialize NLTK with all required resources
    success = initialize_nltk(download_dir=nltk_data_dir, quiet=False)
    
    if success:
        print("Successfully initialized all NLTK resources")
    else:
        print("Error initializing NLTK resources")
        
    # Special case for punkt_tab (if needed)
    print("Setting up punkt_tab...")
    try:
        # Create the punkt_tab directory structure
        punkt_tab_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt_tab', 'english')
        os.makedirs(punkt_tab_dir, exist_ok=True)
        
        # Copy punkt data to punkt_tab
        punkt_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
        if os.path.exists(punkt_dir):
            import shutil
            for file in os.listdir(punkt_dir):
                if file.endswith('.pickle'):
                    shutil.copy(os.path.join(punkt_dir, file), 
                               os.path.join(punkt_tab_dir, 'punkt.pickle'))
                    print(f"Created punkt_tab resource from punkt")
                    break
    except Exception as e:
        print(f"Error setting up punkt_tab: {e}")
    
    print("\nNLTK resources download complete!")
    print(f"Resources downloaded: {', '.join(REQUIRED_RESOURCES)}")

if __name__ == "__main__":
    download_nltk_resources()
