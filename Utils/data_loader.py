# data_loader.py

import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class DataLoader:
    """
    Load and preprocess the data.

    Attributes:
    data (dict): The loaded data.
    """

    def __init__(self, data_file):
        self.data_file = data_file
        self.data = self.load_data()

    def load_data(self):
        """
        Load the data from the JSON file.

        Returns:
        dict: The loaded data.
        """
        with open(self.data_file, 'r') as f:
            return json.load(f)

    def preprocess_data(self):
        """
        Preprocess the data by tokenizing and removing stopwords.

        Returns:
        list: The preprocessed data.
        """
        stop_words = set(stopwords.words('english'))
        preprocessed_data = []
        for text in self.data.values():
            tokens = word_tokenize(text)
            tokens = [token for token in tokens if token not in stop_words]
            preprocessed_data.append(tokens)
        return preprocessed_data
