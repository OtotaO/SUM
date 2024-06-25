import json
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class DataLoader:
    """
    Load and preprocess text data from various sources.

    Attributes:
        data (dict or list): The loaded data after preprocessing.
    """

    def __init__(self, data_file=None, data=None):
        """
        Initialize the DataLoader.

        Args:
            data_file (str, optional): Path to the JSON file containing data. Defaults to None.
            data (dict or list, optional): Pre-loaded data to be used. Defaults to None.

        Raises:
            ValueError: If both data_file and data are provided or neither is provided.
        """

        if data_file is None and data is None:
            raise ValueError("Please provide either data_file or data argument.")

        if data_file is not None and data is not None:
            raise ValueError("Cannot provide both data_file and data argument.")

        self.data = data
        if data_file:
            self.data = self.load_data(data_file)

    def load_data(self, data_file):
        """
        Load the data from a JSON file.

        Args:
            data_file (str): Path to the JSON file.

        Returns:
            dict or list: The loaded data.
        """

        with open(data_file, 'r') as f:
            data = json.load(f)
        return data

    def preprocess_data(self, lowercase=True, stem=False, lemmatize=True):
        """
        Preprocess the loaded data by tokenizing, removing stopwords,
        and optionally applying lowercase conversion, stemming, or lemmatization.

        Args:
            lowercase (bool, optional): Convert tokens to lowercase. Defaults to True.
            stem (bool, optional): Apply stemming to tokens. Defaults to False.
            lemmatize (bool, optional): Apply lemmatization to tokens (recommended for SUM.py). Defaults to True.

        Returns:
            list: The preprocessed data (list of lists of tokens).
        """

        if not isinstance(self.data, list):
            raise TypeError("Data must be a list for preprocessing.")

        stop_words = set(stopwords.words('english'))
        preprocessed_data = []
        for text in self.data:
            tokens = sent_tokenize(text)  # Sentence tokenization for SUM.py
            sentences = []
            for sentence in tokens:
                sentence_words = word_tokenize(sentence)
                if lowercase:
                    sentence_words = [token.lower() for token in sentence_words]
                sentence_words = [token for token in sentence_words if token not in stop_words]

                if lemmatize:
                    lemmatizer = WordNetLemmatizer()
                    sentence_words = [lemmatizer.lemmatize(token) for token in sentence_words]
                elif stem:
                    from nltk.stem import PorterStemmer
                    stemmer = PorterStemmer()
                    sentence_words = [stemmer.stem(token) for token in sentence_words]

                sentences.append(sentence_words)
            preprocessed_data.append(sentences)  # List of lists for sentences
        return preprocessed_data

