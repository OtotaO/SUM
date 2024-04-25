# topic_modeling.py

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

class TopicModeler:
    """
    Topic modeling using Latent Dirichlet Allocation (LDA).

    Parameters:
    num_topics (int): The number of topics to extract.

    Attributes:
    lda_model (LatentDirichletAllocation): The LDA model instance.
    """

    def __init__(self, num_topics):
        self.num_topics = num_topics
        self.lda_model = LatentDirichletAllocation(n_topics=num_topics)

    def fit(self, X):
        """
        Fit the LDA model to the data.

        Parameters:
        X (array-like): The input data.

        Returns:
        self
        """
        self.lda_model.fit(X)
        return self

    def transform(self, X):
        """
        Transform the data into topic space.

        Parameters:
        X (array-like): The input data.

        Returns:
        array-like: The transformed data.
        """
        return self.lda_model.transform(X)

    def get_topics(self):
        """
        Get the topic weights.

        Returns:
        array-like: The topic weights.
        """
        return self.lda_model.components_
