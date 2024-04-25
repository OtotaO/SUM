import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

class TopicModeler:
    def __init__(self, num_topics):
        self.num_topics = num_topics
        self.lda_model = LatentDirichletAllocation(n_topics=num_topics)

    def fit(self, X):
        self.lda_model.fit(X)

    def transform(self, X):
        return self.lda_model.transform(X)

    def get_topics(self):
        return self.lda_model.components_
