# summarizer.py

from topic_modeling import TopicModeler
from data_loader import DataLoader
from nltk.tokenize import sent_tokenize

class Summarizer:
    """
    Summarize the data using topic modeling.

    Attributes:
    data_loader (DataLoader): The data loader instance.
    topic_modeler (TopicModeler): The topic modeler instance.
    """

    def __init__(self, data_file, num_topics):
        self.data_loader = DataLoader(data_file)
        self.topic_modeler = TopicModeler(num_topics)

    def summarize(self):
        """
        Summarize the data using topic modeling.

        Returns:
        list: The summarized data.
        """
        data = self.data_loader.preprocess_data()
        self.topic_modeler.fit(data)
        topic_weights = self.topic_modeler.get_topics()
        summarized_data = []
        for topic_weight in topic_weights:
            summary = self._generate_summary(topic_weight, data)
            summarized_data.append(summary)
        return summarized_data

    def _generate_summary(self, topic_weight, data):
        """
        Generate a summary for a given topic weight.

        Parameters:
        topic_weight (array-like): The topic weight.
        data (list): The preprocessed data.

        Returns:
        str: The generated summary.
        """
        sentence_scores = []
        for sentence in data:
            score = self._calculate_sentence_score(sentence, topic_weight)
            sentence_scores.append((sentence, score))
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        summary_sentences = [sentence for sentence, score in sentence_scores[:5]]
        summary =''.join(summary_sentences)
        return summary

    def _calculate_sentence_score(self, sentence, topic_weight):
        """
        Calculate the score for a given sentence.

        Parameters:
        sentence (list): The sentence tokens.
        topic_weight (array-like): The topic weight.

        Returns:
        float: The sentence score.
        """
        score = 0
        for token in sentence:
            score += topic_weight[token]
        return score
