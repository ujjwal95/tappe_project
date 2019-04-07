from sklearn.base import BaseEstimator, TransformerMixin

class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts string text column, outputs average word length"""

    def __init__(self):
        pass

    def average_word_length(self, name):
        import numpy as np
        """Helper code to compute average word length of a name"""
        try:
            return np.mean([len(word) for word in name.split()])
        except:
            return len(str(name))
        

    def transform(self, df, y=None):
        import numpy as np
        """The workhorse of this feature extractor"""
        avg_str_len = np.array([self.average_word_length(x) for x in df]).reshape(-1, 1)
        return np.concatenate((df.reshape(-1, 1), avg_str_len), axis = 1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
