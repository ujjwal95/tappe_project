from sklearn.base import BaseEstimator, TransformerMixin

class QuestionExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts string text column, outputs whether it is question or not"""

    def __init__(self):
        pass

    def question_word(self, name):
        """Helper code to check whether name is question"""
        start_words = ['who', 'what', 'when', 'where', 'why', 
                       'how', 'is', 'can', 'does', 'do', 'WHICH', 
                       'AM', 'ARE', 'WAS', 'WERE', 'MAY', 'MIGHT', 
                       'CAN', 'COULD', 'WILL', 'SHALL', 'WOULD', 
                       'SHOULD', 'HAS', 'HAVE', 'HAD', 'DID']
        start_words = [x.lower() for x in start_words]
        try:
            if (name.split()[0] in start_words) or (name.split()[1] in start_words):
                return 1
            else:
                return 0
        
        except:
            if (name in start_words):
                return 1
            else:
                return 0
        
    def transform(self, df, y=None):
        import numpy as np
        """The workhorse of this feature extractor"""
        return np.array([self.question_word(x) for x in df]).reshape(-1, 1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

