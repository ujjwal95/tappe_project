from sklearn.base import BaseEstimator, TransformerMixin
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from afinn import Afinn

class SentimentExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts string text column, outputs sentiment from 2 packages"""

    def __init__(self):
        self.afinn = Afinn()
        self.sid = SentimentIntensityAnalyzer()
        pass

    def transform(self, df, y=None):
        import numpy as np
        afinn = self.afinn
        sid = self.sid
        """The workhorse of this feature extractor"""
        sent_frm_vader = np.array([np.array(list(sid.polarity_scores(x).values())) for x in df])
        sent_frm_afinn = np.array([afinn.score(x) for x in df]).reshape(-1, 1)
        return np.concatenate((sent_frm_vader, sent_frm_afinn), axis = 1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


