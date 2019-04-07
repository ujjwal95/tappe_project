from sklearn.base import BaseEstimator, TransformerMixin

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts string text column, outputs after processing"""

    def __init__(self):
        pass

    def transform(self, df, y=None):
        import numpy as np
        from text_processor import TextPreprocessing
        from itertools import groupby
        """The workhorse of this feature extractor"""
        stop_words = ['um','a','the','uh','an']
        return np.array([TextPreprocessing(x).apply_contractions().lower_case().process_html().remove_urls().decode_text().remove_stutterings().remove_short_words().stopwords_remove(stopwords = stop_words).lemmatize().text for x in df]).reshape(-1, 1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


