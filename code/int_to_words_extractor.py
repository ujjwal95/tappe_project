from sklearn.base import BaseEstimator, TransformerMixin

class NumberStringExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts string text column, outputs number to string"""

    def __init__(self):
        pass
    
    def transform_to_word(self, name):
        if isinstance(name, int):
            import inflect
            p = inflect.engine()
            name = p.number_to_words(name)
            return name
        else:
            return name
        
    def transform(self, df, y=None):
        import inflect
        import numpy as np
        """The workhorse of this feature extractor"""
        p = inflect.engine()
        return np.array([self.transform_to_word(x) for x in df]).reshape(-1, 1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


