from sklearn.base import BaseEstimator, TransformerMixin
import spacy
import numpy as np

class NERExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts string text column, outputs list of NERs"""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        pass

    def ners(self, name):
        nlp = self.nlp
        name = nlp(name)
        ner = {'PERSON': 0, 'NORP': 0, 'FAC': 0, 'ORG': 0, 'GPE': 0,
                'LOC': 0, 'PRODUCT': 0, 'EVENT': 0, 'WORK_OF_ART': 0,
                'LAW': 0, 'LANGUAGE': 0, 'DATE': 0, 'TIME': 0, 'PERCENT': 0, 
                'MONEY': 0, 'QUANTITY': 0, 'ORDINAL': 0, 'CARDINAL': 0}

        for ent in name.ents:
            ner[ent.label_] += 1
        return np.array(list(ner.values()))
    
    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return np.array([self.ners(x) for x in df]).reshape(-1, 18)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self



