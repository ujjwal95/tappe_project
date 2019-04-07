from sklearn.base import BaseEstimator, TransformerMixin

class GreetingExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts string text column, outputs whether it is greeting or not"""

    def __init__(self):
        pass

    def greeting_word(self, name):
        """Helper code to check whether name is greeting"""
        greetings = ["hey", "hi", "how's it going", "how are you doing", "what’s up", 
                     "what’s new", "what’s going on", "how’s everything", "how are things", 
                     "how’s life", "how’s your day", "how’s your day going", 
                     "good to see you",  "nice to see you", "long time no see", "it’s been a while", 
                     "good morning", "good afternoon", "good evening", "it’s nice to meet you", 
                     "pleased to meet you", "how have you been", "how do you do", "are you ok", 
                     "you alright", "alright mate", "howdy", "sup",  "whazzup", "hiya", "i'm",
                     "my name is", "i am","i'm based in", "i am based in", "i have been with", 
                    "i've been with", "i'm sorry", "excuse me", "repeat", "nice to meet you", 
                     "help you"]
        
        return any(substring in name for substring in greetings)
        
    def transform(self, df, y=None):
        import numpy as np
        """The workhorse of this feature extractor"""
        return np.array([self.greeting_word(x) for x in df]).reshape(-1, 1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


