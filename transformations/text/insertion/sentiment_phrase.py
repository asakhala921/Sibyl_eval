from ..abstract_transformation import AbstractTransformation
from ..data.phrases import POSITIVE_PHRASES, NEGATIVE_PHRASES
from random import sample 

class InsertSentimentPhrase(AbstractTransformation):
    """
    Appends a sentiment-laden phrase to a string based on 
    a pre-defined list of phrases.  
    """

    def __init__(self, sentiment='positive'):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        sentiment : str
            Determines whether the inserted phraase will 
            feature a positive or negative sentiment.
        """
        self.sentiment = sentiment
        if self.sentiment.lower() not in ['positive', 'negative']:
            raise ValueError("Sentiment must be 'positive' or 'negative'.")
    
    def __call__(self, string):
        """
        Appends a sentiment-laden phrase to a string.

        Parameters
        ----------
        string : str
            Input string

        Returns
        -------
        ret
            String with sentiment phrase appended
        """
        if 'positive' in self.sentiment:
        	phrase = sample(POSITIVE_PHRASES,1)[0]
        if 'negative' in self.sentiment:
        	phrase = sample(NEGATIVE_PHRASES,1)[0]
        ret = string + " " + phrase
        return ret