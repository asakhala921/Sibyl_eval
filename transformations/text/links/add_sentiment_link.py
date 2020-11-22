from ..abstract_transformation import AbstractTransformation
import pandas as pd

class AddSentimentLink(AbstractTransformation):
    """
    Appends a given / constructed URL to a string input.
    Current implementation constructs a default URL that
    makes use of dictionary.com and is sensitive to changes
    in routing structure. 
    """

    def __init__(self, url=None, sentiment='positive'):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        url : str
            The URL to append to an input string

        sentiment : str
            Determines whether the constructed URL will 
            feature a positive or negative sentiment.
        """
        self.url = url
        if self.url is None:
            self.url = 'https://www.dictionary.com/browse/'
            self.default_url = True
        else:
            self.default_url = False
        self.sentiment = sentiment
        if self.sentiment.lower() not in ['positive', 'negative']:
            raise ValueError("Sentiment must be 'positive' or 'negative'.")
        # https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
        self.pos_words = pd.read_csv('../data/opinion-lexicon-English/positive-words.txt', skiprows=30, names=['word'], encoding='latin-1')
        self.neg_words = pd.read_csv('../data/opinion-lexicon-English/negative-words.txt', skiprows=30, names=['word'], encoding='latin-1')
    
    def __call__(self, string):
        """
        Appends a given / constructed URL to a string input.

        Parameters
        ----------
        string : str
            Input string

        Returns
        -------
        ret
            String with URL appended
        """
        if self.default_url:
            if 'positive' in self.sentiment:
                word = self.pos_words.sample(1)['word'].iloc[0]
            if 'negative' in self.sentiment:
                word = self.neg_words.sample(1)['word'].iloc[0]
            link = 'https://www.dictionary.com/browse/' + word
        else:
            link = self.url
        ret = string + ' ' + link
        return ret