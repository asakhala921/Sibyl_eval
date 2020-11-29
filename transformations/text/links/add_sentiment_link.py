from ..abstract_transformation import AbstractTransformation, _get_tran_types
import pandas as pd
import os

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
        cur_path = os.path.dirname(os.path.realpath(__file__))
        pos_path = os.path.join(cur_path, '../data/opinion-lexicon-English/positive-words.txt')
        neg_path = os.path.join(cur_path, '../data/opinion-lexicon-English/negative-words.txt')
        self.pos_words = pd.read_csv(pos_path, skiprows=30, names=['word'], encoding='latin-1')
        self.neg_words = pd.read_csv(neg_path, skiprows=30, names=['word'], encoding='latin-1')
    
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

    def get_tran_types(self, task_name=None, tran_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['SIB', 'INV']
        }
        df = _get_tran_types(self.tran_types, task_name, tran_type)
        return df

class AddPositiveLink(AddSentimentLink):
    def __init__(self):
        super().__init__(url=None, sentiment='positive')
    def __call__(self, string):
        if self.default_url:
            word = self.pos_words.sample(1)['word'].iloc[0]
            link = 'https://www.dictionary.com/browse/' + word
        else:
            link = self.url
        ret = string + ' ' + link
        return ret

    def get_tran_types(self, task_name=None, tran_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['SIB', 'INV']
        }
        df = _get_tran_types(self.tran_types, task_name, tran_type)
        return df

class AddNegativeLink(AddSentimentLink):
    def __init__(self):
        super().__init__(url=None, sentiment='negative')
    def __call__(self, string):
        if self.default_url:
            word = self.neg_words.sample(1)['word'].iloc[0]
            link = 'https://www.dictionary.com/browse/' + word
        else:
            link = self.url
        ret = string + ' ' + link
        return ret

    def get_tran_types(self, task_name=None, tran_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['SIB', 'INV']
        }
        df = _get_tran_types(self.tran_types, task_name, tran_type)
        return df