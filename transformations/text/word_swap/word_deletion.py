from ..abstract_transformation import AbstractTransformation, _get_tran_types
import random

class WordDeletion(AbstractTransformation):
    """
    Deletes words from random indices in the string input
    """

    def __init__(self, p=0.25, task=None):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        p : float
            Randomly delete words from the sentence with probability p
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.p = p
        self.task = task
    
    def __call__(self, string):
        """
        Parameters
        ----------
        string : str
            The input string

        Returns
        ----------
        ret : str
            The output with random words deleted
        """
        words = string.split()
        # obviously, if there's only one word, don't delete it
        if len(words) == 1:
            return words

        #randomly delete words with probability p
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > self.p:
                new_words.append(word)

        #if you end up deleting all words, just return a random word
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            return words[rand_int]

        return " ".join(new_words)

    def get_tran_types(self, task_name=None, tran_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['INV', 'INV']
        }
        df = _get_tran_types(self.tran_types, task_name, tran_type)
        return df

    def transform_Xy(self, X, y):
        X_ = self(X)
        tran_type = self.get_tran_types(task_name=self.task)['tran_type'][0]
        if tran_type == 'INV':
            y_ = y
        if tran_type == 'SIB':
            y_ = 0 if y == 1 else 1
        return X_, y_