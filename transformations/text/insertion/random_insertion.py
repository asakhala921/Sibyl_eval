from ..abstract_transformation import AbstractTransformation, _get_tran_types
import numpy as np
import random
from ..word_swap.change_synse import all_possible_synonyms

class RandomInsertion(AbstractTransformation):
    """
    Inserts random words
    """

    def __init__(self, n=1, task=None):
        """
        Initializes the transformation

        Parameters
        ----------
        n : int
            The number of random insertions to perform
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.n=n
        self.task=task
    
    def __call__(self, words):
        """
        Parameters
        ----------
        word : str
            The input string
        n : int
            Number of word insertions

        Returns
        ----------
        ret : str
            The output with random words inserted
        """
        new_words = words.split()
        for _ in range(self.n):
            add_word(new_words)
        return ' '.join(new_words)

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

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = all_possible_synonyms(random_word) #get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)


