from ..abstract_transformation import AbstractTransformation, _get_tran_types
import numpy as np
import random
from ..word_swap.change_synse import all_possible_synonyms

class RandomInsertion(AbstractTransformation):
    """
    Inserts random words
    """

    def __init__(self, n=1):
        """
        Initializes the transformation

        Parameters
        ----------
        n : int
            The number of random insertions to perform
        """
        self.n=n
    
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


