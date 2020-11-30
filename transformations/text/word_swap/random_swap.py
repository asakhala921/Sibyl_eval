from ..abstract_transformation import AbstractTransformation, _get_tran_types
import numpy as np
import random
import re

class RandomSwap(AbstractTransformation):
    """
    Swaps random words
    """

    def __init__(self, n=1, task=None):
        """
        Initializes the transformation

        Parameters
        ----------
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.n=n
        self.task=task
    
    def __call__(self, string):
        """
        Parameters
        ----------
        string : str
            The input string
        n : int
            Number of word swaps

        Returns
        ----------
        ret : str
            The output with random words swapped
        """
        new_words = (string).split()
        for _ in range(self.n):
            new_words = swap_word(new_words)
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

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words