from ..abstract_transformation import AbstractTransformation, _get_tran_types
import numpy as np
import string

class RandomCharSubst(AbstractTransformation):
    """
    Substitues random chars
    """
    def __init__(self, task=None):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.task = task
    
    def __call__(self, text, n=1):
        """
        Parameters
        ----------
        word : str
            The input string
        n : int
            The number of random Substitutions to perform

        Returns
        ----------
        ret : str
            The output with random Substitutions
        """
        assert n <= len(text), "n is too large. n should be <= "+str(len(text))
        idx = sorted(np.random.choice(len(text), n, replace=False ))
        temp = text
        for i in idx:
            temp = temp[:i] + get_random_letter() + temp[i + 1 :]
        return temp

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

def get_random_letter():
    # printable = digits + ascii_letters + punctuation + whitespace
    return np.random.choice(list(string.printable))

