from ..abstract_transformation import AbstractTransformation, _get_tran_types
import numpy as np
import string

class RandomCharSwap(AbstractTransformation):
    """
    Swaps random consecutive chars
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
        
    def __call__(self, string, n=1):
        """
        Parameters
        ----------
        string : str
            The input string
        n : int
            Number of chars to be transformed

        Returns
        ----------
        ret : str
            The output with random chars pairs swapped
        """
        idx = sorted(np.random.choice(len(string)-1, n, replace=False ))
        for i in idx:
            string = string[:i] + string[i + 1] + string[i] + string[i + 2 :]
        assert type(string) == str
        return string

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