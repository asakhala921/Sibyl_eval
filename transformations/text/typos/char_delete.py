from ..abstract_transformation import AbstractTransformation, _get_tran_types
import numpy as np
import string

class RandomCharDel(AbstractTransformation):
    """
    Deletes random chars
    """
    def __init__(self, task=None, meta=False):
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
        self.metadata = meta
        
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
        string : str
            The output with random chars deleted
        """
        original = string
        idx = sorted(np.random.choice(len(string), n, replace=False ))
        for i in idx:
            string = string[:i] + string[i+1:]
        assert type(string) == str
        meta = {'change': string!=original}
        if self.metadata: return string, meta
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
        if self.metadata: return X_[0], y_, X_[1]
        return X_, y_