from ..abstract_transformation import AbstractTransformation
import numpy as np

class RandomSwapQwerty(AbstractTransformation):
    """
    Substitues random chars
    """

    def __init__(self):
        """
        A transformation that swaps characters with adjacent keys on a
        QWERTY keyboard, replicating the kind of errors that come from typing
        too quickly.

        Parameters
        ----------
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.task = task
        self.keyboard_adjacency = {
            "q": [
                "w",
                "a",
                "s",
            ],
            "w": ["q", "e", "a", "s", "d"],
            "e": ["w", "s", "d", "f", "r"],
            "r": ["e", "d", "f", "g", "t"],
            "t": ["r", "f", "g", "h", "y"],
            "y": ["t", "g", "h", "j", "u"],
            "u": ["y", "h", "j", "k", "i"],
            "i": ["u", "j", "k", "l", "o"],
            "o": ["i", "k", "l", "p"],
            "p": ["o", "l"],
            "a": ["q", "w", "s", "z", "x"],
            "s": ["q", "w", "e", "a", "d", "z", "x"],
            "d": ["w", "e", "r", "f", "c", "x", "s"],
            "f": ["e", "r", "t", "g", "v", "c", "d"],
            "g": ["r", "t", "y", "h", "b", "v", "d"],
            "h": ["t", "y", "u", "g", "j", "b", "n"],
            "j": ["y", "u", "i", "k", "m", "n", "h"],
            "k": ["u", "i", "o", "l", "m", "j"],
            "l": ["i", "o", "p", "k"],
            "z": ["a", "s", "x"],
            "x": ["s", "d", "z", "c"],
            "c": ["x", "d", "f", "v"],
            "v": ["c", "f", "g", "b"],
            "b": ["v", "g", "h", "n"],
            "n": ["b", "h", "j", "m"],
            "m": ["n", "j", "k"],
        }
    
    def __call__(self, text, n=1):
        """
        Parameters
        ----------
        word : str
            The input string
        n : int
            The number of random char Substitutions to perform

        Returns
        ----------
        ret : str
            The output with random Substitutions
        """
        assert n <= len(text), "n is too large. n should be <= "+str(len(text))
        idx = sorted(np.random.choice(len(text), n, replace=False ))
        for i in idx:
            text = text[:i] + get_adjacent_letter(text[i]) + text[i + 1 :]
        return text

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

def get_adjacent_letter(s):
    s_lower = s.lower()
    if s_lower in self.keyboard_adjacency:
        adjacent_keys = self.keyboard_adjacency[s_lower]
        if s.isupper():
            ans = [key.upper() for key in adjacent_keys]
        else:
            ans = adjacent_keys
    else:
        return s
    return np.random.choice(ans)