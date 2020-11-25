from ..abstract_transformation import AbstractTransformation
import numpy as np
import string

class RandomCharSwap(AbstractTransformation):
    """
    Swaps random consecutive chars
    """
    def __init__(self):
        """
        Initializes the transformation
        """
        pass
    def __call__(self, text, n=1):
        """
        Parameters
        ----------
        word : str
            The input string
        n : int
            Number of chars to be transformed

        Returns
        ----------
        ret : str
            The output with random chars pairs swapped
        """
        idx = sorted(np.random.choice(len(text)-1, n, replace=False ))
        for i in idx:
            text = text[:i] + text[i + 1] + text[i] + text[i + 2 :]
        return text