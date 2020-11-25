from ..abstract_transformation import AbstractTransformation
import numpy as np
import string

class RandomCharSubst(AbstractTransformation):
    """
    Substitues random chars
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

def get_random_letter():
    # printable = digits + ascii_letters + punctuation + whitespace
    return np.random.choice(list(string.printable))

