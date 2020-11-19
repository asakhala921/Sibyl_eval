from ..abstract_transformation import AbstractTransformation
import random

class WordDeletion(AbstractTransformation):
    """
    Deletes words from random indices in the string input
    """

    def __init__(self, num_to_delete=1):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        num_to_delete : int
            The number of words to delete at random
        """
        self.num_to_delete = num_to_delete
    
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
        string_list = string.split(' ') # returns a list
        for _ in range(self.num_to_delete):
            idx = random.randint(0, len(string_list)-1)
            del string_list[idx]
        ret = ' '.join(string_list)
        return ret