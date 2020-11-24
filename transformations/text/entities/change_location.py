from ..abstract_transformation import AbstractTransformation
import numpy as np
import en_core_web_sm
from .data import NAMED_ENTITIES

class ChangeLocation(AbstractTransformation):
    """
    Changes person names
    """

    def __init__(self):
        """Transforms an input by replacing names of recognized name entity.
        :param first_only: Whether to change first name only
        :param last_only: Whether to change last name only
        """
        self.nlp = en_core_web_sm.load()
    
    def __call__(self, words):
        """
        Parameters
        ----------
        word : str
            The input string

        Returns
        ----------
        ret : str
            The output with person names replaced
        """
        doc = self.nlp(words)
        newString = words
        for e in reversed(doc.ents): #reversed to not modify the offsets of other entities when substituting
            start = e.start_char
            end = start + len(e.text)
            # print(e.text, "label is ", e.label_)
            if e.label_ in ('GPE', 'NORP', 'FAC', 'ORG', 'LOC'):
                name = newString[start:end]
                name = name.split()
                name[0] =  self._get_loc_name()
                name = " ".join(name)
                newString = newString[:start] + name + newString[end:]
        return newString

    def _get_loc_name(self):
        """Return a random location name."""
        loc = np.random.choice(['country', 'nationality', 'city'])
        return np.random.choice(NAMED_ENTITIES[loc])