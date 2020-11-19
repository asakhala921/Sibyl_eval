from ..abstract_transformation import AbstractTransformation
import numpy as np
import re
import spacy
import en_core_web_sm

class ChangeNumber(AbstractTransformation):
    """
    Contracts all known contractions in a string input or 
    returns the original string if none are found. 
    """

    def __init__(self, multiplier=0.2, replacement=None):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        multiplier : float
            The value by which all numbers in the input
            string will be multiplied

        repalcement : float
            The value by which all numbers in the input
            string will be replaced (overrides multiplier)
        """
        self.multiplier = multiplier
        self.replacement = replacement
        self.nlp = en_core_web_sm.load()
    
    def __call__(self, string):
        """Contracts contractions in a string (if any)

        Parameters
        ----------
        string : str
            Input string

        Returns
        -------
        ret
            String with contractions expanded (if any)
        """
        doc = self.nlp(string)
        nums = [x.text for x in doc if x.text.isdigit()]
        ret = []
        for x in nums:
            # e.g. this is 4 you
            if x == '2' or x == '4':
                continue
            if self.replacement is None:
                change = int(int(x) * self.multiplier)
            else:
                change = self.replacement
            sub_re = re.compile(r'\b%s\b' % x)
            ret = sub_re.sub(str(change), doc.text)
        return ret