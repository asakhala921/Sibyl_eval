from ..abstract_transformation import AbstractTransformation
from emoji_translate import Translator

class Emojify(AbstractTransformation):
    def __init__(self, exact_match_only=False, randomize=True):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed
        
        Parameters
        ----------
        exact_match_only : boolean
            Determines whether we find exact matches for
            the emoji's name for replacement. If false,
            approximate matching is used. 
        randomize : boolean
            If true, randomizes approximate matches.
            If false, always picks the first match.
        """
        self.exact_match_only = exact_match_only
        self.randomize = randomize
        self.emo = Translator(self.exact_match_only, self.randomize)

    def __call__(self, string):
        """
        Parameters
        ----------
        string : str
            The string input

        Returns
        ----------
        ret : str
            The output with as many non-stopwords translated
            to emojis as possible.
        """
        return self.emo.emojify(string)