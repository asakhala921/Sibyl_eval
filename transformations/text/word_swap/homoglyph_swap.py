from ..abstract_transformation import AbstractTransformation, _get_tran_types
import random
import numpy as np

class HomoglyphSwap(AbstractTransformation):
    """
    Transforms an input by replacing its words with 
    visually similar words using homoglyph swaps.
    """
    def __init__(self, change=0.25, task=None):
        """
        Initializes the transformation

        Parameters
        ----------
        change: float
            tells how many of the charachters in string 
            to possibly replace
            warning: it will check change % or charachters 
            for possible replacement
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.homos = {
            "-": "˗",
            "9": "৭",
            "8": "Ȣ",
            "7": "𝟕",
            "6": "б",
            "5": "Ƽ",
            "4": "Ꮞ",
            "3": "Ʒ",
            "2": "ᒿ",
            "1": "l",
            "0": "O",
            "'": "`",
            "a": "ɑ",
            "b": "Ь",
            "c": "ϲ",
            "d": "ԁ",
            "e": "е",
            "f": "𝚏",
            "g": "ɡ",
            "h": "հ",
            "i": "і",
            "j": "ϳ",
            "k": "𝒌",
            "l": "ⅼ",
            "m": "ｍ",
            "n": "ո",
            "o": "о",
            "p": "р",
            "q": "ԛ",
            "r": "ⲅ",
            "s": "ѕ",
            "t": "𝚝",
            "u": "ս",
            "v": "ѵ",
            "w": "ԝ",
            "x": "×",
            "y": "у",
            "z": "ᴢ",
        }
        assert(0<=change<=1)
        self.change = change
        self.task = task
    
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
        """Returns a list containing all possible words with 1 character
        replaced by a homoglyph."""

        # possibly = [k for k,j in enumerate(string) if j in self.homos]
        # indices = list(np.random.choice(possibly, int(np.ceil(self.change*len(string))), replace=False) )
                # try, catch ValueError ? safer option
        indices = np.random.choice(len(string), int(np.ceil(self.change*len(string))), replace=False)
        
        temp = string # deep coppy apparently 
        for i in sorted(indices):
            if string[i] in self.homos:
                repl_letter = self.homos[string[i]]
                temp = temp[:i] + repl_letter + string[i+1:]
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