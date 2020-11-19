from ..abstract_transformation import AbstractTransformation
import random
import numpy as np

class HomoglyphSwap(AbstractTransformation):
    """Transforms an input by replacing its words with visually similar words
    using homoglyph swaps."""
    def __init__(self, random_one=False, all=False):
        """
        Initializes the transformation

        Parameters
        ----------
        random_one : boolean
            Whether to swap a random char or not.
            False by default
            WARNING setting it to true can be dangerous coz it 
            might return empty list
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
        self.random_one = random_one
        self.all= all
    
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
        candidate_words = []
        
        if self.all:
            temp = string # deep coppy apparently 
            for i in range(len(string)):
                if string[i] in self.homos:
                    repl_letter = self.homos[string[i]]
                    temp = temp[:i] + repl_letter + string[i + 1 :]
            return temp

        if self.random_one:
            i = np.random.randint(0, len(string))
            if string[i] in self.homos:
                repl_letter = self.homos[string[i]]
                candidate_word = string[:i] + repl_letter + string[i + 1 :]
                candidate_words.append(candidate_word)
        else:
            for i in range(len(string)):
                if string[i] in self.homos:
                    repl_letter = self.homos[string[i]]
                    candidate_word = string[:i] + repl_letter + string[i + 1 :]
                    candidate_words.append(candidate_word)

        return candidate_words