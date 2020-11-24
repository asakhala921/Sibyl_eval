from ..abstract_transformation import AbstractTransformation
import numpy as np
import random
from ..word_swap.change_synse import all_possible_synonyms

class RandomInsertion(AbstractTransformation):
    """
    Inserts random words
    """

    def __init__(self, n=1):
        """
        Initializes the transformation

        Parameters
        ----------
        """
        self.n=n
    
    def __call__(self, words):
        """
        Parameters
        ----------
        word : str
            The input string
        n : int
            Number of word insertions

        Returns
        ----------
        ret : str
            The output with random words inserted
        """
        new_words = words.split()
        for _ in range(self.n):
            add_word(new_words)
        return ' '.join(new_words)

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = all_possible_synonyms(random_word) #get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

# from nltk.corpus import wordnet
# def get_synonyms(word):
#     synonyms = set()
#     for syn in wordnet.synsets(word): 
#         for l in syn.lemmas():
#             synonym = l.name().replace("_", " ").replace("-", " ").lower()
#             synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
#             synonyms.add(synonym) 
#     if word in synonyms:
#         synonyms.remove(word)
#     return list(synonyms)



    # name, location, synse
    # word_embedding
    # extensionmap and its inverse contractions
    # gradient based
    # swap masked im
    # typos