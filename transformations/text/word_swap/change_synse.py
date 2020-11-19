from ..abstract_transformation import AbstractTransformation
import re
import random
from pattern.en import wordnet
from pattern.text.en.wordlist import STOPWORDS
import spacy
import en_core_web_sm

class ChangeSynse(AbstractTransformation):
    """
    Replaces a specified number of random words a string
    with synses from wordnet. Also supports part-of-speech (pos)
    tagging via spaCy to get more natural replacements. 
    """
    def __init__(self, synse='synonym', num_to_replace=1):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        synse : str
            The type of word replacement to make.
            Currently support 'synonym', 'antonym', 'hyponym', 
            and 'hypernym' 
        num_to_replace : int
            The number of words randomly selected from the input 
            to replace with synonyms (excludes stop words)
        """
        self.synse = synse
        self.synses = {
            'synonym' : all_possible_synonyms,
            'antonym' : all_possible_antonyms,
            'hyponym' : all_possible_hyponyms,
            'hypernym' : all_possible_hypernyms,
        }
        if synse not in self.synses:
            raise ValueError("Invalid synse argument. \n \
                Please select one of the following: 'synonym', 'antonym', 'hyponym', 'hypernym'")
        self.synse_fn = self.synses[self.synse]
        self.num_to_replace = num_to_replace
        self.nlp = en_core_web_sm.load()
    
    def __call__(self, string):
        """Replaces words with synses

        Parameters
        ----------
        string : str
            Input string

        Returns
        -------
        ret : str
            Output string with synses replaced.
        """
        doc = self.nlp(string)
        new_words = string.split(' ').copy()
        random_word_list = list(set(
            [word for word in new_words if strip_punct(word.lower()) not in STOPWORDS]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            idx = new_words.index(random_word)
            pos = doc[idx].pos_
            options = self.synse_fn(strip_punct(random_word), pos)
            if len(options) >= 1:
                option = random.choice(list(options))
                new_words = [option if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= self.num_to_replace:
                break
        ret = ' '.join(new_words)
        return ret

def all_synsets(word, pos=None):
    map = {
        'NOUN': wordnet.NOUN,
        'VERB': wordnet.VERB,
        'ADJ': wordnet.ADJECTIVE,
        'ADV': wordnet.ADVERB
        }
    if pos is None:
        pos_list = [wordnet.VERB, wordnet.ADJECTIVE, wordnet.NOUN, wordnet.ADVERB]
    else:
        pos_list = [map[pos]]
    ret = []
    for pos in pos_list:
        ret.extend(wordnet.synsets(word, pos=pos))
    return ret

def clean_senses(synsets):
    return [x for x in set(synsets) if '_' not in x]

def all_possible_synonyms(word, pos=None):
    ret = []
    for syn in all_synsets(word, pos=pos):
        # if syn.synonyms[0] != word:
        #     continue
        ret.extend(syn.senses)
    return clean_senses(ret)

def all_possible_antonyms(word, pos=None):
    ret = []
    for syn in all_synsets(word, pos=pos):
        if not syn.antonym:
            continue
        for s in syn.antonym:
            ret.extend(s.senses)
    return clean_senses(ret)

def all_possible_hypernyms(word, pos=None, depth=None):
    ret = []
    for syn in all_synsets(word, pos=pos):
        ret.extend([y for x in syn.hypernyms(recursive=True, depth=depth) for y in x.senses])
    return clean_senses(ret)

def all_possible_hyponyms(word, pos=None, depth=None):
    ret = []
    for syn in all_synsets(word, pos=pos):
        ret.extend([y for x in syn.hyponyms(recursive=True, depth=depth) for y in x.senses])
    return clean_senses(ret)

def strip_punct(word):
    puncts = re.compile(r'[^\w\s]')
    return puncts.sub('', word)