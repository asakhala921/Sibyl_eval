from ..abstract_transformation import AbstractTransformation, _get_tran_types
import numpy as np
import re
import random
from pattern.en import wordnet
from nltk.corpus import stopwords
import spacy
import en_core_web_sm

class ChangeSynse(AbstractTransformation):
    """
    Replaces a specified number of random words a string
    with synses from wordnet. Also supports part-of-speech (pos)
    tagging via spaCy to get more natural replacements. 
    """
    def __init__(self, synse='synonym', num_to_replace=-1, task=None, meta=False):
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
            to replace with synonyms (excludes stop words).
            If -1, then replace as many as possible.
        task : str
            the type of task you wish to transform the
            input towards
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
        self.num_to_replace = np.Inf if num_to_replace == -1 else num_to_replace
        self.nlp = en_core_web_sm.load()
        # stopwords 
        useful_stopwords = ['few', 'more', 'most', 'against']
        self.stopwords = [w for w in stopwords.words('english') if w not in useful_stopwords]
        self.task = task
        self.metadata = meta
    
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
            [token.text for token in doc if strip_punct(token.text.lower()) not in self.stopwords]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            pos = [token.pos_ for token in doc if random_word in token.text][0]
            if pos in ['PROPN']:
                continue
            if pos not in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                pos = None
            options = self.synse_fn(strip_punct(random_word), pos)
            if len(options) >= 1:
                option = random.choice(list(options))
                new_words = [option if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= self.num_to_replace:
                break
        ret = ' '.join(new_words)
        assert type(ret) == str
        meta = {'change': string!=ret}
        if self.metadata: return ret, meta
        return ret

    def get_tran_types(self, task_name=None, tran_type=None):
        print("Not implemented.")
        pass

    def transform_Xy(self, X, y):
        X_ = self(X)
        tran_type = self.get_tran_types(task_name=self.task)['tran_type'][0]
        if tran_type == 'INV':
            y_ = y
        if tran_type == 'SIB':
            if self.synse == 'antonym':
                y_ = 0 if y == 1 else 1
            else:
                y_ = y
        if self.metadata: return X_[0], y_, X_[1]
        return X_, y_

class ChangeSynonym(ChangeSynse):
    def __init__(self, num_to_replace=-1, meta=False):
        super().__init__(synse='synonym', num_to_replace=num_to_replace ,meta=meta)
        self.metadata = meta 

    def __call__(self, string):
        return super().__call__(string)

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
            y_ = y
        if self.metadata: return X_[0], y_, X_[1]
        return X_, y_

class ChangeAntonym(ChangeSynse):
    def __init__(self, num_to_replace=-1, meta=False):
        super().__init__(synse='antonym', num_to_replace=num_to_replace,meta=meta)
        self.metadata = meta

    def __call__(self, string):
        return super().__call__(string)

    def get_tran_types(self, task_name=None, tran_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['SIB', 'INV']
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
        if self.metadata: return X_[0], y_, X_[1]
        return X_, y_

class ChangeHyponym(ChangeSynse):
    def __init__(self, num_to_replace=-1, meta=False):
        super().__init__(synse='hyponym', num_to_replace=num_to_replace,meta=meta) 
        self.metadata = meta

    def __call__(self, string):
        return super().__call__(string)

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
            y_ = y
        if self.metadata: return X_[0], y_, X_[1]
        return X_, y_

class ChangeHypernym(ChangeSynse):
    def __init__(self, num_to_replace=-1, meta=False):
        super().__init__(synse='hypernym', num_to_replace=num_to_replace,meta=meta) 
        self.metadata = meta

    def __call__(self, string):
        return super().__call__(string)

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
            y_ = y
        if self.metadata: return X_[0], y_, X_[1]
        return X_, y_

def all_synsets(word, pos=None):
    pos_map = {
        'NOUN': wordnet.NOUN,
        'VERB': wordnet.VERB,
        'ADJ': wordnet.ADJECTIVE,
        'ADV': wordnet.ADVERB
        }
    if pos is None:
        pos_list = [wordnet.VERB, wordnet.ADJECTIVE, wordnet.NOUN, wordnet.ADVERB]
    else:
        pos_list = [pos_map[pos]]
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
    if not ret:
        synos = all_possible_synonyms(word, pos)
        for syno in synos:
            for syn in all_synsets(syno, pos=pos):
                if not syn.antonym:
                    continue
                for s in syn.antonym:
                    ret.extend(s.senses)
    return clean_senses(ret)

def all_possible_hypernyms(word, pos=None, depth=3):
    ret = []
    for syn in all_synsets(word, pos=pos):
        ret.extend([y for x in syn.hypernyms(recursive=True, depth=depth) for y in x.senses])
    return clean_senses(ret)

def all_possible_hyponyms(word, pos=None, depth=3):
    ret = []
    for syn in all_synsets(word, pos=pos):
        ret.extend([y for x in syn.hyponyms(recursive=True, depth=depth) for y in x.senses])
    return clean_senses(ret)

def strip_punct(word):
    puncts = re.compile(r'[^\w\s]')
    return puncts.sub('', word)