# Diversity Metrics
from text_diversity import TokenSemanticDiversity, SentenceSemanticDiversity, SyntacticDiversity
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from scipy import spatial
import torch
import numpy as np
from tqdm import tqdm
import seaborn as sns 
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.translate.bleu_score import SmoothingFunction
import os
import itertools
from multiprocessing import Pool
import spacy

sns.set("paper", 
    rc={"font.size":20,
        "axes.titlesize":25,
        "axes.labelsize":20,
        "lines.linewidth":2,
        "lines.markersize":5,
        "xtick.labelsize":14,
        "ytick.labelsize":14
        }) 

sns.set_style("white")
plt.rc("axes", labelweight="normal")

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_ttr_text(text):
    '''
    Type Token Ratio (TTR)
    higher -> more diversity

    Getting the TTR by text gives a better indication of the
    actual diversifying potential of the transformations on
    the text corpus. Tokenizer's typically strip out much of 
    the diversity in text to simplify the task for the model
    by replacing unknown words with [UNK] or other strategies
    '''
    tokens = [word_tokenize(x) for x in text]
    tokens = [item for sublist in tokens for item in sublist]
    toks, tok_counts = np.unique(tokens, return_counts=True)
    ttr = len(toks) / tok_counts.sum()
    return ttr

def get_ttr_tokens(text, tokenizer=None):
    '''
    Type Token Ratio (TTR)
    higher -> more diversity
    '''
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
    undesirable_tokens = [
        tokenizer.pad_token_id, 
        tokenizer.cls_token_id, 
        tokenizer.sep_token_id
    ]
    input_ids = tokenizer(text, return_tensors='pt', padding=True, truncation=True).input_ids
    token_ids, counts = np.unique(input_ids, return_counts=True)
    idx = np.isin(token_ids, undesirable_tokens, assume_unique=True, invert=True)
    token_ids, counts = token_ids[idx], counts[idx]
    ttr = len(token_ids) / counts.sum()
    return ttr

class TSD(TokenSemanticDiversity):

    use_me = True
    default_config = {}

    def __init__(self, config={}):
        config = {**super().default_config, **self.default_config, **config} 
        super().__init__(config)

    def __call__(self, response_set):
        return super().__call__(response_set)


class SSD(SentenceSemanticDiversity):

    use_me = True
    default_config = {}

    def __init__(self, config={}):
        config = {**super().default_config, **self.default_config, **config} 
        super().__init__(config)

    def __call__(self, response_set):
        return super().__call__(response_set)


class SD(SyntacticDiversity):

    use_me = True
    default_config = {}

    def __init__(self, config={}):
        config = {**super().default_config, **self.default_config, **config} 
        super().__init__(config)

    def __call__(self, response_set):
        return super().__call__(response_set)

class TSD_norm(TokenSemanticDiversity):

    use_me = True
    default_config = {'normalize': True}

    def __init__(self, config={}):
        config = {**super().default_config, **self.default_config, **config} 
        super().__init__(config)

    def __call__(self, response_set):
        return super().__call__(response_set)


class SSD_norm(SentenceSemanticDiversity):

    use_me = True
    default_config = {'normalize': True}

    def __init__(self, config={}):
        config = {**super().default_config, **self.default_config, **config} 
        super().__init__(config)

    def __call__(self, response_set):
        return super().__call__(response_set)


class SD_norm(SyntacticDiversity):

    use_me = True
    default_config = {'normalize': True}

    def __init__(self, config={}):
        config = {**super().default_config, **self.default_config, **config} 
        super().__init__(config)

    def __call__(self, response_set):
        return super().__call__(response_set)


class SelfBleu:
    def __init__(self, test_text='', gram=3):
        super().__init__()
        self.name = 'Self-Bleu'
        self.test_data = test_text
        self.gram = gram
        self.sample_size = 500
        self.reference = None
        self.is_first = True

    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            for text in self.test_data:
                toks = nltk.word_tokenize(text)
                reference.append(toks)
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        for hypothesis in self.test_data:
            hypothesis = nltk.word_tokenize(hypothesis)
            bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                                smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        sentence_num = len(reference)
        for index in range(sentence_num):
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            result.append(pool.apply_async(self.calc_bleu, args=(other, hypothesis, weight)))

        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt


def merge_bpe(tok, boe, chars="##"):
    new_tok = []
    new_boe = []

    emb = []
    append = ""
    for t, e in zip(tok[::-1], boe[::-1]):
        t += append
        emb.append(e)
        if t.startswith(chars):
            append = t.replace(chars, "")
        else:
            append = ""
            new_tok.append(t)
            new_boe.append(np.stack(emb).mean(axis=0))
            emb = []  
    new_tok = np.array(new_tok)[::-1]
    new_boe = np.array(new_boe)[::-1]
    
    return new_tok, new_boe

def find_max_list(lists):
    list_len = [len(l) for l in lists]
    return max(list_len)