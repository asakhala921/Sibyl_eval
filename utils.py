import os
import pickle
import numpy as np
import pandas as pd

from transformations.text.contraction.expand_contractions import ExpandContractions
from transformations.text.contraction.contract_contractions import ContractContractions
from transformations.text.emoji.emojify import Emojify, AddPositiveEmoji, AddNegativeEmoji, AddNeutralEmoji
from transformations.text.emoji.demojify import Demojify, RemovePositiveEmoji, RemoveNegativeEmoji, RemoveNeutralEmoji
from transformations.text.negation.remove_negation import RemoveNegation
from transformations.text.negation.add_negation import AddNegation
from transformations.text.contraction.expand_contractions import ExpandContractions
from transformations.text.contraction.contract_contractions import ContractContractions
from transformations.text.word_swap.change_number import ChangeNumber
from transformations.text.word_swap.change_synse import ChangeSynonym, ChangeAntonym, ChangeHyponym, ChangeHypernym
from transformations.text.word_swap.word_deletion import WordDeletion
from transformations.text.word_swap.homoglyph_swap import HomoglyphSwap
from transformations.text.word_swap.random_swap import RandomSwap
from transformations.text.insertion.random_insertion import RandomInsertion
from transformations.text.insertion.sentiment_phrase import InsertSentimentPhrase, InsertPositivePhrase, InsertNegativePhrase
from transformations.text.links.add_sentiment_link import AddSentimentLink, AddPositiveLink, AddNegativeLink
from transformations.text.links.import_link_text import ImportLinkText
from transformations.text.entities.change_location import ChangeLocation
from transformations.text.entities.change_name import ChangeName
from transformations.text.typos.char_delete import RandomCharDel
from transformations.text.typos.char_insert import RandomCharInsert
from transformations.text.typos.char_substitute import RandomCharSubst
from transformations.text.typos.char_swap import RandomCharSwap
from transformations.text.typos.char_swap_qwerty import RandomSwapQwerty 
from transformations.text.mixture.text_mix import TextMix, SentMix, WordMix

TRANSFORMATIONS = [
    ExpandContractions,
    ContractContractions,
    Emojify,
    AddPositiveEmoji,
    AddNegativeEmoji,
    AddNeutralEmoji,
    Demojify, 
    RemovePositiveEmoji,
    RemoveNegativeEmoji,
    RemoveNeutralEmoji,
    ChangeLocation,
    ChangeName,
    InsertPositivePhrase,
    InsertNegativePhrase,
    RandomInsertion,
    AddPositiveLink,
    AddNegativeLink,
    ImportLinkText,
    AddNegation,
    RemoveNegation,
    RandomCharDel,
    RandomCharInsert, 
    RandomCharSubst, 
    RandomCharSwap, 
    RandomSwapQwerty,
    ChangeNumber,
    ChangeSynonym, 
    ChangeAntonym, 
    ChangeHyponym, 
    ChangeHypernym,
    WordDeletion, 
    HomoglyphSwap, 
    RandomSwap, 
    TextMix, 
    SentMix, 
    WordMix
]

def pkl_save(file, path):
    base_path = os.path.dirname(path)
    os.makedirs(base_path, exist_ok=True)
    with open(path, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def pkl_load(path):
    with open(path, 'rb') as handle:
        file = pickle.load(handle)
    return file

def npy_save(path, file):
    base_path = os.path.dirname(path)
    os.makedirs(base_path, exist_ok=True)
    np.save(path, file)
        
def npy_load(path):
    return np.load(path)

def init_transforms(task=None, tran=None, meta=True):
    df_all = []
    for transform in TRANSFORMATIONS:
        t = transform(meta=meta)
        df = t.get_tran_types()
        df['transformation'] = t.__class__.__name__
        df['tran_fn'] = t
        df_all.append(df)
    df = pd.concat(df_all)
    if task is not None:
        task_df = df['task_name'] == task
        df = df[task_df]
    if tran is not None:
        tran_df = df['tran_type'] == tran
        df = df[tran_df]
    return df

def apply_transforms(test_suites, num_transforms=2, task=None, tran=None):
    df = init_transforms(task=task, tran=tran, meta=True)
    new_test_suites = {}
    for i, test_suite in tqdm(test_suites.items()):
        new_X, new_y, new_ts = [], [], []
        for X, y in zip(test_suite['data'], test_suite['target']):
            ts = []
            n = 0
            while n < num_transforms:
                t_df   = df.sample(1)
                t_fn   = t_df['tran_fn'][0]
                t_name = t_df['transformation'][0]
                if t_name in ts:
                    continue
                else:
                    ts.append(t_name)
                X, y, meta = t_fn.transform_Xy(X, y)
                if meta['change']:
                    n += 1
                else:
                    ts.remove(t_name)
            new_X.append(X)
            new_y.append(y)
            new_ts.append(ts)
        new_test_suites[i] = {'data': new_X, 'target': new_y, 'ts': new_ts}
    return new_test_suites