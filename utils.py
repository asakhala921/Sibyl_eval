import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import tzip

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
    return np.load(path, allow_pickle=True)

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

def transform_test_suites(test_suites, num_transforms=2, task=None, tran=None):
    df = init_transforms(task=task, tran=tran, meta=True)
    new_test_suites = {}
    for i, test_suite in tqdm(test_suites.items()):
        if tran=='SIB-mix':
            if type(test_suite['data']) == list:
                data = np.array(test_suite['data'], dtype=np.string_)
                targets = np.array(test_suite['target'])
            batch = (test_suite['data'], test_suite['target'])
            ts = []
            n = 0
            num_tries = 0
            while n < num_transforms:
                if num_tries > 25:
                    break
                t_df   = df.sample(1)
                t_fn   = t_df['tran_fn'].iloc[0]
                t_name = t_df['transformation'].iloc[0]
                if t_name in ts:
                    continue
                else:
                    ts.append(t_name)
                batch, meta = t_fn(batch)
                if meta['change']:
                    n += 1
                else:
                    ts.remove(t_name)
            new_test_suites[i] = {'data': batch[0], 'target': batch[1], 'ts': ts}
        else: 
            new_X, new_y, new_ts = [], [], []
            for X, y in zip(test_suite['data'], test_suite['target']):
                ts = []
                n = 0
                num_tries = 0
                while n < num_transforms:
                    if num_tries > 25:
                        break
                    t_df   = df.sample(1)
                    t_fn   = t_df['tran_fn'].iloc[0]
                    t_name = t_df['transformation'].iloc[0]
                    if t_name in ts:
                        continue
                    else:
                        ts.append(t_name)
                    X, y, meta = t_fn.transform_Xy(str(X), y)
                    if meta['change']:
                        n += 1
                    else:
                        ts.remove(t_name)
                new_X.append(X)
                new_y.append(y)
                new_ts.append(ts)
            new_test_suites[i] = {'data': new_X, 'target': new_y, 'ts': new_ts}
    return new_test_suites

def transform_dataset(dataset, num_transforms=2, task=None, tran=None):
    df = init_transforms(task=task, tran=tran, meta=True)
    text, label = dataset['text'], dataset['label'] 
    new_text, new_label, trans = [], [], []
    if tran == 'SIB-mix':
        if type(text) == list:
            text = np.array(text, dtype=np.string_)
            label = pd.get_dummies(label).to_numpy(dtype=np.float)
        batch_size= 1000
        for i in tqdm(range(0, len(label), batch_size)):
            text_batch = text[i:i+batch_size]
            label_batch = label[i:i+batch_size]
            batch = (text_batch, label_batch)
            t_trans = []
            n = 0
            num_tries = 0
            while n < num_transforms:
                if num_tries > 25:
                    break
                t_df   = df.sample(1)
                t_fn   = t_df['tran_fn'].iloc[0]
                t_name = t_df['transformation'].iloc[0]
                if t_name in trans:
                    continue
                else:
                    t_trans.append(t_name)
                batch, meta = t_fn(batch)
                if meta['change']:
                    n += 1
                else:
                    t_trans.remove(t_name)
                num_tries += 1
            new_text.extend(batch[0].tolist())
            new_label.extend(batch[1].tolist())
            trans.append(t_trans)
    else:
        for X, y in tzip(text, label):
            t_trans = []
            n = 0
            num_tries = 0
            while n < num_transforms:
                if num_tries > 25:
                    break
                t_df   = df.sample(1)
                t_fn   = t_df['tran_fn'].iloc[0]
                t_name = t_df['transformation'].iloc[0]
                if t_name in t_trans:
                    continue
                else:
                    t_trans.append(t_name)
                X, y, meta = t_fn.transform_Xy(str(X), y)
                if meta['change']:
                    n += 1
                else:
                    t_trans.remove(t_name)
            new_text.append(X)
            new_label.append(y)
            trans.append(t_trans)
    return new_text, new_label, trans

class CorrectKCounter:
    def __init__(self, k=1):
        self.k = k
    def __call__(self, logits, y_true):
        print(y_true.shape, logits.shape)
        y_weights, y_idx = torch.topk(y_true, k=self.k, dim=1)
        out_weights, out_idx = torch.topk(logits, k=self.k, dim=1)
        correct = torch.sum(torch.eq(y_idx, out_idx) * y_weights)
        return correct

class CorrectCounter:
    def __call__(self, logits, y_true):
        y_pred = torch.argmax(logits, dim=1)
        correct = (y_pred == y_true).sum().item()
        return correct