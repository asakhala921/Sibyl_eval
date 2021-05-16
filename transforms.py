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

import numpy as np
import pandas as pd
from tqdm import tqdm

def init_transforms(task_type=None, tran_type=None, label_type=None, meta=True):
    df_all = []
    for transform in TRANSFORMATIONS:
        t = transform(task=task_type, meta=meta)
        df = t.get_tran_types()
        df['transformation'] = t.__class__.__name__
        df['tran_fn'] = t
        df_all.append(df)
    df = pd.concat(df_all)
    if task_type is not None:
        task_df = df['task_name'] == task_type
        df = df[task_df]
    if tran_type is not None:
        tran_df = df['tran_type'] == tran_type
        df = df[tran_df]
    if label_type is not None:
        label_df = df['label_type'] == label_type
        df = df[label_df]
    df.reset_index(drop=True, inplace=True)
    return df

def transform_test_suites(
    test_suites, 
    num_INV_required=1, 
    num_SIB_required=1, 
    task_type=None, 
    tran_type=None, 
    label_type=None,
    one_hot=True):
    
    df = init_transforms(task_type=task_type, tran_type=tran_type, label_type=label_type, meta=True)

    new_test_suites = {}
    for i, test_suite in tqdm(test_suites.items()):
        new_text, new_label, trans = [], [], []
        text, label = test_suite.items()
        text, label = text[1], label[1]
        num_classes = len(np.unique(label))   
        for X, y in zip(text, label): 
            t_trans = []
            num_tries = 0
            num_INV_applied = 0
            while num_INV_applied < num_INV_required:
                if num_tries > 25:
                    break
                t_df   = df[df['tran_type']=='INV'].sample(1)
                t_fn   = t_df['tran_fn'].iloc[0]
                t_name = t_df['transformation'].iloc[0]                
                if t_name in trans:
                    continue
                X, y, meta = t_fn.transform_Xy(str(X), y)
                if one_hot:
                    y = one_hot_encode(y, num_classes)
                if meta['change']:
                    num_INV_applied += 1
                    t_trans.append(t_name)
                num_tries += 1

            num_tries = 0
            num_SIB_applied = 0       
            while num_SIB_applied < num_SIB_required:
                if num_tries > 25:
                    break
                t_df   = df[df['tran_type']=='SIB'].sample(1)
                t_fn   = t_df['tran_fn'].iloc[0]
                t_name = t_df['transformation'].iloc[0]                
                if t_name in trans:
                    continue
                if 'AbstractBatchTransformation' in t_fn.__class__.__bases__[0].__name__:
                    Xs, ys = sample_Xy(text, label, num_sample=1)
                    Xs.append(X); ys.append(y) 
                    Xs = [str(x) for x in Xs]
                    ys = [np.squeeze(one_hot_encode(y, num_classes)) for y in ys]
                    (X, y), meta = t_fn((Xs, ys), num_classes=num_classes)
                    X, y = X[0], y[0]
                else:
                    X, y, meta = t_fn.transform_Xy(str(X), y)
                if meta['change']:
                    num_SIB_applied += 1
                    t_trans.append(t_name)
                num_tries += 1

            new_text.append(X)
            new_label.append(y)
            trans.append(t_trans)
        
        new_test_suites[i] = {'data': new_text, 'target': new_label, 'ts': trans}
        
    return new_test_suites

def transform_dataset(dataset, num_transforms=2, task_type=None, tran_type=None, label_type=None):
    df = init_transforms(task_type=task_type, tran_type=tran_type, label_type=label_type, meta=True)
    text, label = dataset['text'], dataset['label'] 
    new_text, new_label, trans = [], [], []
    if tran_type == 'SIB':
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
                if len(batch[0].shape) > 1:
                    # print(t_name)
                    break
            new_text.extend(batch[0].tolist())
            new_label.extend(batch[1].tolist())
            trans.append(t_trans)
    else:
        for X, y in tqdm(zip(text, label), total=len(label)):
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

def transform_dataset_INVSIB(
    dataset, 
    num_INV_required=1, 
    num_SIB_required=1, 
    task_type=None, 
    tran_type=None, 
    label_type=None,
    one_hot=True):
    
    df = init_transforms(task_type=task_type, tran_type=tran_type, label_type=label_type, meta=True)
    
    text, label = dataset['text'], dataset['label']
    new_text, new_label, trans = [], [], []

    num_classes = len(np.unique(label))
    
    for X, y in tqdm(zip(text, label), total=len(label)): 
        t_trans = []

        num_tries = 0
        num_INV_applied = 0
        while num_INV_applied < num_INV_required:
            if num_tries > 25:
                break
            t_df   = df[df['tran_type']=='INV'].sample(1)
            t_fn   = t_df['tran_fn'].iloc[0]
            t_name = t_df['transformation'].iloc[0]                
            if t_name in trans:
                continue
            X, y, meta = t_fn.transform_Xy(str(X), y)
            if one_hot:
                y = one_hot_encode(y, num_classes)
            if meta['change']:
                num_INV_applied += 1
                t_trans.append(t_name)
            num_tries += 1

        num_tries = 0
        num_SIB_applied = 0       
        while num_SIB_applied < num_SIB_required:
            if num_tries > 25:
                break
            t_df   = df[df['tran_type']=='SIB'].sample(1)
            t_fn   = t_df['tran_fn'].iloc[0]
            t_name = t_df['transformation'].iloc[0]                
            if t_name in trans:
                continue
            if 'AbstractBatchTransformation' in t_fn.__class__.__bases__[0].__name__:
                Xs, ys = sample_Xy(text, label, num_sample=1)
                Xs.append(X); ys.append(y) 
                Xs = [str(x).encode('utf-8') for x in Xs]
                ys = [np.squeeze(one_hot_encode(y, num_classes)) for y in ys]
                (X, y), meta = t_fn((Xs, ys))
                X, y = X[0], y[0]
            else:
                X, y, meta = t_fn.transform_Xy(str(X), y)
            if meta['change']:
                num_SIB_applied += 1
                t_trans.append(t_name)
            num_tries += 1

        new_text.append(X)
        new_label.append(y)
        trans.append(t_trans)
                
    new_text = [str(x).encode('utf-8') for x in new_text]
    return np.array(new_text, dtype=np.string_), np.array(new_label), np.array(trans, dtype=np.string_)

def one_hot_encode(y, nb_classes):
    if isinstance(y, np.ndarray):
        return y
    y = np.array(y)
    res = np.eye(nb_classes)[np.array(y).reshape(-1)]
    return res.reshape(list(y.shape)+[nb_classes])

def sample_Xy(text, label, num_sample=1):
    idx = np.random.randint(0, len(text), num_sample)
    return list(np.array(text)[idx]), list(np.array(label)[idx])    