import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import tzip

from transforms import *

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
        y_weights, y_idx = torch.topk(y_true, k=self.k, dim=1)
        out_weights, out_idx = torch.topk(logits, k=self.k, dim=1)
        correct = torch.sum(torch.eq(y_idx, out_idx) * y_weights)
        return correct

class CorrectCounter:
    def __call__(self, logits, y_true):
        y_pred = torch.argmax(logits, dim=1)
        correct = (y_pred == y_true).sum().item()
        return correct

def get_acc(logits, y_true, k=2):
    total = len(y_true)
    y_true = torch.tensor(y_true)
    y_pred = torch.argmax(logits, dim=1)
    correct = (y_pred == y_true).sum().item()
    acc = correct / total
    return acc.item()

def get_acc_at_k(logits, y_true, k=2):
    logits = torch.tensor(logits)
    y_true = torch.tensor(y_true)
    total = len(y_true)
    y_weights, y_idx = torch.topk(y_true, k=k, dim=-1)
    out_weights, out_idx = torch.topk(logits, k=k, dim=-1)
    correct = torch.sum(torch.eq(y_idx, out_idx) * y_weights)
    acc = correct / total
    return acc.item()