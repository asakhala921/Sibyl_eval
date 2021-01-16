import os
import pickle
import numpy as np

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

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