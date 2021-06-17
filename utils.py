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

def parse_path_list(path_str, default_path, file_extension='.npy'):
    path_list = []
    input_split = [default_path] if path_str == '' else path_str.split(',')

    for path in input_split:
        if os.path.isfile(path) and path.endswith(file_extension):
            path_list.append(path)
        elif os.path.isdir(path):
            for subdir, dirs, files in os.walk(path):
                for file in files:
                    sub_path = os.path.join(subdir, file)
                    if os.path.isfile(sub_path) and sub_path.endswith(file_extension):
                        path_list.append(sub_path)
        else:
            raise FileNotFoundError('[{}] not exists.'.format(path))

    return path_list

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


def sample_Xy(text, label, num_sample=1):
    idx = np.random.randint(0, len(text), num_sample)
    return list(np.array(text)[idx]), list(np.array(label)[idx])    
