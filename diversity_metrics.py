# Diversity Metrics
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
from scipy import spatial
import torch
import numpy as np
from tqdm import tqdm
import seaborn as sns 
import matplotlib.pyplot as plt

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

from utils import *

def get_ttr_by_ids(input_ids):
    '''
    Type Token Ratio (TTR)
    higher -> more diversity
    '''
    token_ids, counts = np.unique(input_ids, return_counts=True)
    idx = np.isin(token_ids, [0, 101, 102], assume_unique=True, invert=True)
    token_ids, counts = token_ids[idx], counts[idx]
    ttr = len(token_ids) / counts.sum() * 100
    return ttr

def get_ttr_by_text(text):
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
    ttr = len(toks) / tok_counts.sum() * 100
    return ttr

class TextDiversity:
    def __init__(self, 
                 MODEL_NAME="bert-base-uncased", 
                 batch_size=16, 
                 use_gpu=True, 
                 verbose=False):
        self.MODEL_NAME = MODEL_NAME
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.undesirable_tokens = [
            self.tokenizer.pad_token_id, 
            self.tokenizer.cls_token_id, 
            self.tokenizer.sep_token_id
        ]
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.verbose = verbose

        self.tokens = None
        self.token_similarities = None
        self.bag_of_embeddings = None

        # move model to device
        self.model.to(self.device)
    
    def __call__(self, 
                 corpus, 
                 q=1, 
                 distance_fn=None, 
                 n_components='auto', 
                 normalize=False, 
                 ignore_similarities=False):
        
        # get bag_of_embeddings (boe) from model + tokens
        boe, tok = self.get_embeddings(corpus, n_components)

        # get similarity matrix (Z)
        Z = self.get_token_similarities(boe, distance_fn)
        
        # get diversity
        num_tok = len(tok)
        p = np.full(num_tok, 1/num_tok)
        D = self.get_diversity(p, Z, q, normalize, ignore_similarities)

        if self.verbose:
            ret = {
                'diversity': D,
                'diversity_normalized': D / len(self.tokens),
                'entropy': np.log(D),
                'tokens': self.tokens,
                'token_similarities': self.token_similarities,
                'bag_of_embeddings': self.bag_of_embeddings
            } 
        else:
            ret = {
                'diversity': D,
                'diversity_normalized': D / len(self.tokens),
            }    
        
        return ret

    def encode(self, input_ids, attention_mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids, attention_mask=attention_mask)
        emb = out[0]
        return emb

    def get_embeddings(self, corpus, n_components='auto'):
        # TODO: figure out how to handle BPE breakdown
        #       of single word into multiple token_ids
        inputs = self.tokenizer(corpus, return_tensors='pt', padding=True, truncation=True)
        batches = zip(chunker(inputs.input_ids, self.batch_size), 
                      chunker(inputs.attention_mask, self.batch_size))
        if self.verbose:
            print('getting token embeddings...')
            batches = tqdm(batches, total=int(len(inputs.input_ids)/self.batch_size))

        outputs = []
        for input_ids, attention_mask in batches:
          emb = self.encode(input_ids.to(self.device), 
                       attention_mask.to(self.device))
          outputs.append(emb)
        embeddings = torch.cat(outputs)

        idx = np.isin(inputs['input_ids'],  self.undesirable_tokens, assume_unique=True, invert=True).reshape(-1)
        tok = np.array(self.tokenizer.convert_ids_to_tokens(inputs.input_ids.view(-1)))[idx]
        boe = embeddings.view(-1, embeddings.shape[-1])[idx].detach().cpu()

        # compress embedding to speed up similarity matrix computation
        if n_components == "auto":
        	n_components = min(max(2, len(boe) // 10), boe.shape[-1])
        	if self.verbose:
        		print('Using n_components={}'.format(str(n_components)))

        if type(n_components) == int and n_components > 0 and len(boe) > 1:
            boe = PCA(n_components=n_components).fit_transform(boe)

        self.tokens = tok
        self.bag_of_embeddings = boe

        return boe, tok

    def get_token_similarities(self, boe, dist_fn=None):
        if dist_fn is None:
            dist_fn = spatial.distance.chebyshev  

        num_embeddings = len(boe)

        tok_sims = np.ones((num_embeddings, num_embeddings))
        iu = np.triu_indices(num_embeddings, k=1)
        il = (iu[1], iu[0])

        iterable = range(num_embeddings)
        if self.verbose:
            print('calculating similarity matrix...')
            iterable = tqdm(iterable)

        for e1 in iterable:
            for e2 in range(1, num_embeddings - e1):
                d = dist_fn(boe[e1], boe[e1 + e2])
                scaled_d = np.exp(-d)
                tok_sims[e1][e1 + e2] = scaled_d        
        tok_sims[il] = tok_sims[iu]

        self.token_similarities = tok_sims

        return tok_sims    

    def get_diversity(self, p, Z, q=1, normalize=False, ignore_similarities=False):
        if ignore_similarities:
            Z = np.eye(len(Z))
        Zp =  Z @ p
        if q == 1:
            D = 1 / np.prod(Zp ** p)
        elif q == np.inf:
            D = 1 / Zp.max()
        else:
            D = (p * Zp ** (q-1)).sum() ** (1/(1-q))
        if normalize:
            D /= len(p)
        return D    

    def get_diversity_profile(self, 
                              corpus, 
                              distance_fn=None, 
                              n_components=2, 
                              normalize=False, 
                              ignore_similarities=False, 
                              range=None):
        # get bag_of_embeddings (boe) from model + tokens
        boe, tok = self.get_embeddings(corpus, n_components)

        # get similarity matrix (Z)
        Z = self.get_token_similarities(boe, distance_fn)

        # plot diversity profile
        num_tok = len(tok)
        p = np.full(num_tok, 1/num_tok)
        
        if range is None:
            range = np.arange(0, 101)
        Ds = []
        for q in range:
            D = self.get_diversity(p, Z, q, normalize)
            Ds.append(D)
        ax = sns.lineplot(x=range, y=Ds)  
        ax.set(xlabel="Sensitivity Parameter, $q$", 
               ylabel="Diversity $^qD(\mathbf{p})$" if ignore_similarities else "Diversity $^qD^{\mathbf{Z}}(\mathbf{p})$", 
               title="Corpus Diversity Profile")
        plt.show()