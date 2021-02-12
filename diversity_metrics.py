# Diversity Metrics
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

class TextDiversity:
    def __init__(self, 
                 Z_type="token_semantics",
                 MODEL_NAME="bert-base-uncased", 
                 batch_size=16, 
                 use_gpu=True, 
                 verbose=False
                ):

        self.MODEL_NAME = MODEL_NAME
        self.Z_type = Z_type
        if self.Z_type == "token_semantics":
            self.model = AutoModel.from_pretrained(MODEL_NAME)
        elif self.Z_type == "sentence_semantics":
            self.model = SentenceTransformer('stsb-distilbert-base')
        else:
            self.model = spacy.load("en_core_web_trf")
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

        self.species = None
        self.Z = None
        self.boe = None

        # move model to device
        if isinstance(self.model, torch.nn.Module):
            self.model.to(self.device)
    
    def __call__(self, 
                 corpus, 
                 q=1, 
                 distance_fn=None, 
                 n_components='auto', 
                 remove_stopwords=True,
                 normalize=False,
                 scale_dist="exp",
                 sq_reg=False, 
                 mean_adj=True,
                 ignore_similarities=False):
        
        # get bag_of_embeddings (boe) from model + tokens
        if self.Z_type == 'token_semantics':
            boe, species = self.get_embeddings(corpus, 
                                            n_components=n_components, 
                                            remove_stopwords=remove_stopwords)
        elif self.Z_type == 'sentence_semantics':
            boe, species = self.get_sent_embeddings(corpus)
        elif self.Z_type == 'sentence_structure':
            boe, species = self.get_sentstruct_embeddings(corpus, 
                                            remove_stopwords=remove_stopwords)
            scale_dist = "invert"
            distance_fn = spatial.distance.hamming
            mean_adj = False

        # get similarity matrix (Z)
        Z = self.get_emb_similarities(boe, distance_fn, scale_dist, sq_reg, mean_adj)
        
        # get diversity
        num_species = len(species)
        p = np.full(num_species, 1/num_species)
        D = self.get_diversity(p, Z, q, normalize, ignore_similarities)

        if self.verbose:
            ret = {
                'diversity': D,
                'diversity_normalized': (D / len(self.species)),
                'entropy': np.log(D),
                'species': self.species,
                'Z': self.Z,
                'boe': self.boe
            } 
        else:
            ret = {
                'diversity': D,
                'diversity_normalized': D / len(self.species),
            }    
        
        return ret

    def encode(self, input_ids, attention_mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids, attention_mask=attention_mask)
        emb = out[0]
        return emb

    def get_embeddings(self, corpus, n_components='auto', remove_stopwords=True):
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

        # remove undesirable tokens
        idx = np.isin(inputs['input_ids'],  self.undesirable_tokens, assume_unique=True, invert=True).reshape(-1)
        tok = np.array(self.tokenizer.convert_ids_to_tokens(inputs.input_ids.view(-1)))[idx]
        boe = embeddings.view(-1, embeddings.shape[-1])[idx].detach().cpu()

        # remove stopwords
        if remove_stopwords:
            idx = np.isin(tok, stopwords.words('english'), invert=True)
            tok = tok[idx]
            boe = boe[idx]

        # compress embedding to speed up similarity matrix computation
        if n_components == "auto":
            n_components = min(max(2, len(boe) // 10), boe.shape[-1])
            if self.verbose:
                print('Using n_components={}'.format(str(n_components)))

        if type(n_components) == int and n_components > 0 and len(boe) > 1:
            boe = PCA(n_components=n_components).fit_transform(boe)

        tok, boe = merge_bpe(tok, boe)

        self.species = tok
        self.boe = boe

        return boe, tok

    def get_sent_embeddings(self, corpus, n_components='auto'):
        boe = self.model.encode(corpus)
        
        # compress embedding to speed up similarity matrix computation
        if n_components == "auto":
            n_components = min(max(2, len(boe) // 10), boe.shape[-1])
            if self.verbose:
                print('Using n_components={}'.format(str(n_components)))

        if type(n_components) == int and n_components > 0 and len(boe) > 1:
            boe = PCA(n_components=n_components).fit_transform(boe)

        self.species = corpus
        self.boe = boe

        return boe, corpus
    
    def get_sentstruct_embeddings(self, corpus, part='pos_', part2int=True, remove_stopwords=True):

        # convert to spacy docs to get parts
        doc_parts = []
        for doc in corpus:
            doc_ = []
            for w in self.model(doc):
                if remove_stopwords and w.text in stopwords.words('english'):
                    continue
                part_ = getattr(w, part)
                doc_.append(part_)
            doc_parts.append(doc_)

        # pad to max sentence doc length
        pad_to = find_max_list(doc_parts)
        doc_parts = np.array([s + ['NULL']*(pad_to-len(s)) for s in doc_parts])

        # convert doc parts to int
        if part2int:
            # build dict of unique doc parts
            part_map = set(itertools.chain(*doc_parts))
            part_map = {tag: i for i, tag in enumerate(part_map)}
            # convert to int for distance comparison
            part2int_fn = np.vectorize(part_map.get)
            boe = part2int_fn(doc_parts)
            
        self.boe = doc_parts
        self.species = doc_parts

        return doc_parts, corpus

    def get_emb_similarities(
        self, 
        boe, 
        dist_fn=None, 
        scale_dist="exp", 
        sq_reg=True, 
        mean_adj=True):
        
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
                if scale_dist == "exp":
                    d = np.exp(-d)
                elif scale_dist == "invert":
                    d = 1 - d
                tok_sims[e1][e1 + e2] = d     
        tok_sims[il] = tok_sims[iu]

        if sq_reg:
            tok_sims **= 1.5

        if mean_adj:
            off_diag = np.where(~np.eye(tok_sims.shape[0],dtype=bool))
            tok_sims[off_diag] -= tok_sims[off_diag].mean()
            tok_sims = np.where(tok_sims < 0, 0 , tok_sims)

        self.Z = tok_sims

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
        Z = self.get_emb_similarities(boe, distance_fn)

        # plot diversity profile
        num_species = len(num_species)
        p = np.full(num_species, 1/num_species)
        
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

    def get_species_heatmap(self, n=10):
        g = sns.heatmap(np.around(self.Z[:n,:n], 2), annot=True, annot_kws={"fontsize": 10}, fmt='g')
        g.set_xticklabels(self.species[:n], rotation=90)
        g.set_yticklabels(self.species[:n], rotation=0)
        g.set_title('Token Semantic Similarities')
        plt.show()


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