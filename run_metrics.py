import argparse
import os
import sys
import random

from datasets import load_dataset, load_metric, Dataset
from transformers import AutoTokenizer
import tensorflow as tf

import pandas as pd

from utils import *
from transforms import *
from diversity_metrics import *

def calc_metrics(params):

    # init metrics 
    sacrebleu_score = load_metric("sacrebleu")
    bert_score = load_metric('bertscore')
    bleurt_score = load_metric('bleurt')
    tsd = TSD()
    ssd = SSD()
    sd = SD()
    tsd_norm = TSD_norm()
    ssd_norm = SSD_norm()
    sd_norm = SD_norm()

    # parse input csv path
    dataset_paths = parse_path_list(params.dataset_dir, 'datasets')
    print(dataset_paths)
    dataset_paths = [x for x in dataset_paths if 'text' in x]

    ORIG_text_path = [x for x in dataset_paths if 'ORIG' in x][0]
    ORIG_text = npy_load(ORIG_text_path)

    if params.n_samples > 0:
        idxs = list(range(len(ORIG_text)))
        random.shuffle(idxs)
        idxs = idxs[:params.n_samples]
        ORIG_text = [ORIG_text[i] for i in idxs]

    ORIG_text = [str(x) for x in ORIG_text]
    ORIG_text_listed = [[x] for x in ORIG_text]

    tf.compat.v1.flags.DEFINE_string('dataset_dir','','')
    tf.compat.v1.flags.DEFINE_string('n_samples','','')

    print('#' * 30)
    print("Calculating naturalness for the following datasets: \n", dataset_paths)
    results = []
    for p in dataset_paths:

        run = {}

        print("Running", p)

        DATA_text = npy_load(p)
        DATA_text = [str(x) for x in DATA_text]
        if params.n_samples > 0:
            DATA_text = [DATA_text[i] for i in idxs]
        run['dataset'] = p

        # naturalness metrics
        sacrebleu_s = sacrebleu_score._compute(DATA_text, ORIG_text_listed)['score']
        run['sacrebleu_score'] = sacrebleu_s
        print('sacrebleu_score:', sacrebleu_s)

        bert_s = bert_score._compute(DATA_text, ORIG_text, lang="en")['f1'].mean().item()
        run['bert_score'] = bert_s
        print('bert_score:', bert_s)

        bleurt_s = bleurt_score._compute(DATA_text, ORIG_text)
        bleurt_s = np.array(bleurt_s['scores']).mean()
        run['bleurt_score'] = bleurt_s
        print('bleurt_score:', bleurt_s)

        # diversity metrics
        ttr_text = get_ttr_text(DATA_text)
        run['ttr_text'] = ttr_text
        print('ttr_text:', ttr_text)

        ttr_token = get_ttr_tokens(DATA_text)
        run['ttr_token'] = ttr_token
        print('ttr_token:', ttr_token)

        TSD_score = tsd(DATA_text)
        run['TSD'] = TSD_score
        print('TSD:', TSD_score)
        
        SSD_score = ssd(DATA_text)
        run['SSD'] = SSD_score
        print('SSD:', SSD_score)
        
        SD_score = sd(DATA_text)
        run['SD'] = SD_score
        print('SD:', SD_score)

        TSD_norm_score = tsd_norm(DATA_text)
        run['TSD_norm'] = TSD_norm_score
        print('TSD_norm:', TSD_norm_score)
        
        SSD_norm_score = ssd_norm(DATA_text)
        run['SSD_norm'] = SSD_norm_score
        print('SSD_norm:', SSD_norm_score)
        
        SD_norm_score = sd_norm(DATA_text)
        run['SD_norm'] = SD_norm_score
        print('SD_norm:', SD_norm_score)

        sbleu = SelfBleu(DATA_text).get_score()
        run['SelfBleu'] = sbleu
        print('SelfBleu:', sbleu)

        results.append(run)

    df = pd.DataFrame(results)

    if params.output_dir == "":
        params.output_dir = params.dataset_dir
    df.to_csv(os.path.join(params.output_dir, "naturalness_metrics.csv"))

    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate naturalness metrics for input npy files'
                                                 ' result to a csv file.')

    # main parameters
    parser.add_argument("--dataset_dir", type=str, default='',
                        help='Datasets you want to calculate naturalness for. '
                        'Support multiple, comma separated, or dirs.')
    parser.add_argument("--n_samples", type=int, default=-1,
                        help='Number of samples from each dataset to sample. '
                        '-1 indicates the use of the entire dataset.')
    parser.add_argument("--output_dir", type=str, default='',
                        help='Output directory for the saved csv results.')

    params = parser.parse_args()

    calc_metrics(params)
