# Naturalness

import argparse
import os
import sys

from datasets import load_dataset, load_metric, Dataset
from transformers import AutoTokenizer
import tensorflow as tf

import pandas as pd

from utils import *
from transforms import *
from diversity_metrics import *

def calc_nat(params):

    # init metrics 
    bleu_score = load_metric("bleu")
    bert_score = load_metric('bertscore')
    bleurt_score = load_metric('bleurt')

    # parse input csv path
    dataset_paths = parse_path_list(params.dataset_dir, 'datasets')
    dataset_paths = [x for x in dataset_paths if 'text' in x]

    ORIG_text_path = [x for x in dataset_paths if 'ORIG' in x][0]
    ORIG_text = npy_load(ORIG_text_path)[:10]
    ORIG_text = [str(x) for x in ORIG_text]

    tf.compat.v1.flags.DEFINE_string('dataset_dir','','')

    print('#' * 30)
    print("Calculating naturalness for the following datasets: \n", dataset_paths)
    results = []
    for p in dataset_paths:

        run = {}

        DATA_text = npy_load(p)[:10]
        DATA_text = [str(x) for x in DATA_text]
        run['dataset'] = p

        print(ORIG_text, DATA_text)

        # naturalness metrics
        bleu_s = bleu_score._compute(DATA_text, ORIG_text)['bleu']
        run['bleu_score'] = bleu_s
        print('bleu_score:', bleu_s)

        bert_s = bert_score._compute(DATA_text, ORIG_text, lang="en")['f1'].mean().item()
        run['bert_score'] = bert_s
        print('bert_score:', bert_s)

        bleurt_s = bleurt_score._compute(DATA_text, ORIG_text)
        bleurt_s = np.array(bleurt_s['scores']).mean()
        run['bleurt_score'] = bleurt_s
        print('bleurt_score:', bleurt_s)

        results.append(run)


    df = pd.DataFrame(results)

    if params.output_dir == "":
        params.output_dir = "datasets"
    df.to_csv(os.path.join(params.output_dir, "naturalness_metrics"))

    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate naturalness metrics for input npy files'
                                                 ' result to a csv file.')

    # main parameters
    parser.add_argument("--dataset_dir", type=str, default='',
                        help='Datasets you want to calculate naturalness for. '
                        'Support multiple, comma separated, or dirs. ')
    parser.add_argument("--output_dir", type=str, default='',
                        help='Output directory for the saved csv results.')

    params = parser.parse_args()

    calc_nat(params)
