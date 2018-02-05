## basic
import os
import numpy as np
import pandas as pd
import sys
sys.path.append("../src")
## fast text or word2vec
import gensim
from gensim.models import Word2Vec
## multithreading
from multiprocessing import Pool
## parameter parser
import opts_parser
import json

ref = "1234567890qwertyuiopasdfghjklzxcvbnm "
def _process(sent):
    return ''.join([c for c in sent if c in ref])
def get_key_sentences(path):
    extra_sent = []
    for f in os.listdir(path):
        if f[-3:] == 'txt':
            fp = open(path + '/' + f, 'r')
            if 'phrase' in f:
                for line in fp:
                    extra_sent.append(_process(''.join(line.split('-')[:-1])))
            else:
                for line in f:
                    extra_sent.append(_process(' '.join(line.split(' ')[:-1])))
            fp.close()
    print("Getting {0} extra sentences.".format(str(len(extra_sent))))
    return extra_sent


def _get_sentences(s):
    return s.lower().split()


if __name__ == '__main__':
    train_data_file, test_data_file, config_file = opts_parser.getopts()
    train = pd.read_csv(train_data_file)
    test = pd.read_csv(test_data_file)
    
    ## Read From Config file
    cfg = json.load(open(config_file))
    print cfg
    path = cfg["path"]
    extra = cfg["extra"]
    vec_size = cfg["vec_size"]
    window = cfg['window']
    min_count = cfg["min_count"]
    workers = cfg["workers"]
    
    key_words_phrase = []
    if extra:
        key_words_phrase = get_key_sentences(path)
    
    pool=Pool(4)
    train_sents = pool.map(_get_sentences, train.comment_text.str.lower())
    test_sents = pool.map(_get_sentences, test.comment_text.str.lower())
    key_sents = pool.map(_get_sentences, key_words_phrase)
    pool.close()
    pool.join()
    
    sentences = train_sents+test_sents+key_sents
    model = Word2Vec(sentences, size=vec_size, window=window, min_count=min_count, workers=workers)
    filename = "toxic"
    if extra:
        filename += "_key"
    filename += "_{0}_{1}_{2}.txt".format(str(vec_size), str(window), str(min_count))
    model.save("data/" + filename)