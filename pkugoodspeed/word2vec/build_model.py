## basic
import os
import numpy as np
import pandas as pd
import sys
sys.path.append("../src")
## Tokenizer
from keras.preprocessing.text import Tokenizer
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


def get_word_index(train_data_file, test_data_file, key_path):
    train = pd.read_csv(train_data_file)
    test = pd.read_csv(test_data_file)
    extra = np.array(get_key_sentences(key_path))
    raw_text = np.hstack([train.comment_text.str.lower(), test.comment_text.str.lower(), extra])
    tok_raw = Tokenizer()
    tok_raw.fit_on_texts(raw_text)
    return tok_raw.word_index


def _get_sentences(s):
    return s.lower().split()


def _get_emb_from_file(model_file, binary=False):
    return gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=binary)
def _get_emb_from_train(sentences, emb_size=100, window=5, min_count=1, workers=4):
    return Word2Vec(sentences, size=emb_size, window=window, min_count=min_count, workers=workers)
def get_emb(emb_type, kargs):
    print("Build embedding model ...")
    if emb_type == 'generate':
        return _get_emb_from_train(**kargs)
    else:
        return _get_emb_from_file(**kargs)


def dump_to_file(word_index, model, filename):
    print("Writing embedding to {0} ...".format(filename))
    embedding_index = {}
    for w in word_index.keys():
        if w in model.wv.vocab:
            embedding_index[w] = model.wv[w]
    
    fp = open(filename, 'w')
    for word, vec in embedding_index.items():
        fp.write(word)
        for x in vec:
            fp.write(' '+str(x))
        fp.write('\n')
    fp.close()



if __name__ == '__main__':
    train_data_file, test_data_file, config_file = opts_parser.getopts()
    train = pd.read_csv(train_data_file)
    test = pd.read_csv(test_data_file)
    
    ## Read From Config file
    cfg = json.load(open(config_file))
    key_path = cfg["key_path"]
    word_index = get_word_index(train_data_file, test_data_file, key_path)

    kargs = cfg["kargs"]
    if cfg["emb_type"] == "generate":
        key_words_phrase = get_key_sentences(key_path)
        pool=Pool(4)
        train_sents = pool.map(_get_sentences, train.comment_text.str.lower())
        test_sents = pool.map(_get_sentences, test.comment_text.str.lower())
        key_sents = pool.map(_get_sentences, key_words_phrase)
        pool.close()
        pool.join()
        kargs["sentences"] = train_sents+test_sents+key_sents
        assert cfg["emb_size"] == kargs["emb_size"]

    model = get_emb(cfg["emb_type"], kargs)
    filename = cfg["output_dir"] + "/{0}.{1}.txt".format(cfg["emb_type"], str(cfg["emb_size"]))
    dump_to_file(word_index, model, filename)