import numpy as np
import pandas as pd
import time

ref = "1234567890qwertyuiopasdfghjklzxcvbnm "
def _process(sent):
    return ''.join([c for c in sent if c in ref]).lower().strip()


def _get_keywords(label, filepath="../data/keywords"):
    filename = "{0}/{1}_words.txt".format(filepath, label)
    key_words = []
    fp = open(filename, 'r')
    for line in fp:
        key_words.append(_process(''.join(line.split(' ')[:-1])))
    fp.close()
    return key_words


def _get_keyphrase(label, filepath="../data/keywords"):
    filename = "{0}/{1}_phrases.txt".format(filepath, label)
    key_phrases = []
    fp = open(filename, 'r')
    for line in fp:
        key_phrases.append(_process(' '.join(line.split('-')[:-1])))
    fp.close()
    return key_phrases


def get_key_word_features(df, label, vec_size=128, keypath="../data/keywords"):
    print("Building Key Word Embedding Feature ...")
    start_time = time.time()
    key_words = _get_keywords(label=label, filepath=keypath)[: vec_size]
    print("Got key_words!")
    keyword_vec = []
    for sentence in df.comment_text.values:
        keyword_vec.append([int(w in sentence) for w in key_words])
    df['keyword_vec'] = keyword_vec
    print("Time for extracting key word feature: " + str(time.time() - start_time) + " sec")
    return df


def get_key_phrase_features(df, label, vec_size=128, keypath="../data/keywords"):
    print("Building Key Phrase Embedding Feature ...")
    start_time = time.time()
    key_phrases = _get_keyphrase(label=label, filepath=keypath)[: vec_size]
    print("Got key_phrases!")
    keyphra_vec = []
    for sentence in df.comment_text.values:
        keyphra_vec.append([int(w in sentence) for w in key_phrases])
    df['keyphra_vec'] = keyphra_vec
    print("Time for extracting key phrase feature: ", str(time.time() - start_time), " sec")
    return df          
    


def _test():
    kw = _get_keywords(label='toxic')
    kp = _get_keyphrase(label='toxic')
    print len(kw)
    print len(kp)
    print kw[:10]
    print kp[:10]
    df = pd.read_csv('../data/train_processed.csv')
    df = get_key_word_features(df, label='toxic')
    df = get_key_phrase_features(df, label='toxic')
    print df[:3]


if __name__ == '__main__':
    _test()
