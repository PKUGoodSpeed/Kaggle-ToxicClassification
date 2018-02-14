import numpy as np
import pandas as pd

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


def get_key_word_features(df, label, keypath="../data/keywords"):
    print("Building Key Word Embedding Feature ...")
    key_words = _get_keywords(label=label, filepath=keypath)
    keyword_vec = np.zeros((len(df), len(key_words)))
    for i in range(len(df)):
        for j in range(len(key_words)):
            if key_words[j] in df.comment_text.values[i]:
                keyword_vec[i][j] = 1.
    df['keyword_vec'] = keyword_vec
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
    print df[:3]


if __name__ == '__main__':
    _test()
