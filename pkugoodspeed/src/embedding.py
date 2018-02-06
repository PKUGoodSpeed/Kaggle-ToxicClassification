import numpy as np
def _get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')
def emb_from_emb(embedding_file):
    print "Load embedding information..."
    return dict(_get_coefs(*o.strip().split()) for o in open(embedding_file))