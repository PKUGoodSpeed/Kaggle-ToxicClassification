
# -*- coding: utf-8 -*-

import re, string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.utils import resample
from sklearn.utils import shuffle
import scipy

class FeatureExtraction(object):

    def tfIdf(self, df, COMMENT = 'comment_text'):

        assert(COMMENT in df.columns)

        re_tok = re.compile('([' + string.punctuation + '“”¨«»®´·º½¾¿¡§£₤‘’])')

        def tokenize(s): return re_tok.sub(r' \1 ', s).split()

        vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )

        doc = vec.fit_transform(df[COMMENT])

        return doc

    def tf(self, df, n_feature, COMMENT = 'comment_text'):

        assert(COMMENT in df.columns)

        re_tok = re.compile('([' + string.punctuation + '“”¨«»®´·º½¾¿¡§£₤‘’])')

        def tokenize(s): return re_tok.sub(r' \1 ', s).split()

        self.tf_vec = CountVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', max_features = n_feature, stop_words = 'english')

        doc = self.tf_vec.fit_transform(df[COMMENT])

        return doc


    def extractVocab(self, filename):
        ''' extract vocabulary from file
        '''

        with open(filename, 'r') as f:
            r = map(lambda x: x.split(' '), f.read().split('\n'))
            #print( r )
            if len(r[0]) == 2:
                df = pd.DataFrame(r, columns = ['term', 'freq'])
            elif len(r[0]) == 3:
                df = pd.DataFrame(r, columns = ['term', 'dummy', 'freq'])


            #print( df )
            df = df.dropna()
            #print(df)
            if 'dummy' in df.columns:
                df = df.drop('dummy', axis = 1)
            #tmp = zip(*r)
            #print tmp
        return df


    def tfKeyWord(self, df, n_feature, vocab, COMMENT = 'comment_text'):
        ''' build term freq from key word vocabulary
        '''

        assert(COMMENT in df.columns)

        re_tok = re.compile('([' + string.punctuation + '“”¨«»®´·º½¾¿¡§£₤‘’])')

        def tokenize(s): return re_tok.sub(r' \1 ', s).split()

        self.tf_vec = CountVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', max_features = n_feature,
               stop_words = 'english', vocabulary = vocab)

        doc = self.tf_vec.fit_transform(df[COMMENT])

        return doc

    def tfKeyWordEnsemble(self, df, n_feature, vocabfile = [], COMMENT = 'comment_text' ):
        ''' use key word from each category to build features, then concatenate them

        '''
        res = []
        for f in vocabfile:
            print ("extracting tf from ", f)
            vocab = self.extractVocab(f)
            print ("Extracted term doc")
            print vocab.head(10)
            feature = self.tfKeyWord(df, n_feature,  vocab['term'].values, COMMENT)
            print("feature shape", feature.shape )
            res.append(feature)

        #print("feature shape")
        #for i in res:
        #    print(i.shape)

        res = scipy.sparse.hstack(res)

        return res


    def reSample(self, trn_term_doc, y, doShuffle = True):
        ''' resample to balance labels
            NOTE: the indices after resample will be different from original DF

        '''
        assert(np.array_equal(y.unique(),  np.array([0,1])) )
        assert(np.array_equal(y.unique(),  np.array([0,1])) )
        lenLabel0 = np.sum(y == 0)
        lenLabel1 = np.sum(y == 1)
        print('label0 - {}'.format(lenLabel0) )
        print('label1 - {}'.format(lenLabel1) )
        #len( trn_term_doc[y == 0] )
        #len( trn_term_doc[y == 1] )
        #trn_term_doc[]
        sample0 = trn_term_doc.tocsr()[np.where(y == 0)[0], : ]
        sample1 = trn_term_doc.tocsr()[np.where(y == 1)[0], : ]
        assert(sample0.shape[0] == lenLabel0)
        assert(sample1.shape[0] == lenLabel1)

        if lenLabel1 < lenLabel0:
            print("Resample label 1")
            sample1 = resample(sample1,
                replace=True,     # sample with replacement
                n_samples=lenLabel0,    # to match majority class
                random_state=123) # reproducible results
        elif lenLabel0 < lenLabel1:
            print("Resample label 0")
            sample0 = resample(sample0,
                replace=True,     # sample with replacement
                n_samples=lenLabel1,    # to match majority class
                random_state=123) # reproducible results

        reSample = scipy.sparse.vstack([sample0, sample1])
        assert(sample0.shape[0] == sample1.shape[0])
        reLabel = np.hstack( [ np.zeros(sample0.shape[0]), np.ones(sample1.shape[0]) ] )
        assert( reSample.shape[0] == len(reLabel) )
        if doShuffle == True:
            reSample, reLabel = shuffle(reSample, reLabel, random_state=0)

        return reSample, reLabel
