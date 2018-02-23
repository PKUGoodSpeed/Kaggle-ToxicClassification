
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

        #re_tok = re.compile('([' + string.punctuation + '“”¨«»®´·º½¾¿¡§£₤‘’])')

        #def tokenize(s): return re_tok.sub(r' \1 ', s).split()

        #vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
        #       lowercase = False,
        #       min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
        #       smooth_idf=1, sublinear_tf=1 )

        self.vec = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english',
            ngram_range=(1, 2),
            max_features=20000)

        doc = self.vec.fit_transform(df[COMMENT])

        return doc

    def tfIdf_charNGram(self, df, n_feature, COMMENT = 'comment_text'):
        assert(COMMENT in df.columns)

        #re_tok = re.compile('([' + string.punctuation + '“”¨«»®´·º½¾¿¡§£₤‘’])')

        #def tokenize(s): return re_tok.sub(r' \1 ', s).split()

        #self.tfidf_ngram_vec = TfidfVectorizer(ngram_range=(2,6),
        #       analyzer = 'char',
        #       max_features = n_feature,
        #       tokenizer=tokenize,
        #       min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
        #       lowercase = False,
        #       smooth_idf=1, sublinear_tf=1 )

        self.tfidf_ngram_vec = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='char',
            stop_words='english',
            #lowercase = False,
            ngram_range=(2, 6),
            max_features=30000)

        doc = self.tfidf_ngram_vec.fit_transform(df[COMMENT])
        return doc


    def tf(self, df, n_feature, COMMENT = 'comment_text'):

        assert(COMMENT in df.columns)

        re_tok = re.compile('([' + string.punctuation + '“”¨«»®´·º½¾¿¡§£₤‘’])')

        def tokenize(s): return re_tok.sub(r' \1 ', s).split()

        self.tf_vec = CountVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               lowercase = False,
               min_df=3, max_df=0.9, strip_accents='unicode',
               max_features = n_feature, stop_words = 'english')

        doc = self.tf_vec.fit_transform(df[COMMENT])

        return doc

    def charNGram(self, df, n_feature, COMMENT = 'comment_text'):

        assert(COMMENT in df.columns)

        #re_tok = re.compile('([' + string.punctuation + '“”¨«»®´·º½¾¿¡§£₤‘’])')

        #def tokenize(s): return re_tok.sub(r' \1 ', s).split()

        self.tfngram_vec = CountVectorizer(ngram_range=(3,5),
                analyzer = 'char',
                min_df=3, max_df=0.9,
                lowercase = False,
                strip_accents='unicode', max_features = n_feature, stop_words = 'english')

        doc = self.tfngram_vec.fit_transform(df[COMMENT])

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


    def covarianceShiftCorrection(self, trn_term_doc, test_term_doc):
        ''' compute weights to weight training sample based on probability that it comes from
            same distribution with test data
        :term_doc: feature matrix containing both train and test
        :label: 0 for samples from test set, 1 for samples from training set
        '''
        print("covarianceShiftCorrection")
        from sklearn.linear_model import LogisticRegression

        #print ("feature shape = {}".format(trn_term_doc.shape))
        trn_label = np.zeros(trn_term_doc.shape[0]).reshape(-1,1)
        test_label = np.ones(test_term_doc.shape[0]).reshape(-1,1)
        label = np.vstack([trn_label, test_label]).reshape(1,-1)[0]
        term_doc = scipy.sparse.vstack([trn_term_doc, test_term_doc])
        assert(len(label) == term_doc.shape[0])

        #shuffle
        from sklearn.utils import shuffle
        term_doc_re, label_re= shuffle(term_doc, label, random_state=0)

        clf = LogisticRegression(penalty="l2", dual=True,
                   tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1,
                   class_weight=None, random_state=32,
                   max_iter=200, verbose=0,
                   warm_start=False, n_jobs=1)

        clf.fit(term_doc_re, label_re)
        trn_proba = clf.predict_proba(trn_term_doc)
        #trn_weights = np.exp(trn_proba)
        trn_weights = np.array(zip(*trn_proba)[1]) / np.array(zip(*trn_proba)[0])
        trn_weights = np.clip(trn_weights, 1e-12, 200)
        return trn_weights
