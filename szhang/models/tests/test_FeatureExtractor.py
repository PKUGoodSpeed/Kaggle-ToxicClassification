from __future__ import print_function
import pandas as pd
import sys, os
import unittest
import numpy as np
import subprocess
import os,sys

## Math and dataFrame
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

#import matplotlib.pyplot as plt

from models.FeatureExtraction import FeatureExtraction

class Test_FeatureExtractor(unittest.TestCase):

    def setUp(self):
        train = pd.read_csv('data/train.csv')
        test = pd.read_csv('data/test.csv')

        #train['dirtyness'] = train.apply(lambda x: x.iloc[2::].sum(), axis = 1)
        #test['dirtyness'] = test.apply(lambda x: x.iloc[2::].sum(), axis = 1)

        COMMENT = 'comment_text'
        train[COMMENT].fillna("unknown", inplace=True)
        test[COMMENT].fillna("unknown", inplace=True)

        print("train set len ", len(train) )
        print("test set len ", len(test) )
        #print("clean samples", len(train[train['dirtyness'] == 0]))
        #print("toxic samples", len(train[train['dirtyness'] != 0]))
        self.train = train
        self.test = test

        self.fe = FeatureExtraction()


    def test_tfKeyword(self):
        keyword_dir = '../ZhiHaoSun/'
        keyword = keyword_dir + 'toxic_words.txt'

        #with open(keyword, 'r') as f:
        #    for line in f:
        #        print (line)
        vocab = self.fe.extractVocab(keyword)

        #print(vocab)

        print ("train comment")
        dut = self.train[self.train['toxic'] == 1].iloc[0:1]
        print(dut['comment_text'])

        feature = self.fe.tfKeyWord(dut.iloc[0:1], n_feature = 2000, vocab = vocab['term'].values,
                COMMENT = 'comment_text')

        for i in scipy.sparse.find(feature)[1]:
            print( vocab.iloc[ i ]['term'] )

    def test_tfKeywordEnsemble(self):
        keyword_dir = '../ZhiHaoSun/'
        keyfiles = [
                keyword_dir + 'toxic_words.txt',
                keyword_dir + 'identity_hate_words.txt',
                keyword_dir + 'insult_words.txt',
                keyword_dir + 'obscene_words.txt',
                keyword_dir + 'threat_words.txt',
                keyword_dir + 'identity_hate_words.txt',
                ]

        print ("train comment")
        dut = self.train[self.train['toxic'] == 1].iloc[0:1]
        print(dut['comment_text'])

        feature = self.fe.tfKeyWordEnsemble(
                dut, n_feature = 2000, vocabfile = keyfiles,
                COMMENT = 'comment_text'
                )

        print("Exracted feature shape {}".format(feature.shape))




if __name__ == "__main__":
    suite = unittest.TestSuite()
    #suite.addTest(Test_FeatureExtractor('test_tfKeyword'))
    suite.addTest(Test_FeatureExtractor('test_tfKeywordEnsemble'))

    unittest.TextTestRunner(verbosity = 2).run(suite)



