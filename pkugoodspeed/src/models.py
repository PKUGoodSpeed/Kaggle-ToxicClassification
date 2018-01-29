'''
keras models are loaded from here
rnn, cnn or combined ones
'''

## system
import os

## Math and dataFrame
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix, hstack

## Traditional Machine Learning
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

## Keras
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate
from keras.layers import Embedding, Flatten, Conv1D, Conv2D, GRU, LSTM, SimpleRNN, MaxPooling1D
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping#, TensorBoard
from keras import backend as K
from keras import optimizers
from keras.optimizers import SGD
from keras import initializers
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
VECTORIZOR = ['embedding', 'word2vec']
layermap = {
    'embedding': Embedding,
    'dense': Dense,
    'conv': Conv1D,
    'gru': GRU,
    'lstm': LSTM,
    'simplernn': SimpleRNN,
    'flatten': Flatten,
    'dropout': Dropout,
    'activation': Activation,
    'maxpooling': MaxPooling1D
}
learning_rate = 1.0
decay_rate = 1.0

optmap = {
    'sgd': SGD
}

# Make training data set be more balanced
def _makeBalance(df, target_label):
    L = len(df)
    minor = df[df[target_label] != 0]
    while len(df) < 2*L:
        df = df.append(minor, ignore_index=True)
    return df
    

    
class SingleLabelModel:
    _target = None      ## Which label to train
    _padlength = None   ## padlength
    _vectorize = None   ## which type of vectorization to use
    _splitratio = None  ## train/ train + cv
    _train = None       ## train set
    _cv = None          ## cross validation set
    _test = None        ## test set
    _inshape = None     ## Input shape
    
    def __init__(self, target='toxic', padlength=150, vectorize='embedding', splitratio=0.7):
        self._target = target
        assert self._target in LABELS, "Target does not exist"
        self._padlength = padlength
        self._vectorize = vectorize
        assert self._vectorize in VECTORIZOR, "Vectorizor approach does not exist"
        self._splitratio = splitratio
        assert self._splitratio > 0.5 and self._splitratio <= 1, "Wrong split ratio"
        
    def _loadDataEmb(self, train, test):
        '''loading data for embedding case'''
        print("Text to seq process...")
        print("   Fitting tokenizer...")
        raw_text = np.hstack([train.comment_text.str.lower(), test.comment_text.str.lower()])
        tok_raw = Tokenizer()
        tok_raw.fit_on_texts(raw_text)
        print("   Transforming text to seq...")
        train["input"] = tok_raw.texts_to_sequences(train.comment_text.str.lower())
        test["input"] = tok_raw.texts_to_sequences(test.comment_text.str.lower())
        self._train, self._cv = train_test_split(train[['input', self._target]], train_size=self._splitratio)
        self._train = _makeBalance(self._train, self._target)
        self._cv = _makeBalance(self._cv, self._target)
        self._test = test
        self._inshape = (self._padlength, )

    
    def loadData(self, train, test):
        if self._vectorize == 'embedding':
            self._loadDataEmb(train, test)
        else:
            self._loadDataEmb(train, test)
            
    
    def getSequentialModel(self, layers):
        input_layer = Input(shape=self._inshape, name='input')
        tmp = input_layer
        for l in layers:
            assert l['name'] in layermap.keys(), "Wrong layer name"
            print l['name']
            if l['name'] == 'conv':
                tmp = layermap[l['name']](l['filters'], l['ksize'], activation=l['actv'], padding=l['pad']) (tmp)
            else:
                tmp = layermap[l['name']](*l['args']) (tmp)
        
        output_layer = Dense(2, activation='softmax') (tmp)
        return Model(input_layer, output_layer)
        
    def trainModel(self, model, args):
        global learning_rate
        global decay_rate
        assert args['optimizer'] in optmap, "Wrong optimizer name"
        learning_rate = args['learning_rate']
        decay_rate = args['decay_rate']
        N_epoch = args['epoch']
        adaptive_step = args['adaptive_step']
        ## Using adaptive decaying learning rate
        def scheduler(epoch):
            global learning_rate
            global decay_rate
            if epoch%adaptive_step == 0:
                learning_rate *= decay_rate
                print("CURRENT LEARNING RATE = " + str(learning_rate))
            return learning_rate
        change_lr = LearningRateScheduler(scheduler)
        
        optimizer = optmap[args['optimizer']](learning_rate)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        train_x = pad_sequences(self._train.input, maxlen=self._padlength)
        cv_x = pad_sequences(self._cv.input, maxlen=self._padlength)
        train_y = np_utils.to_categorical(self._train.toxic.values, 2)
        cv_y = np_utils.to_categorical(self._cv.toxic.values, 2)
        
        return model.fit(train_x, train_y, batch_size = 128, epochs = N_epoch,
        verbose = 1, validation_data = (cv_x, cv_y), callbacks=[change_lr])
    
    
    