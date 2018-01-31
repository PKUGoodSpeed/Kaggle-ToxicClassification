'''
Constructing keras models
'''
## Math and dataFrame
import numpy as np
import pandas as pd
import plot

## Traditional Machine Learning
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

## Keras
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, Callback
from keras.layers import Embedding, Flatten, Conv1D, Conv2D, GRU, LSTM, SimpleRNN, MaxPooling1D
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, AveragePooling1D

## Select keras layer via config input
layermap = {
    'embedding': Embedding,
    'flatten': Flatten,
    'conv': Conv1D,
    'conv2d': Conv2D,
    'gru': GRU,
    'lstm': LSTM,
    'simplernn': SimpleRNN,
    'dense': Dense,
    'dropout': Dropout,
    'batchnorm': BatchNormalization,
    'activation': Activation,
    'maxpooling': MaxPooling1D,
    'avepooling': AveragePooling1D,
    'globalmaxpooling': GlobalMaxPooling1D,
    'globalavepooling': GlobalAveragePooling1D
}

## Select keras optimizor via config input
optmap = {
    'sgd': SGD,
    'adam': Adam
}

## Model type list
model_type_list = ['sequential', 'parallel']

## Global learning rate
global_learning_rate = 1.0

## Glabal learning rate decay
global_decay_rate = 1.0
    
## The model class
class KerasModel:
    _input_shape = None         ## Input Shape
    _output_dim = None          ## Output dimension
    _model = None               ## keras model object
    _history = None             ## Plotting Use
    
    def __init__(self, input_shape, output_dim):
        '''
        Here we only need input shape and output dimension
        '''
        self._input_shape = input_shape
        self._output_dim = output_dim
        
    def _buildSequModel(self, layer_list):
        '''Building a sequential model'''
        input_layer = Input(shape=self._input_shape, name='input')
        tmp = input_layer
        for layer in layer_list:
            assert layer['name'] in layermap, "Layer {0} does not exist, you can invent one :)".format(layer['name'])
            tmp = layermap[layer['name']](*layer['args'], **layer['kargs']) (tmp)
        print self._output_dim
        output_layer = Dense(self._output_dim, activation='sigmoid') (tmp)
        print "Finish loading layers."
        self._model = Model(input_layer, output_layer)
    
    def _buildParaModel(self, layer_stack, combined_args):
        '''Building a parallel model'''
        assert len(layer_stack) >= 2, "You need at least 2 layer sequences to make parallel."
        for layer_seq in layer_stack:
            n = len(layer_seq)
            assert n, "Existing empty layer sequence(s)."
            assert layer_seq[n-1]['name'] == 'dense', "All layer sequences should end at a dense layer."
        output_layers = []
        for layer_seq in layer_stack:
            tmp = input_layer
            for layer in layer_seq:
                assert layer['name'] in layermap, "Layer {0} does not exist, you can invent one :)".format(layer['name'])
                tmp = layermap[layer['name']](*layer['args'], **layer['kargs']) (tmp)
            output_layers.append(tmp)
        main_layer = concatenate(output_layers)
        for layer in combined_args:
            main_layer = Dense(*layer['args'], **layer['kargs']) (main_layer)
        output_layer = Dense(self._output_dim, activation='sigmoid') (main_layer)
        print "Finish loading layers."
        self._model = Model(input_layer, output_layer)
    
    def getModel(self, model_type, **kargs):
        '''Getting the model'''
        assert model_type in model_type_list, "Wrong model type, should be \"sequential\" or \"parallel\"."
        if model_type == 'sequential':
            self._buildSequModel(**kargs)
        else:
            self._buildParaModel(**kargs)
        return self._model

    def train(self, train_x, train_y, valid_x, valid_y, optimizer='sgd', learning_rate=0.02, 
    decay_rate=0.85, epochs=36, adaptive_step=2, loss='binary_crossentropy', metrics=None):
        global global_learning_rate
        global global_decay_rate
        global_learning_rate = learning_rate
        global_decay_rate = decay_rate
        assert optimizer in optmap, "Wrong optimizer name"
        ## Using adaptive decaying learning rate
        def scheduler(epoch):
            global global_learning_rate
            global global_decay_rate
            if epoch%adaptive_step == 0:
                global_learning_rate *= global_decay_rate
                print("CURRENT LEARNING RATE = " + str(global_learning_rate))
            return global_learning_rate
        change_lr = LearningRateScheduler(scheduler)

        ## Compile the model
        self._model.compile(optimizer=optmap[optimizer](learning_rate), loss=loss, metrics=metrics)
        
        self._history = self._model.fit(train_x, train_y, batch_size=128, epochs=epochs,
        verbose=1, validation_data=(valid_x, valid_y), callbacks=[change_lr])
        return self._history
    
    def plot(self, filename='convergence.png'):
        '''Plot the convergence behavior'''
        plot.plotResult(self._history, filename=filename)