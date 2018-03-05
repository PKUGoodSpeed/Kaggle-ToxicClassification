'''
Constructing keras models
'''
## Math and dataFrame
import os
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
from keras.callbacks import LearningRateScheduler, Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Flatten, Conv1D, Conv2D, GRU, LSTM, SimpleRNN, MaxPooling1D
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, AveragePooling1D
from keras.layers import Bidirectional

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
        
    def _buildSequModel(self, layer_list, max_features=20000, emb_size=100, emb_weights=None, bidirect=False):
        '''Building a sequential model'''
        input_layer = Input(shape=self._input_shape, name='input')
        tmp = input_layer
        if emb_weights is None:
            tmp = Embedding(max_features, emb_size) (tmp)
        else:
            tmp = Embedding(max_features, emb_size, weights=[emb_weights]) (tmp)
        for layer in layer_list:
            assert layer['name'] in layermap, "Layer {0} does not exist, you can invent one :)".format(layer['name'])
            if bidirect and layer['name'] in ['lstm', 'gru']:
                tmp = Bidirectional(layermap[layer['name']](*layer['args'], **layer['kargs'])) (tmp)
            else:
                tmp = layermap[layer['name']](*layer['args'], **layer['kargs']) (tmp)
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

    def train(self, train, valid, target_list, optimizer='sgd', learning_rate=0.02, 
    decay_rate=0.85, epochs=36, adaptive_step=2, loss='binary_crossentropy', metrics=None,
    check_file='weights.h5'):
        train_x = np.array(train.input.tolist())
        valid_x = np.array(valid.input.tolist())
        train_y = train[target_list].values
        valid_y = valid[target_list].values
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

        earlystopper = EarlyStopping(monitor='val_acc', patience=5, mode='max', verbose=1)
        if not os.path.exists('./checkpoints'):
            os.system('mkdir checkpoints')
        checkpointer = ModelCheckpoint(filepath='./checkpoints/'+check_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        ## Compile the model
        self._model.compile(optimizer=optmap[optimizer](learning_rate), loss=loss, metrics=metrics)
        
        self._history = self._model.fit(train_x, train_y, batch_size=128, epochs=epochs,
        verbose=1, validation_data=(valid_x, valid_y), callbacks=[earlystopper, checkpointer, change_lr])
        self._model.load_weights("./checkpoints/"+check_file)
        return self._history
    
    def plot(self, filename='convergence.png'):
        '''Plot the convergence behavior'''
        plot.plotResult(self._history, filename=filename)

    def make_predict(self, test, target_list):
        test_x = np.array(test.input.tolist())
        for target in target_list:
            test[target] = [0]*len(test)
        test[target_list] = self._model.predict(test_x, batch_size=128)
        return test


    def getCombinedModel(self, layer_list, max_features=20000, emb_size=100, 
    emb_weights=None, bidirect=False, kw_dims=[64, 256, 32], kp_dim=[64, 256, 32]):
        '''Building a sequential model'''
        input_layer = Input(shape=self._input_shape, name='input')
        tmp = input_layer
        if emb_weights is None:
            tmp = Embedding(max_features, emb_size) (tmp)
        else:
            tmp = Embedding(max_features, emb_size, weights=[emb_weights]) (tmp)
        for layer in layer_list:
            assert layer['name'] in layermap, "Layer {0} does not exist, you can invent one :)".format(layer['name'])
            if bidirect and layer['name'] in ['lstm', 'gru']:
                tmp = Bidirectional(layermap[layer['name']](*layer['args'], **layer['kargs'])) (tmp)
            else:
                tmp = layermap[layer['name']](*layer['args'], **layer['kargs']) (tmp)
        ## Adding keywords and keyphrases vectorized features
        assert len(kw_dims) > 1 and len(kp_dim) > 1, "Should have at least one hidden layer for key vectors"
        keyword_layer = Input(shape=[kw_dims[0]], name='keywords')
        kw = keyword_layer
        for neuron in kw_dims[1:]:
            kw = Dropout(0.3) (Dense(neuron, activation='sigmoid') (kw))
        
        keyphra_layer = Input(shape=[kp_dim[0]], name='keyphrases')
        kp = keyphra_layer
        for neuron in kp_dim[1:]:
            kp = Dropout(0.3) (Dense(neuron, activation='sigmoid') (kp))
        
        main_layer = concatenate([tmp, kw, kp])
        main_layer = Dropout(0.5) (Dense(128, activation='relu') (main_layer))
        output_layer = Dense(self._output_dim, activation='sigmoid') (main_layer)
        print "Finish loading layers."
        self._model = Model([input_layer, keyword_layer, keyphra_layer], output_layer)
        return self._model
    
    def trainCombinedModel(self, train, valid, target_list, optimizer='sgd', learning_rate=0.02, 
    decay_rate=0.85, epochs=36, adaptive_step=2, loss='binary_crossentropy', metrics=None,
    check_file='weights.h5'):
        train_x = {
            'input': np.array(train.input.tolist()),
            'keywords': np.array(train.keyword_vec.tolist()),
            'keyphrases': np.array(train.keyphra_vec.tolist())
        }
        print train_x['input'].shape
        print train_x['keywords'].shape
        print train_x['keyphrases'].shape
        valid_x = {
            'input': np.array(valid.input.tolist()),
            'keywords': np.array(valid.keyword_vec.tolist()),
            'keyphrases': np.array(valid.keyphra_vec.tolist())
        }
        train_y = train[target_list].values
        valid_y = valid[target_list].values
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
        earlystopper = EarlyStopping(patience=5, verbose=1)
        if not os.path.exists('./checkpoints'):
            os.system('mkdir checkpoints')
        checkpointer = ModelCheckpoint(filepath='./checkpoints/'+check_file, verbose=1, save_best_only=True)

        ## Compile the model
        self._model.compile(optimizer=optmap[optimizer](learning_rate), loss=loss, metrics=metrics)
        
        self._history = self._model.fit(train_x, train_y, batch_size=128, epochs=epochs,
        verbose=1, validation_data=(valid_x, valid_y), callbacks=[earlystopper, checkpointer, change_lr])
        self._model.load_weights("./checkpoints/"+check_file)
        return self._history

    def predictCombinedModel(self, test, target_list):
        test_x = {
            'input': np.array(test.input.tolist()),
            'keywords': np.array(test.keyword_vec.tolist()),
            'keyphrases': np.array(test.keyphra_vec.tolist())
        }
        for target in target_list:
            test[target] = [0]*len(test)
        test[target_list] = self._model.predict(test_x, batch_size=128)
        return test
        