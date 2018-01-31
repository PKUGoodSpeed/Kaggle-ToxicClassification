'''
Preprocess data into trainable input sets
'''
## Basic
import pandas as pd
import numpy as np

## Keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

## Spliting train and valid
from sklearn.model_selection import train_test_split, cross_val_score

# Make training data set be more balanced
def _makeBalance(df, target_list, expand_ratio=2.):
    L = len(df)
    while len(df) < expand_ratio * L:
        for target in target_list:
            minor = df[df[target] != 0]
            df = df.append(minor, ignore_index=True)
    return df

def embProcess(train_df, test_df, target_list=['toxic'], split_ratio=0.7, expand_ratio = 2.0, padlength=150):
    '''loading data for embedding case'''
    print("Text to seq process...")
    print("   Fitting tokenizer...")
    raw_text = np.hstack([train_df.comment_text.str.lower(), test_df.comment_text.str.lower()])
    tok_raw = Tokenizer()
    tok_raw.fit_on_texts(raw_text)
    print("   Transforming text to seq...")
    train_df["seq"] = tok_raw.texts_to_sequences(train_df.comment_text.str.lower())
    test_df["seq"] = tok_raw.texts_to_sequences(test_df.comment_text.str.lower())
    train, valid = train_test_split(train_df, train_size=split_ratio)
    train = _makeBalance(train, target_list, expand_ratio)
    valid = _makeBalance(valid, target_list, expand_ratio)
    train_x = pad_sequences(train.seq, maxlen=padlength)
    valid_x = pad_sequences(valid.seq, maxlen=padlength)
    train_y = train[target_list].values
    valid_y = valid[target_list].values
    test_df['input'] = list(pad_sequences(test_df.seq, maxlen=padlength))
    return train_x, train_y, valid_x, valid_y, test_df[['id','input']]
    
    