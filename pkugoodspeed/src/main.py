'''
testing things
'''
import pandas as pd
import json
from models import KerasModel
import opts_parser
import preprocess

if __name__ == '__main__':
    train_data_file, test_data_file, config_file = opts_parser.getopts()
    train = pd.read_csv(train_data_file)
    test = pd.read_csv(test_data_file)
    
    ## Read From Config file
    cfg = json.load(open(config_file))
    print cfg
    
    # target_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    target_list = ["toxic"]
    preprc_kargs = {
        'target_list': target_list,
        'split_ratio': 0.7,
        'expand_ratio': 2.0,
        'padlength': 150
    }
    train_x, train_y, valid_x, valid_y, test = preprocess.embProcess(train, test, **preprc_kargs)
    print train_x.shape
    print train_y.shape
    print valid_x.shape
    print valid_y.shape
    print test[:3]
    keras_model = KerasModel(input_shape=train_x[0].shape, output_dim=len(train_y[0]))
    model = keras_model.getModel('sequential', [])
    model.summary()
    history = keras_model.train(train_x, train_y, valid_x, valid_y)
    
    
    
'''
    
    target = cfg["target"]
    padlength = cfg["padlength"]
    vectorize = cfg["vectorize"]
    splitratio = cfg["splitratio"]
    
    single = SingleLabelModel(target=target, padlength=padlength, vectorize=vectorize, splitratio=splitratio)
    single.loadData(train, test)
    model = single.getSequentialModel(cfg['layers'])
    model.summary()
    res = single.trainModel(model, cfg['train_args'])
    plot.plotResult(res, filename=cfg['fig_file'])
    
    
# Make training data set be more balanced
def _makeBalance(df, target_list, expand_ratio=2.):
    L = len(df)
    while len(df) < expand_ratio * L:
        for target in target_list:
            minor = df[df[target] != 0]
            df = df.append(minor, ignore_index=True)
    return df
'''