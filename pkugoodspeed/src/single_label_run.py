'''
testing things
'''
from models import SingleLabelModel
import plot
import pandas as pd
import json

if __name__ == '__main__':
    train = pd.read_csv('../data/train_processed.csv')
    test = pd.read_csv('../data/train_processed.csv')
    
    ## Read From Config file
    cfg = json.load(open('cfgs/cnn.cfg'))
    print cfg
    
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
    
    
    