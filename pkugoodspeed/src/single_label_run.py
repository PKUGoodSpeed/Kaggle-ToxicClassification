'''
testing things
'''
from models import SingleLabelModel
import plot
import pandas as pd
import json
import parse_opts

if __name__ == '__main__':
    train_data_file, test_data_file, config_file = parse_opts.getopts()
    train = pd.read_csv(train_data_file)
    test = pd.read_csv(test_data_file)
    
    ## Read From Config file
    cfg = json.load(open(config_file))
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
    
    
    