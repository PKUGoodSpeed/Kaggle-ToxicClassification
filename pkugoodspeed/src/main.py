'''
testing things
'''
import pandas as pd
import json
from models import KerasModel
import opts_parser
import preprocess
from embedding import *

if __name__ == '__main__':
    train_data_file, test_data_file, config_file = opts_parser.getopts()
    train = pd.read_csv(train_data_file)
    test = pd.read_csv(test_data_file)
    
    ## Read From Config file
    cfg = json.load(open(config_file))
    print cfg
    embedding_index = None
    if cfg["vectorizor"] == "embedding":
        assert "embedding_file" in cfg, "No embedding_file in config."
        embedding_index = emb_from_emb(cfg["embedding_file"])
    
    preprc_kargs = dict({
        "max_features": cfg["max_features"],
        "emb_size": cfg["emb_size"],
        "embedding_index": embedding_index
    }, **cfg["preprc_kargs"])
    
    train_x, train_y, valid_x, valid_y, test, embedding_matrix = preprocess.embProcess(train, test, **preprc_kargs)
    print train_x.shape
    print train_y.shape
    print valid_x.shape
    print valid_y.shape

    keras_model = KerasModel(input_shape=train_x[0].shape, output_dim=len(train_y[0]))
    model_type = cfg["model_kargs"]["model_type"]
    kargs = dict({
        "max_features": cfg["max_features"],
        "emb_size": cfg["emb_size"],
        "emb_weights": embedding_matrix
    }, **cfg["model_kargs"]["kargs"])
    model = keras_model.getModel(model_type, **kargs)
    model.summary()
    history = keras_model.train(train_x, train_y, valid_x, valid_y, **cfg["train_kargs"])
    
    output_file="{0}_{1}_convergence.png".format(cfg['model_name'], '.'.join(cfg["preprc_kargs"]["target_list"]))
    keras_model.plot(cfg['output_dir'] + '/' + output_file)
