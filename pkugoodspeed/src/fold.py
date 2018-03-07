'''
K-folder stacking usage
'''
import pandas as pd
import json
from models import KerasModel
import opts_parser
import preprocess
from embedding import *
from sklearn.cross_validation import KFold

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
    
    target_list = preprc_kargs['target_list']
    
    kf = KFold(len(train), n_folds=5, random_state=414)
    results = pd.DataFrame()
    for i, (train_index, valid_index) in enumerate(kf):
        train_fold = train.iloc[train_index]
        test_fold = train.iloc[valid_index][["id", "comment_text"]]
        train_fold, valid_fold, test_fold, embedding_matrix = preprocess.embProcess(train_fold, test_fold, **preprc_kargs)
        keras_model = KerasModel(input_shape=[preprc_kargs['padlength']], output_dim=len(target_list))
        model_type = cfg["model_kargs"]["model_type"]
        kargs = dict({
            "max_features": cfg["max_features"],
            "emb_size": cfg["emb_size"],
            "emb_weights": embedding_matrix
        }, **cfg["model_kargs"]["kargs"])
        model = keras_model.getModel(model_type, **kargs)
        model.summary()
        cfg["train_kargs"]["check_file"] = cfg['model_name']+"cv.h5"
    
        history = keras_model.train(train_fold, valid_fold, target_list, **cfg["train_kargs"])
        test_fold = keras_model.make_predict(test_fold, target_list)[['id'] + target_list]
        results = results.append(test_fold, ignore_index=True)
    stacking_file = "{0}_stk.csv".format(cfg['model_name'])
    assert(set(results.id.tolist()) == set(train.id.tolist()))
    results = pd.merge(pd.DataFrame({'id': train.id}), results, on='id')
    for a, b in zip(results.id.values, train.id.values):
        assert a == b, "NMB!!!"
    results.to_csv(stacking_file, index=False)
