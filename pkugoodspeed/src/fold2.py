'''
testing things
'''
import pandas as pd
import json
from models import KerasModel
import opts_parser
import preprocess
from embedding import *
import key

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
    assert len(target_list) == 1, "Here we forcused to train on one target"
    ## Getting keyword features:
    train = key.get_key_word_features(train, label=target_list[0], vec_size=128)
    train = key.get_key_phrase_features(train, label=target_list[0], vec_size=128)
    test = key.get_key_word_features(test, label=target_list[0], vec_size=128)
    test = key.get_key_phrase_features(test, label=target_list[0], vec_size=128)
    train, valid, test, embedding_matrix = preprocess.embProcess(train, test, **preprc_kargs)

    keras_model = KerasModel(input_shape=[preprc_kargs['padlength']], output_dim=len(target_list))
    model_type = cfg["model_kargs"]["model_type"]
    kargs = dict({
        "max_features": cfg["max_features"],
        "emb_size": cfg["emb_size"],
        "emb_weights": embedding_matrix,
        "kw_dims": [128, 1024, 32],
        "kp_dim": [128, 1024, 32]
    }, **cfg["model_kargs"]["kargs"])
    # model = keras_model.getModel(model_type, **kargs)
    model = keras_model.getCombinedModel(**kargs)
    model.summary()
    history = keras_model.trainCombinedModel(train, valid, target_list, **cfg["train_kargs"])
    
    output_file="{0}_{1}_convergence.png".format(cfg['model_name'], '.'.join(cfg["preprc_kargs"]["target_list"]))
    keras_model.plot(cfg['output_dir'] + '/' + output_file)

    test = keras_model.predictCombinedModel(test, target_list)

    test = test[['id'] + target_list]
    sub_file = "{0}_sum.csv".format(cfg['model_name'])
    test.to_csv(sub_file, index=False)
