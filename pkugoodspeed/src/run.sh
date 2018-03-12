python fold.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/obscene_glove.cfg
python main.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/obscene_glove.cfg
python fold.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/obscene_lexW.cfg
python main.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/obscene_lexW.cfg