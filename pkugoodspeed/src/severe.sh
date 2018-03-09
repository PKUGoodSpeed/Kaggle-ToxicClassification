python main.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/sever_glove.cfg
python fold.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/sever_lexW.cfg
python main.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/sever_lexW.cfg
python fold.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/sever_ft.cfg
python main.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/sever_ft.cfg