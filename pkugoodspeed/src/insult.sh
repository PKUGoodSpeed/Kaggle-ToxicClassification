python fold.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/insult_glove.cfg
python main.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/insult_glove.cfg
python fold.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/insult_lexW.cfg
python main.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/insult_lexW.cfg
python fold.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/insult_ft.cfg
python main.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/insult_ft.cfg