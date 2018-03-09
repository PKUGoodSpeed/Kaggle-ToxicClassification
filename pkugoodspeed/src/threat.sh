python fold.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/toxic_ft.cfg
python main.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/toxic_ft.cfg
python fold.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/threat_glove.cfg
python main.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/threat_glove.cfg
python fold.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/threat_lexW.cfg
python main.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/threat_lexW.cfg
python fold.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/threat_ft.cfg
python main.py --train ../data/train_processed.csv --test ../data/test_processed.csv --config cfgs/threat_ft.cfg