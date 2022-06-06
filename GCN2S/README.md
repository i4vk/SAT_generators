# Example Run

You can try out the following 4 steps one by one.

1. Preprocess data
```bash
python eval/conversion.py --src dataset/train_formulas/ -s dataset/train_set/
python eval/conversion.py --src dataset/test_formulas/ -s dataset/test_set/
```

2. Train G2SAT
```bash
python main_train.py --epoch_num 201
```
After this step, trained G2SAT models will be saved in `model/` directory.

3. Use G2SAT to generate Formulas
```bash
python main_test.py --epoch_load 200
```
After this step, generated graphs will be saved to `graphs/` directory. 1 graph is generated out of 1 template.

Graphs will be saved in 2 formats: a single `.dat` file containing all the generated graphs; a directory where each generated graph is saved as a single `.dat` file. 

(It may take fairly long time: Runing G2SAT is fast, but updating networkx takes the majority of time in current implementation.)

We can then generate CNF formulas from the generated graphs
```bash
python conversion.py --src graphs/GCN_3_32_preTrue_dropFalse_yield1_019501.120000_0.dat --store-dir formulas --action=lcg2sat
```