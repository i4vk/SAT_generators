# Example Run

1. Preprocess data
```bash
python eval/conversion.py --src dataset/train_formulas/ -s dataset/train_set/
python eval/conversion.py --src dataset/test_formulas/ -s dataset/test_set/
```

2. Train EGNN2S
```bash
python main_train.py --epoch_num 201
```
After this step, trained EGNN2S models will be saved in `model/` directory.

3. Use EGNN2S to generate Formulas
```bash
python main_test.py --epoch_load 200
```
After this step, generated graphs will be saved to `graphs/` directory. 1 graph is generated out of 1 template.

Graphs will be saved in 2 formats: a single `.dat` file containing all the generated graphs; a directory where each generated graph is saved as a single `.dat` file. 

(It may take fairly long time: Runing EGNN2S is fast, but updating networkx takes the majority of time in current implementation.)

We can then generate CNF formulas from the generated graphs
```bash
python eval/conversion.py --src graphs/GCN_3_32_preTrue_dropFalse_yield1_019501.120000_0.dat --store-dir formulas --action=fg2sat
```