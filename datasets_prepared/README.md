# datasets_prepared

This directory contains intermediate datasets generated during the experimental pipeline.

Note: This directory contains only scripts and folder structures by default. All data files are generated automatically by running `REDI.py`.

## Structure

- `train/`  
  Training datasets (`{DATASET}_train.csv`) used to train the classifiers

- `test_accuracy/`  
  Test datasets (`{DATASET}_test.csv`) used to evaluate classification accuracy

- `occ_table/`  
  Occurrence tables (`{DATASET}_occ_table.csv`) constructed from training data,  
  representing observed 2-way (pairwise) feature value combinations for validity checking

- `test_IFr/`  
  Test inputs (`{DATASET}_{PROTECTED}_test_IFr_set.csv`) used to compute IFr

- `test_valid_IFr/`  
  Test inputs (`{DATASET}_{PROTECTED}_test_valid_IFr_set.csv`) used to compute valid-IFr

## Generation

For each dataset (and protected attribute where applicable), the following scripts are executed:

- `split_train_test.py`  
  Generates training and test datasets

- `make_occ_table.py`  
  Constructs occurrence tables for validity checking

- `make_test_IFr.py`  
  Generates test inputs for IFr

- `make_test_valid_IFr.py`  
  Generates valid test inputs for valid-IFr