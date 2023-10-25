# KGE-UNIT


[KGE-UNIT: Towards the unification of molecular interactions prediction based on knowledge graph and multi-task learning on drug discovery
KGE-UNIT for Luo's datasets]

## Getting Started

### Installation

Setup conda environment:
```
# Create Environment
conda create -n kge-unit python=3.8 -y
conda activate kge-unit
# Install requirements
pip install pykeen
```


## Prepare Datasets
```
${DATA_ROOT}
├── luo
│   ├── data_folds
│   │   ├──warm_start_1_1
│   │   ├──warm_start_1_1_ddi
│   │   ├──cold_start_1_1
│   │   ├──cold_start_1_1_ddi
|   ├── cv_train_1.csv
├── 2
    ...
├── 5
├── add_feature.json
├── pos_sample.csv
├── neg_sample.csv
├── independent.csv
```


/***************************Structure***************************/

data/

----luo/

features/

----test_dense_features0_ddi.csv

----...

weights/

----dismult_400_warm_1_10_ddi_luo_0.pkl

----...

/*************************************************************/

Steps 3

  mkdir res

  python train.py --lr 0.02 --epochs 40

Steps 4

  If you want to retrain DistMult, you can use AmpliGraph by tensorflow.

  Environment

    tensorflow 1.5.0

    ampligraph 1.3.2






