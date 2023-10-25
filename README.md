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
or:

pip install -r requirments.txt

## Prepare Datasets
```
${DATA_ROOT}
├── luo
│   ├── data_folds
│   │   ├──warm_start_1_1
│   │   ├──warm_start_1_1_ddi
│   │   ├──cold_start_1_1
│   │   ├──cold_start_1_1_ddi
│   │   ├──KGE_COLD_PATH
│   │   ├──KGE_PATH
│   │   ├──kg.csv
│   │   ├──kg_all.txt
│   │   ├──kg_cold.csv
│   │   ├──...
    ...
├── BioKG
│   ├── data_folds
│   │   ├──warm_start_1_1
│   │   ├──warm_start_1_1_ddi
│   │   ├──kg.csv
```
[datasets](http://www.cnblogs.com/sxdcgaq8080/p/7894828.html)
## Training
To train KGE-UNIT for cross-validation:
```
python train.py --epochs 40 --save --batch-size 32 --scenario warm
```
--save:        save the best model

--epochs:      the total epoches of training

--batch-size:  the batch size of the training, the default value is 32

--scenario:   the warm or cold scenario for the drugs, value: warm/cold

## Testing

To test KGE-UNIT by the existing model:

```

```

## Ablation



