# KGE-UNIT


[KGE-UNIT: Towards the unification of molecular interactions prediction based on knowledge graph and multi-task learning on drug discovery]

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

pip install -r requirements.txt

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
dowloads:

https://drive.google.com/file/d/1eWw5PKOi-q7ovPWEAbmVSLnNhIQa6Cwn/view?usp=drive_link


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
python test.py --scenario warm --kge KGE-PATH --kgeunit 'KGE-UNIT-PATH'
```
The pretrained model can be downloaded:
https://doi.org/10.5281/zenodo.8435763


## Comparison

```
python comparison/kge_auroc.py --kge rotate --save
```

--save:        save the model

--kge:         choose the kge model

other codes:
```
python comparison/deepdti_luo.py
python comparison/deepdti_biokg.py
python comparison/ml_luo_ddi.py
python comparison/ml_luo_dti.py
python comparison/svm_biokg_ddi.py
python comparison/svm_biokg_dti.py
```
KGE_NFM:https://github.com/hlf20010508/KGE_NFM
KGNN:   https://github.com/xzenglab/KGNN
