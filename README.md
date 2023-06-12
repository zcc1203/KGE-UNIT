# KGE-UNIT


KGE-UNIT: Towards the unification of molecular interactions prediction based on knowledge graph and multi-task learning on drug discovery


Environment

Pytorch: 1.8.1

torchvision:0.9.1

KGE-UNIT for Luo's datasets


Steps

1 

download data.zip features.zip weights.zip from baiduyun

https://pan.baidu.com/s/1wfSRhR88ap6z6mY_vhzqrg
code:kgeu

2
Unzip data/data.zip

Unzip features/features.zip

Unzip weights/weights.zip

---------------------------------------
data

----luo

features

----test_dense_features0_ddi.csv

----...

weights

----dismult_400_warm_1_10_ddi_luo_0.pkl

----...

2 

mkdir res

python train.py --lr 0.02 --epochs 40

3

If you want to retrain DistMult, you can use AmpliGraph by tensorflow.

Environment

tensorflow 1.5.0

ampligraph 1.3.2






