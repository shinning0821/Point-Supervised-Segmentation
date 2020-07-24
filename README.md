# lcfcn-pseudo

## 1. Introduction

This project is a re-implementation of PseudoEdgeNet([paper](https://arxiv.org/pdf/1906.02924)) and LCFCN([paper](https://arxiv.org/abs/1807.09856)|[code](https://github.com/ElementAI/LCFCN.git)). 
The LCFCN part contains most of the original code while the PseudoEdgeNet was implemented according to the paper as we weren't able to find existing code from the project.

## 2. Usage

### 2.1 Environment setup

To pre-setup the environment, we recommand to use 
```
conda env create -f environment.yml
```
and
```
pip install -r requirements.txt
```
This command installs pydicom and the [Haven library](https://github.com/ElementAI/haven) which helps in managing the experiments.
### 2.2 Download Dataset


### 2.3 Train and cross-validation
#### 2.3.1 Example

Train and validate the lcfcn model with TNBC dataset
```
python trainval.py -d TNBC -e exp_config_lcfcn.json -r 1
```

Train and validate the pseudoedgenet model with MoNuSeg dataset
```
python trainval.py -d MoNuSegTrainingData -e exp_config_penet.json -r 1
```

Both [`trainval.py`](trainval.py) and [`trainval.ipynb`](trainval.ipynb) can be used to train and validate the model. In [`trainval.ipynb`](trainval.ipynb), you can get a brief introduction of the setup of the dataset.

#### 2.3.2 Module location

The models are defined in [`src/models`](src/models). The dataset is defined in [`src/datasets/__init__.py`](src/datasets/__init__.py). The loss function is defined in [`lcfcn/lcfcn_loss.py`](lcfcn/lcfcn_loss.py). The function `compute_weighted_crossentropy` is used in the PseudoEdgeNet model. The `compute_lcfcn_loss` is used in LCFCN model.
