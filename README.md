# ADMM Model Pruning

This repository contains the source for ADMM-DS, an extensible library for inducing sparse models under custom constraint conditions. To 

## Installation

For ease of development, ADMM-DS is provided as an easily installable package. To avoid re-installing dependencies, however, the current wheel does not accurately reflect its dependencies, which must be installed separately. The only pre-requisite is a Python (>=3.6) environment and access to pip.

To install the components for ADMM as well as the examples:
```
pip install -r requirements.txt
./install.sh
```

## Running Examples

Example scripts are provided to perform hyperparameter sweeps for both regular training and ADMM training. Note that multi-GPU training is currently not functional, so it is necessary to specify a CUDA device to the training script as shown in the examples. 