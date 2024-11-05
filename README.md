# SICGNN: Structurally informed convolutional graph neural networks for protein classification
This project is the code for the paper 'SICGNN: Structurally informed convolutional graph neural networks for protein classification' which is currently in review.

The goal of this architecture is utilizing global graph structural information to various conventional convolutional graph neural networks (GraphSAGE/GIN/GAT) for graph classification. 

## Experimental Setup
The experimental environment and versions used are as follows.

```
Anaconda : 23.1.0
Python : 3.7
CUDA : 11.2
PyTorch : 1.7.0
torchvision : 0.8.0
torchaudio : 0.7.0
torch-sparse : 0.6.8
torch-scatter : 2.0.5
torch-cluster : 1.5.9
torch-spline-conv : 1.2.1
```

## Usage
### Data Preparation
`python PrepareDatasets.py DATA/CHEMICAL --dataset-name <name> --outer-k 10`

`cp -r DATA/CHEMICAL/<name> DATA`

< Name > is the name of the dataset : DD / ENZYMES / PROTEINS


### Launch Experiments
`python Launch_Experiments.py --config-file <config> --dataset-name <name> --result-folder <result-folder> --debug`

The experiment is performed on the <name> dataset with the model name <config>, and the results are saved in <result-folder>.

## Note
This code was adapted from and modified based on the code in the paper by Errica et al. below.

[Federico Errica, Marco Podda, Davide Bacciu, Alessio Micheli: A Fair Comparison of Graph Neural Networks for Graph Classification. Proceedings of the 8th International Conference on Learning Representations (ICLR 2020).](https://openreview.net/pdf?id=HygDF6NFPB)
