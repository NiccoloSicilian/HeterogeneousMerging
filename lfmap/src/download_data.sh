#!/bin/bash
DATA_DIR="/leonardo_scratch/fast/IscrC_eff-SAM2/HeterogeneousMerging/data"
mkdir -p "$DATA_DIR"

python3 -c "
from torchvision import datasets
root = '$DATA_DIR'
print('Downloading MNIST...')
datasets.MNIST(root=root, train=True, download=True)
datasets.MNIST(root=root, train=False, download=True)
print('Downloading FashionMNIST...')
datasets.FashionMNIST(root=root, train=True, download=True)
datasets.FashionMNIST(root=root, train=False, download=True)
print('Done.')
"
