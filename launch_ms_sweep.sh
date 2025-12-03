#!/bin/bash

MS_LR_VALUES=(1e-2 5e-3 2.5e-3 1e-3 7.5e-4 5e-4 1e-4)

for lr in "${MS_LR_VALUES[@]}"; do
    sbatch --export=ALL,MS_LR=$lr train_cifar10_ms_sweep.sh
    echo "Submitted job with ms-lr=$lr"
done