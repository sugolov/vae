#!/bin/bash
#SBATCH --job-name=vdvae
#SBATCH --ntasks=1
#SBATCH --output=slurm/output/%j
#SBATCH --error=slurm/error/%j

#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1

which -a python python3
readlink -f "$(command -v python3)"

cd /mnt/data0/shared/anton/projects/vdvae
source .venv/bin/activate

python train.py --hps cifar10

deactivate