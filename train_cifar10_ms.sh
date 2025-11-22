#!/bin/bash
#SBATCH --job-name=ms_VQVAE
#SBATCH --ntasks=1
#SBATCH --output=vae/slurm/output/%j
#SBATCH --error=vae/slurm/error/%j

#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=lambda-hyperplane

export EXPERIMENT_NAME=vqvae_cifar10_ms
export EXPERIMENT_TAG=${EXPERIMENT_NAME}_${SLURM_JOB_ID}
export DATA_NAME=CIFAR10

# ==============================================================================
# Keys + Directories
# ==============================================================================

export WANDB_API_KEY=53b82424aa495d0f59c8002f14b238bde52b09ed
export TORCH_HOME=/mnt/data0/shared/anton/cache/torch
export AIM_REPO=/mnt/data0/shared/anton/cache/aim
export MPLCONFIGDIR=/mnt/data0/shared/anton/cache/.matplotlib

# self output dirs
export DATA_DIR=/mnt/data0/shared/anton/cache/data
export OUTPUT_DIR=/mnt/data0/shared/anton/cache/vqvae_checkpoints/$EXPERIMENT_TAG

# Fix JAX CUDA
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda/data0/shared/anton/projects/vae

# ==============================================================================
# Environment
# ==============================================================================

cd /mnt/data0/shared/anton/projects/vae
source .venv/bin/activate
which -a python python3

# ==============================================================================
# Training Config
# ==============================================================================

cd vae/train
python train_vqvae_ms.py \
    --exp_name $EXPERIMENT_NAME \
    --tag $SLURM_JOB_ID \
    --data_name $DATA_NAME \
    --epochs 750 \
    --batch_size 256 \
    --lr 1e-3 \
    --log_interval 5 \
    --save_interval 10 \
    --vis_interval 10 \
    --n_fid_samples 1000 \
    --save_dir $OUTPUT_DIR \
    --data_dir $DATA_DIR/$DATA_NAME \
    --aim_repo $AIM_REPO \
    --ch 128 \
    --ch_mult "1,2,4" \
    --num_res_blocks 2 \
    --num_embeddings 1024 \
    --embedding_dim 256 \
    --beta_commit 1.0 \
    --seed 42
    #--resume "/mnt/data0/shared/anton/cache/vqvae_checkpoints/vqvae_cifar10_205880_epoch_900_model.eqx"

deactivate
