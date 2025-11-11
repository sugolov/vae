import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from vae.data import build_dataset

def test_cifar10():
    """Quick test for CIFAR10 dataset builder"""
    
    print("Testing CIFAR10 dataset builder...")
    
    # Build dataset
    dataloader, num_classes, n_train, image_size = build_dataset(
        dataset_name='cifar10',
        data_dir='./data',
        batch_size=32,
        is_train=True,
        num_workers=2
    )
    
    print(f"Dataset info:")
    print(f"  Classes: {num_classes}")
    print(f"  Training samples: {n_train}")
    print(f"  Batches: {n_train // 32}")
    
    # Test one batch
    images, labels = next(iter(dataloader))
    print(f"\nFirst batch:")
    print(f"  Images: {images.shape}, type: {type(images)}")
    print(f"  Labels: {labels.shape}, type: {type(labels)}")
    print(f"  Memory: {images.nbytes / 1024**2:.2f} MB")
    
    # Basic assertions
    assert num_classes == 10, f"Expected 10 classes, got {num_classes}"
    assert n_train == 50000, f"Expected 50000 training samples, got {n_train}"
    assert images.shape == (32, 3, 32, 32), f"Wrong image shape: {images.shape}"
    assert labels.shape == (32,), f"Wrong label shape: {labels.shape}"
    
    print("\nTest passed!")