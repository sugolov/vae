import os
import json

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

class SimCLRWrapper(Dataset):
    """Minimal wrapper to create positive pairs."""
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get image twice with different augmentations
        x_i, label = self.dataset[idx]
        x_j, _ = self.dataset[idx]
        return x_i, x_j, label
    

def simclr_collate(batch):
    """Collate function for SimCLR pairs - returns numpy arrays in BHWC format"""
    x_i = torch.stack([item[0] for item in batch]).numpy()  # (B, C, H, W)
    x_j = torch.stack([item[1] for item in batch]).numpy()  # (B, C, H, W)
    
    # Transpose from (B, C, H, W) to (B, H, W, C)
    x_i = np.ascontiguousarray(np.transpose(x_i, (0, 2, 3, 1)))  # (B, H, W, C)
    x_j = np.ascontiguousarray(np.transpose(x_j, (0, 2, 3, 1)))  # (B, H, W, C)
    
    labels = np.array([item[2] for item in batch])
    
    return x_i, x_j, labels


def numpy_collate(batch):
    """Collate function to convert batch to numpy arrays in BHWC format"""
    images, labels = zip(*batch)
    images = torch.stack(images).numpy()  # (B, C, H, W)
    images = np.ascontiguousarray(np.transpose(images, (0, 2, 3, 1)))  # (B, H, W, C)
    labels = np.array(labels)
    return images, labels


def build_dataset(
    dataset_name,
    data_dir,
    batch_size=32,
    is_train=True,
    num_workers=4,
    simclr=False  
):
    dataset_name = dataset_name.upper()
    
    # Determine image size based on dataset
    if dataset_name in ['CIFAR10', 'CIFAR', 'CIFAR100']:
        image_size = 32  # CIFAR is 32x32
    elif dataset_name in ['IMAGENET', 'IMNET']:
        image_size = 224  # Standard ImageNet size
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Build transforms
    if is_train:
        if simclr:
            # SimCLR augmentations from the paper
            color_jitter = transforms.ColorJitter(0.7, 0.7, 0.7, 0.2)
            transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ])
        else:
            # Your original transforms
            transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
    else:
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
    
    # Create dataset
    if dataset_name == 'CIFAR10':
        dataset = datasets.CIFAR10(data_dir, train=is_train, transform=transform, download=True)
        num_classes = 10
    elif dataset_name in ['CIFAR', 'CIFAR100']:
        dataset = datasets.CIFAR100(data_dir, train=is_train, transform=transform, download=True)
        num_classes = 100
    elif dataset_name in ['IMAGENET', 'IMNET']:
        root = os.path.join(data_dir, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        num_classes = 1000
    
    # Wrap dataset for SimCLR if needed
    if simclr and is_train:
        dataset = SimCLRWrapper(dataset)
    
    # Get dataset size
    n_samples = len(dataset)
    
    # Modify collate function for SimCLR
    if simclr and is_train:
        collate_fn = simclr_collate
    else:
        collate_fn = numpy_collate
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
        collate_fn=collate_fn
    )
    
    return dataloader, num_classes, n_samples, image_size