
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LazyTFFlowersDataset(Dataset):
    
    def __init__(self, split_name: str, transform=None, data_dir='./data'):
        self.split_name = split_name
        self.transform = transform
        self.data_dir = data_dir
        
        self.ds_info = tfds.builder('tf_flowers').info
        
        self.tf_dataset = tfds.load(
            'tf_flowers',
            split='train',
            as_supervised=True,
            data_dir=data_dir
        )
        
        self._prepare_splits()
        
    def _prepare_splits(self):
        self.data_indices = []
        
        for idx, (_, label) in enumerate(self.tf_dataset.take(-1)):
            self.data_indices.append((idx, label.numpy()))
            
        from sklearn.model_selection import train_test_split
        
        indices = [item[0] for item in self.data_indices]
        labels = [item[1] for item in self.data_indices]
        
        train_idx, temp_idx, train_labels, temp_labels = train_test_split(
            indices, labels, test_size=0.3, stratify=labels, random_state=42
        )
        
        val_idx, test_idx, val_labels, test_labels = train_test_split(
            temp_idx, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
        )
        
        if self.split_name == 'train':
            self.indices = train_idx
        elif self.split_name == 'val':
            self.indices = val_idx
        else:
            self.indices = test_idx
            
        logger.info(f"{self.split_name} split: {len(self.indices)} samples")
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        item = list(self.tf_dataset.skip(actual_idx).take(1))[0]
        image, label = item
        
        image_np = image.numpy()
        image_pil = Image.fromarray(image_np)
        
        if self.transform:
            image_pil = self.transform(image_pil)
            
        return image_pil, label.numpy()

def get_transforms(split: str = 'train', image_size: int = 224):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

def get_data_loaders(batch_size: int = 32, num_workers: int = 4) -> Dict[str, DataLoader]:
    logger.info("Creating memory-efficient data loaders...")
    
    train_transform = get_transforms('train')
    eval_transform = get_transforms('eval')
    
    train_dataset = LazyTFFlowersDataset('train', train_transform)
    val_dataset = LazyTFFlowersDataset('val', eval_transform)
    test_dataset = LazyTFFlowersDataset('test', eval_transform)
    
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"Data loaders created:")
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val: {len(val_dataset)} samples") 
    logger.info(f"  Test: {len(test_dataset)} samples")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

if __name__ == "__main__":
    print("Testing memory-efficient data loading...")
    
    data_loaders = get_data_loaders(batch_size=2, num_workers=0)
    
    print("Testing first batch...")
    for batch_idx, (images, labels) in enumerate(data_loaders['train']):
        print(f"Batch {batch_idx}: images {images.shape}, labels {labels.shape}")
        if batch_idx >= 2:
            break
            
    print("Memory-efficient loading test completed!")