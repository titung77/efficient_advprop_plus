
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from typing import Tuple, Dict
import logging
import random
import tensorflow_datasets as tfds

logging.basicConfig(level=logging.INFO)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

class MemoryEfficientTFFlowersDataset(Dataset):
    
    def __init__(self, split_name: str, transform=None, data_dir='./data'):
        self.split_name = split_name
        self.transform = transform
        self.data_dir = data_dir
        
        logger.info(f"Initializing {split_name} dataset...")
        
        self.tf_dataset = tfds.load(
            'tf_flowers',
            split='train',
            as_supervised=True,
            data_dir=data_dir,
            download=True,
            try_gcs=False
        )
        
        self._create_index_splits()
        
    def _create_index_splits(self):
        total_samples = 3670
        
        random.seed(42)
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
        
        if self.split_name == 'train':
            self.indices = indices[:train_size]
        elif self.split_name == 'val':
            self.indices = indices[train_size:train_size + val_size]
        else:
            self.indices = indices[train_size + val_size:]
        
        logger.info(f"{self.split_name} split: {len(self.indices)} samples")
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        try:
            item = list(self.tf_dataset.skip(actual_idx).take(1))[0]
            image, label = item
            
            image_np = image.numpy()
            image_pil = Image.fromarray(image_np)
            
            if self.transform:
                image_pil = self.transform(image_pil)
                
            return image_pil, label.numpy()
            
        except Exception as e:
            logger.warning(f"Error loading image {actual_idx}: {e}")
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, 0

def get_transforms(split: str = 'train', image_size: int = 224):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
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

def get_efficient_data_loaders(batch_size: int = 4, num_workers: int = 0) -> Dict[str, DataLoader]:
    logger.info("Creating memory-efficient TF-Flowers data loaders...")
    
    train_transform = get_transforms('train')
    eval_transform = get_transforms('eval')
    
    train_dataset = MemoryEfficientTFFlowersDataset('train', train_transform)
    val_dataset = MemoryEfficientTFFlowersDataset('val', eval_transform)
    test_dataset = MemoryEfficientTFFlowersDataset('test', eval_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    logger.info(f"Memory-efficient data loaders created:")
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val: {len(val_dataset)} samples") 
    logger.info(f"  Test: {len(test_dataset)} samples")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

if __name__ == "__main__":
    print("Testing memory-efficient TF-Flowers loading...")
    
    data_loaders = get_efficient_data_loaders(batch_size=2, num_workers=0)
    
    print("Testing first batch...")
    for split_name, loader in data_loaders.items():
        print(f"\nTesting {split_name}...")
        for batch_idx, (images, labels) in enumerate(loader):
            print(f"  Batch {batch_idx}: {images.shape}, {labels}")
            if batch_idx >= 1:
                break
    
    print("Memory-efficient TF-Flowers test completed!")