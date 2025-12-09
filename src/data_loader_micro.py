
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from typing import Tuple, Dict
import logging
import requests
import zipfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MicroFlowersDataset(Dataset):
    
    def __init__(self, split_name: str, transform=None, data_dir='./micro_data'):
        self.split_name = split_name
        self.transform = transform
        self.data_dir = data_dir
        
        self._prepare_micro_dataset()
        
        total_samples = len(self.image_paths)
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
        
        if split_name == 'train':
            self.indices = list(range(train_size))
        elif split_name == 'val':
            self.indices = list(range(train_size, train_size + val_size))
        else:
            self.indices = list(range(train_size + val_size, total_samples))
        
        logger.info(f"{split_name} split: {len(self.indices)} samples")
    
    def _prepare_micro_dataset(self):
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.image_paths = []
        self.labels = []
        
        class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for img_idx in range(20):
                img_path = os.path.join(class_dir, f"{class_name}_{img_idx:02d}.png")
                
                if not os.path.exists(img_path):
                    np.random.seed(class_idx * 100 + img_idx)
                    color = np.random.rand(3) * 255
                    
                    image = np.zeros((224, 224, 3), dtype=np.uint8)
                    image[:, :] = color.astype(np.uint8)
                    
                    if class_idx == 0:
                        image[100:124, 100:124] = [255, 255, 0]
                    elif class_idx == 1:
                        image[:, :] = [255, 255, 0]
                    elif class_idx == 2:
                        image[:, :] = [255, 0, 0]
                    elif class_idx == 3:
                        image[:, :] = [255, 255, 0]
                        image[100:124, 100:124] = [139, 69, 19]
                    else:
                        image[:, :] = [255, 182, 193]
                    
                    pil_img = Image.fromarray(image)
                    pil_img.save(img_path)
                
                self.image_paths.append(img_path)
                self.labels.append(class_idx)
        
        logger.info(f"Created micro dataset with {len(self.image_paths)} synthetic images")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        image_path = self.image_paths[actual_idx]
        label = self.labels[actual_idx]
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

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

def get_micro_data_loaders(batch_size: int = 16, num_workers: int = 0) -> Dict[str, DataLoader]:
    logger.info("Creating micro data loaders...")
    
    train_transform = get_transforms('train')
    eval_transform = get_transforms('eval')
    
    train_dataset = MicroFlowersDataset('train', train_transform)
    val_dataset = MicroFlowersDataset('val', eval_transform)
    test_dataset = MicroFlowersDataset('test', eval_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    logger.info(f"Micro data loaders created:")
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val: {len(val_dataset)} samples") 
    logger.info(f"  Test: {len(test_dataset)} samples")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

if __name__ == "__main__":
    print("Testing micro data loading...")
    
    data_loaders = get_micro_data_loaders(batch_size=4, num_workers=0)
    
    print("Testing batches from each split...")
    
    for split_name, loader in data_loaders.items():
        print(f"\nTesting {split_name} loader...")
        for batch_idx, (images, labels) in enumerate(loader):
            print(f"  Batch {batch_idx}: images {images.shape}, labels {labels}")
            if batch_idx >= 1:
                break
            
    print("\nMicro data loading test completed successfully!")