
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TFFlowersDataset:
    
    def __init__(self, data_dir: str = './data'):
        self.data_dir = data_dir
        self.num_classes = 5
        self.image_size = 224
        self.classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def download_and_prepare(self):
        logger.info("Downloading TF-Flowers dataset...")
        
        ds, info = tfds.load(
            'tf_flowers',
            with_info=True,
            as_supervised=True,
            data_dir=self.data_dir
        )
        
        logger.info(f"Dataset info: {info}")
        logger.info(f"Total samples: {info.splits['train'].num_examples}")
        
        train_data = []
        train_labels = []
        
        for image, label in ds['train']:
            image_np = image.numpy()
            train_data.append(image_np)
            train_labels.append(label.numpy())
            
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        
        logger.info(f"Data shape: {train_data.shape}")
        logger.info(f"Labels shape: {train_labels.shape}")
        
        return train_data, train_labels
    
    def get_transforms(self, split: str = 'train'):
        if split == 'train':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        
        return transform
    
    def create_stratified_split(self, data, labels) -> Tuple[np.ndarray, ...]:
        from sklearn.model_selection import train_test_split
        
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            data, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        val_data, test_data, val_labels, test_labels = train_test_split(
            temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        logger.info(f"Train split: {len(train_data)} samples")
        logger.info(f"Validation split: {len(val_data)} samples")
        logger.info(f"Test split: {len(test_data)} samples")
        
        return train_data, val_data, test_data, train_labels, val_labels, test_labels

class FlowersDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders(batch_size: int = 32, num_workers: int = 4) -> Dict[str, DataLoader]:
    
    tf_flowers = TFFlowersDataset()
    
    data, labels = tf_flowers.download_and_prepare()
    
    train_data, val_data, test_data, train_labels, val_labels, test_labels = tf_flowers.create_stratified_split(data, labels)
    
    train_transform = tf_flowers.get_transforms('train')
    eval_transform = tf_flowers.get_transforms('eval')
    
    train_dataset = FlowersDataset(train_data, train_labels, train_transform)
    val_dataset = FlowersDataset(val_data, val_labels, eval_transform)
    test_dataset = FlowersDataset(test_data, test_labels, eval_transform)
    
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
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

if __name__ == "__main__":
    print("Testing TF-Flowers data loading...")
    
    data_loaders = get_data_loaders(batch_size=32)
    
    for split, loader in data_loaders.items():
        print(f"{split.upper()} - Batches: {len(loader)}, Total samples: {len(loader.dataset)}")
        
        for batch_idx, (images, labels) in enumerate(loader):
            print(f"  Batch {batch_idx}: Images shape: {images.shape}, Labels shape: {labels.shape}")
            if batch_idx == 0:
                break