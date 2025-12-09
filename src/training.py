
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import logging
import numpy as np
from typing import Dict, Optional, Tuple, Any
from tqdm import tqdm

from attacks import create_attacker
from models import create_model

logger = logging.getLogger(__name__)

class BaseTrainer:
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_epochs: int = 90,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 32
    ):
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.model.to(device)
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_stats = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'time_per_epoch': []
        }
        
    def warmup_lr(self, epoch: int, warmup_epochs: int = 5):
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * warmup_factor
    
    def evaluate(self, dataloader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                if hasattr(self.model, 'set_bn_mode'):
                    self.model.set_bn_mode('clean')
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy

class VanillaTrainer(BaseTrainer):
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            if hasattr(self.model, 'set_bn_mode'):
                self.model.set_bn_mode('clean')
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy

class PGDATTrainer(BaseTrainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.attacker = create_attacker(
            'pgd20', self.model, device=self.device
        )
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='PGD-AT Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            adv_images = self.attacker.attack(images, labels)
            
            if hasattr(self.model, 'set_bn_mode'):
                self.model.set_bn_mode('adversarial')
            
            outputs = self.model(adv_images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy

class AdvPropTrainer(BaseTrainer):
    
    def __init__(self, *args, adv_lr: float = 1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.adv_optimizer = optim.Adam(
            self.model.parameters(),
            lr=adv_lr,
            weight_decay=kwargs.get('weight_decay', 1e-4)
        )
        
        self.attacker = create_attacker(
            'pgd20', self.model, device=self.device
        )
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='AdvProp Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            if hasattr(self.model, 'set_bn_mode'):
                self.model.set_bn_mode('clean')
            
            clean_outputs = self.model(images)
            clean_loss = self.criterion(clean_outputs, labels)
            clean_loss.backward()
            self.optimizer.step()
            
            self.adv_optimizer.zero_grad()
            adv_images = self.attacker.attack(images, labels)
            
            if hasattr(self.model, 'set_bn_mode'):
                self.model.set_bn_mode('adversarial')
            
            adv_outputs = self.model(adv_images)
            adv_loss = self.criterion(adv_outputs, labels)
            adv_loss.backward()
            self.adv_optimizer.step()
            
            total_loss += clean_loss.item()
            _, predicted = torch.max(clean_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'clean_loss': f'{clean_loss.item():.4f}',
                'adv_loss': f'{adv_loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy

class EfficientAdvPropPlusTrainer(BaseTrainer):
    
    def __init__(
        self, 
        *args, 
        adv_lr: float = 1e-4,
        clean_to_adv_ratio: float = 0.6,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.adv_optimizer = optim.Adam(
            self.model.parameters(),
            lr=adv_lr,
            weight_decay=kwargs.get('weight_decay', 1e-4)
        )
        
        self.attacker = create_attacker(
            'pgd1', self.model, device=self.device
        )
        
        self.clean_to_adv_ratio = clean_to_adv_ratio
        
        logger.info(f"Efficient AdvProp+ initialized with clean:adv ratio = {clean_to_adv_ratio}")
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Efficient AdvProp+ Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            is_clean_batch = np.random.random() < self.clean_to_adv_ratio
            
            if is_clean_batch:
                self.optimizer.zero_grad()
                if hasattr(self.model, 'set_bn_mode'):
                    self.model.set_bn_mode('clean')
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
            else:
                self.adv_optimizer.zero_grad()
                
                adv_images = self.attacker.attack(images, labels)
                
                if hasattr(self.model, 'set_bn_mode'):
                    self.model.set_bn_mode('adversarial')
                
                outputs = self.model(adv_images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.adv_optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'mode': 'clean' if is_clean_batch else 'adv',
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy

def create_trainer(
    trainer_type: str,
    model: nn.Module,
    device: torch.device,
    **kwargs
) -> BaseTrainer:
    
    
    
    if trainer_type == 'vanilla':
        base_kwargs = {k: v for k, v in kwargs.items() if k != 'adv_lr'}
        return VanillaTrainer(model, device, **base_kwargs)
    elif trainer_type == 'pgd_at':
        base_kwargs = {k: v for k, v in kwargs.items() if k != 'adv_lr'}
        return PGDATTrainer(model, device, **base_kwargs)
    elif trainer_type == 'advprop':
        return AdvPropTrainer(model, device, **kwargs)
    elif trainer_type == 'efficient_advprop_plus':
        return EfficientAdvPropPlusTrainer(model, device, **kwargs)
    else:
        raise ValueError(f"Unknown trainer_type: {trainer_type}")

def train_model(
    trainer: BaseTrainer,
    train_loader,
    val_loader,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    
    
    
    best_val_acc = 0.0
    total_training_time = 0.0
    
    logger.info(f"Starting training for {trainer.num_epochs} epochs...")
    
    for epoch in range(trainer.num_epochs):
        start_time = time.time()
        
        trainer.warmup_lr(epoch, warmup_epochs=5)
        
        train_loss, train_acc = trainer.train_epoch(train_loader)
        
        val_loss, val_acc = trainer.evaluate(val_loader)
        
        trainer.scheduler.step()
        
        epoch_time = time.time() - start_time
        total_training_time += epoch_time
        
        trainer.train_stats['epoch'].append(epoch + 1)
        trainer.train_stats['train_loss'].append(train_loss)
        trainer.train_stats['train_acc'].append(train_acc)
        trainer.train_stats['val_loss'].append(val_loss)
        trainer.train_stats['val_acc'].append(val_acc)
        trainer.train_stats['time_per_epoch'].append(epoch_time)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                torch.save(trainer.model.state_dict(), save_path)
        
        logger.info(
            f"Epoch [{epoch+1}/{trainer.num_epochs}] "
            f"Train: {train_acc:.4f} ({train_loss:.4f}) | "
            f"Val: {val_acc:.4f} ({val_loss:.4f}) | "
            f"Time: {epoch_time:.1f}s"
        )
    
    training_hours = total_training_time / 3600
    logger.info(f"Training completed in {training_hours:.2f} hours")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return {
        'stats': trainer.train_stats,
        'best_val_acc': best_val_acc,
        'total_training_time': total_training_time,
        'training_hours': training_hours
    }

if __name__ == "__main__":
    print("Testing trainer implementations...")
    
    from torch.utils.data import DataLoader, TensorDataset
    
    dummy_images = torch.randn(100, 3, 224, 224)
    dummy_labels = torch.randint(0, 5, (100,))
    dummy_dataset = TensorDataset(dummy_images, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=16)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model('dual_bn', num_classes=5)
    
    trainer_types = ['vanilla', 'efficient_advprop_plus']
    
    for trainer_type in trainer_types:
        print(f"\nTesting {trainer_type} trainer...")
        
        trainer = create_trainer(
            trainer_type,
            model=create_model('dual_bn', num_classes=5),
            device=device,
            num_epochs=1
        )
        
        train_loss, train_acc = trainer.train_epoch(dummy_loader)
        val_loss, val_acc = trainer.evaluate(dummy_loader)
        
        print(f"  Train: {train_acc:.4f} ({train_loss:.4f})")
        print(f"  Val: {val_acc:.4f} ({val_loss:.4f})")
    
    print("Trainer implementation test completed successfully!")