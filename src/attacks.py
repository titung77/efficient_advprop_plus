
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

class PGDAttack:
    
    def __init__(
        self,
        model: nn.Module,
        eps: float = 8/255,
        alpha: float = 2/255,
        steps: int = 20,
        random_start: bool = True,
        targeted: bool = False,
        device: Optional[torch.device] = None
    ):
        
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.targeted = targeted
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        logger.info(f"PGD Attack initialized: eps={eps:.4f}, alpha={alpha:.4f}, steps={steps}")
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        
            
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        adv_images = images.clone().detach()
        
        if self.random_start:
            delta = torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images + delta, 0, 1).detach()
        
        for step in range(self.steps):
            adv_images.requires_grad_(True)
            
            if hasattr(self.model, 'set_bn_mode'):
                self.model.set_bn_mode('adversarial')
            outputs = self.model(adv_images)
            
            loss = F.cross_entropy(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            grad_sign = adv_images.grad.sign()
            
            if self.targeted:
                adv_images = adv_images.detach() - self.alpha * grad_sign
            else:
                adv_images = adv_images.detach() + self.alpha * grad_sign
            
            delta = torch.clamp(adv_images - images, -self.eps, self.eps)
            adv_images = torch.clamp(images + delta, 0, 1).detach()
        
        return adv_images

class PGD1Attack(PGDAttack):
    
    def __init__(
        self,
        model: nn.Module,
        eps: float = 8/255,
        alpha: float = 8/255,
        device: Optional[torch.device] = None
    ):
        super().__init__(
            model=model,
            eps=eps,
            alpha=alpha,
            steps=1,
            random_start=True,
            targeted=False,
            device=device
        )
        
        logger.info(f"PGD-1 Attack initialized with random starts")

class PGD20Attack(PGDAttack):
    
    def __init__(
        self,
        model: nn.Module,
        eps: float = 8/255,
        alpha: float = 2/255,
        device: Optional[torch.device] = None
    ):
        super().__init__(
            model=model,
            eps=eps,
            alpha=alpha,
            steps=20,
            random_start=True,
            targeted=False,
            device=device
        )
        
        logger.info(f"PGD-20 Attack initialized for robust evaluation")

class FGSM:
    
    def __init__(
        self,
        model: nn.Module,
        eps: float = 8/255,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.eps = eps
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        images.requires_grad_(True)
        
        if hasattr(self.model, 'set_bn_mode'):
            self.model.set_bn_mode('adversarial')
        outputs = self.model(images)
        
        loss = F.cross_entropy(outputs, labels)
        
        self.model.zero_grad()
        loss.backward()
        
        grad_sign = images.grad.sign()
        adv_images = images + self.eps * grad_sign
        
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images.detach()

def create_attacker(attack_type: str, model: nn.Module, **kwargs):
    
    
    
    if attack_type == 'pgd1':
        return PGD1Attack(model, **kwargs)
    elif attack_type == 'pgd20':
        return PGD20Attack(model, **kwargs)
    elif attack_type == 'fgsm':
        return FGSM(model, **kwargs)
    else:
        raise ValueError(f"Unknown attack_type: {attack_type}")

def evaluate_robustness(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    attack_type: str = 'pgd20',
    device: Optional[torch.device] = None
) -> float:
    
        
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    attacker = create_attacker(attack_type, model, device=device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            adv_images = attacker.attack(images, labels)
            
            if hasattr(model, 'set_bn_mode'):
                model.set_bn_mode('adversarial')
            outputs = model(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    robust_accuracy = correct / total
    logger.info(f"Robust accuracy ({attack_type}): {robust_accuracy:.4f}")
    
    return robust_accuracy

if __name__ == "__main__":
    print("Testing PGD attack implementations...")
    
    from models import create_model
    model = create_model('dual_bn', num_classes=5)
    
    batch_size = 4
    test_images = torch.randn(batch_size, 3, 224, 224)
    test_labels = torch.randint(0, 5, (batch_size,))
    
    print("Testing PGD-1 attack...")
    pgd1 = create_attacker('pgd1', model)
    adv_images_1 = pgd1.attack(test_images, test_labels)
    print(f"PGD-1 perturbation L∞ norm: {torch.max(torch.abs(adv_images_1 - test_images)).item():.6f}")
    
    print("Testing PGD-20 attack...")
    pgd20 = create_attacker('pgd20', model)
    adv_images_20 = pgd20.attack(test_images, test_labels)
    print(f"PGD-20 perturbation L∞ norm: {torch.max(torch.abs(adv_images_20 - test_images)).item():.6f}")
    
    print("Testing FGSM attack...")
    fgsm = create_attacker('fgsm', model)
    adv_images_fgsm = fgsm.attack(test_images, test_labels)
    print(f"FGSM perturbation L∞ norm: {torch.max(torch.abs(adv_images_fgsm - test_images)).item():.6f}")
    
    print("Attack implementation test completed successfully!")