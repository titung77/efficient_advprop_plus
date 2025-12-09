
import torch
import torch.nn as nn
import timm
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DualBatchNorm2d(nn.Module):
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super(DualBatchNorm2d, self).__init__()
        
        self.bn_clean = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        
        self.bn_adv = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        
        self.training_mode = 'clean'
    
    def set_mode(self, mode: str):
        if mode not in ['clean', 'adversarial']:
            raise ValueError(f"Mode must be 'clean' or 'adversarial', got {mode}")
        self.training_mode = mode
    
    def forward(self, x):
        if self.training_mode == 'clean':
            return self.bn_clean(x)
        else:
            return self.bn_adv(x)

class EfficientNetB0WithDualBN(nn.Module):
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(EfficientNetB0WithDualBN, self).__init__()
        
        self.backbone = timm.create_model(
            'efficientnet_b0', 
            pretrained=pretrained, 
            num_classes=num_classes
        )
        
        self._replace_bn_layers()
        
        self.num_classes = num_classes
        
        logger.info(f"Created EfficientNet-B0 with dual BN, num_classes={num_classes}")
    
    def _replace_bn_layers(self):
        
        def replace_bn_recursive(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    dual_bn = DualBatchNorm2d(
                        child.num_features,
                        eps=child.eps,
                        momentum=child.momentum
                    )
                    
                    if child.weight is not None:
                        dual_bn.bn_clean.weight.data.copy_(child.weight.data)
                        dual_bn.bn_adv.weight.data.copy_(child.weight.data)
                    if child.bias is not None:
                        dual_bn.bn_clean.bias.data.copy_(child.bias.data)
                        dual_bn.bn_adv.bias.data.copy_(child.bias.data)
                    
                    dual_bn.bn_clean.running_mean.data.copy_(child.running_mean.data)
                    dual_bn.bn_clean.running_var.data.copy_(child.running_var.data)
                    dual_bn.bn_adv.running_mean.data.copy_(child.running_mean.data)
                    dual_bn.bn_adv.running_var.data.copy_(child.running_var.data)
                    
                    setattr(module, name, dual_bn)
                else:
                    replace_bn_recursive(child)
        
        replace_bn_recursive(self.backbone)
        logger.info("Replaced all BatchNorm2d layers with DualBatchNorm2d")
    
    def set_bn_mode(self, mode: str):
        def set_mode_recursive(module):
            for child in module.children():
                if isinstance(child, DualBatchNorm2d):
                    child.set_mode(mode)
                else:
                    set_mode_recursive(child)
        
        set_mode_recursive(self)
    
    def forward(self, x):
        return self.backbone(x)

class EfficientNetB0Vanilla(nn.Module):
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(EfficientNetB0Vanilla, self).__init__()
        
        self.backbone = timm.create_model(
            'efficientnet_b0', 
            pretrained=pretrained, 
            num_classes=num_classes
        )
        
        self.num_classes = num_classes
        
        logger.info(f"Created Vanilla EfficientNet-B0, num_classes={num_classes}")
    
    def forward(self, x):
        return self.backbone(x)
    
    def set_bn_mode(self, mode: str):
        pass

def create_model(model_type: str, num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    
    
    
    if model_type == 'vanilla':
        return EfficientNetB0Vanilla(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'dual_bn':
        return EfficientNetB0WithDualBN(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def count_parameters(model: nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

if __name__ == "__main__":
    print("Testing model creation...")
    
    vanilla_model = create_model('vanilla', num_classes=5)
    vanilla_params = count_parameters(vanilla_model)
    print(f"Vanilla EfficientNet-B0: {vanilla_params}")
    
    dual_bn_model = create_model('dual_bn', num_classes=5)
    dual_bn_params = count_parameters(dual_bn_model)
    print(f"Dual BN EfficientNet-B0: {dual_bn_params}")
    
    test_input = torch.randn(2, 3, 224, 224)
    
    print("Testing vanilla model forward pass...")
    with torch.no_grad():
        output = vanilla_model(test_input)
        print(f"Vanilla output shape: {output.shape}")
    
    print("Testing dual BN model forward pass...")
    dual_bn_model.set_bn_mode('clean')
    with torch.no_grad():
        output_clean = dual_bn_model(test_input)
        print(f"Dual BN (clean) output shape: {output_clean.shape}")
    
    dual_bn_model.set_bn_mode('adversarial') 
    with torch.no_grad():
        output_adv = dual_bn_model(test_input)
        print(f"Dual BN (adversarial) output shape: {output_adv.shape}")
    
    print("Model creation test completed successfully!")