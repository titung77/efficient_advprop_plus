
import torch
import torch.nn as nn
import numpy as np
import time
import json
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from attacks import create_attacker, evaluate_robustness

logger = logging.getLogger(__name__)

class ModelEvaluator:
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def evaluate_clean_accuracy(
        self, 
        model: nn.Module, 
        dataloader: torch.utils.data.DataLoader
    ) -> float:
        model.to(self.device)
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating clean accuracy"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                if hasattr(model, 'set_bn_mode'):
                    model.set_bn_mode('clean')
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        logger.info(f"Clean accuracy: {accuracy:.4f}")
        return accuracy
    
    def evaluate_robust_accuracy(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        attack_type: str = 'pgd20',
        eps: float = 8/255
    ) -> float:
        model.to(self.device)
        model.eval()
        
        attacker = create_attacker(
            attack_type, model, eps=eps, device=self.device
        )
        
        correct = 0
        total = 0
        
        for images, labels in tqdm(dataloader, desc=f"Evaluating robust accuracy ({attack_type})"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            adv_images = attacker.attack(images, labels)
            
            with torch.no_grad():
                if hasattr(model, 'set_bn_mode'):
                    model.set_bn_mode('adversarial')
                
                outputs = model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        logger.info(f"Robust accuracy ({attack_type}): {accuracy:.4f}")
        return accuracy
    
    def measure_training_time(
        self,
        trainer,
        train_loader,
        val_loader,
        num_epochs: int = 5
    ) -> Dict[str, float]:
        
        original_epochs = trainer.num_epochs
        trainer.num_epochs = num_epochs
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            train_loss, train_acc = trainer.train_epoch(train_loader)
            
            val_loss, val_acc = trainer.evaluate(val_loader)
            
            epoch_time = time.time() - epoch_start
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | "
                f"Time: {epoch_time:.1f}s"
            )
        
        total_time = time.time() - start_time
        
        estimated_full_time = (total_time / num_epochs) * 90
        
        trainer.num_epochs = original_epochs
        
        return {
            'measured_time_hours': total_time / 3600,
            'estimated_90_epochs_hours': estimated_full_time / 3600,
            'time_per_epoch_seconds': total_time / num_epochs
        }
    
    def estimate_gpu_energy(
        self,
        training_time_hours: float,
        gpu_type: str = 'T4'
    ) -> float:
        gpu_power = {
            'T4': 70,
            'V100': 250,
            'A100': 250,
            'RTX3060': 170,
            'RTX3080': 220,
            'RTX4090': 450
        }
        
        if gpu_type not in gpu_power:
            logger.warning(f"Unknown GPU type {gpu_type}, using T4 estimate")
            gpu_type = 'T4'
        
        energy_kwh = (gpu_power[gpu_type] * training_time_hours) / 1000
        
        logger.info(f"Estimated energy consumption ({gpu_type}): {energy_kwh:.3f} kWh")
        return energy_kwh
    
    def run_comprehensive_evaluation(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        model_name: str
    ) -> Dict[str, float]:
        
        logger.info(f"Running comprehensive evaluation for {model_name}...")
        
        results = {
            'model_name': model_name,
        }
        
        results['clean_acc'] = self.evaluate_clean_accuracy(model, test_loader)
        
        results['robust_acc_pgd20'] = self.evaluate_robust_accuracy(
            model, test_loader, 'pgd20'
        )
        
        results['robust_acc_pgd1'] = self.evaluate_robust_accuracy(
            model, test_loader, 'pgd1'
        )
        
        results['robust_acc_fgsm'] = self.evaluate_robust_accuracy(
            model, test_loader, 'fgsm'
        )
        
        return results

class ExperimentRunner:
    
    def __init__(self, results_dir: str = './results'):
        self.results_dir = results_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = ModelEvaluator(self.device)
        self.experiment_results = {}
        
        logger.info(f"Experiment runner initialized, using device: {self.device}")
    
    def run_single_experiment(
        self,
        trainer_type: str,
        train_loader,
        val_loader, 
        test_loader,
        training_config: Dict
    ) -> Dict:
        
        from training import create_trainer, train_model
        from models import create_model
        
        logger.info(f"Running experiment: {trainer_type}")
        
        model_type = 'dual_bn' if trainer_type in ['advprop', 'efficient_advprop_plus'] else 'vanilla'
        model = create_model(model_type, num_classes=5)
        
        trainer = create_trainer(
            trainer_type,
            model=model,
            device=self.device,
            **training_config
        )
        
        timing_results = self.evaluator.measure_training_time(
            trainer, train_loader, val_loader, num_epochs=5
        )
        
        training_config_full = training_config.copy()
        training_config_full['num_epochs'] = 10
        
        trainer_full = create_trainer(
            trainer_type,
            model=create_model(model_type, num_classes=5),
            device=self.device,
            **training_config_full
        )
        
        training_results = train_model(
            trainer_full,
            train_loader,
            val_loader,
            save_path=f"{self.results_dir}/{trainer_type}_best.pth"
        )
        
        eval_results = self.evaluator.run_comprehensive_evaluation(
            trainer_full.model,
            test_loader,
            trainer_type
        )
        
        gpu_energy = self.evaluator.estimate_gpu_energy(
            timing_results['estimated_90_epochs_hours']
        )
        
        results = {
            'trainer_type': trainer_type,
            'clean_acc': eval_results['clean_acc'] * 100,
            'robust_acc': eval_results['robust_acc_pgd20'] * 100,
            'training_time_hours': timing_results['estimated_90_epochs_hours'],
            'gpu_energy_kwh': gpu_energy,
            'training_stats': training_results['stats'],
            'detailed_eval': eval_results
        }
        
        return results
    
    def run_all_experiments(
        self,
        data_loaders: Dict,
        save_results: bool = True
    ) -> Dict:
        
        training_config = {
            'num_epochs': 90,
            'learning_rate': 3e-4,
            'weight_decay': 1e-4,
            'batch_size': 32
        }
        
        experiments = [
            'vanilla',
            'pgd_at', 
            'advprop',
            'efficient_advprop_plus'
        ]
        
        all_results = {}
        
        for trainer_type in experiments:
            try:
                results = self.run_single_experiment(
                    trainer_type,
                    data_loaders['train'],
                    data_loaders['val'],
                    data_loaders['test'],
                    training_config
                )
                
                all_results[trainer_type] = results
                
                logger.info(f"Results for {trainer_type}:")
                logger.info(f"  Clean accuracy: {results['clean_acc']:.1f}%")
                logger.info(f"  Robust accuracy: {results['robust_acc']:.1f}%")
                logger.info(f"  Training time: {results['training_time_hours']:.1f} hours")
                logger.info(f"  GPU energy: {results['gpu_energy_kwh']:.2f} kWh")
                
            except Exception as e:
                logger.error(f"Error in {trainer_type} experiment: {e}")
                continue
        
        if save_results:
            self.save_results(all_results)
        
        return all_results
    
    def save_results(self, results: Dict):
        import os
        os.makedirs(self.results_dir, exist_ok=True)
        
        with open(f"{self.results_dir}/experiment_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.create_results_table(results)
        
        logger.info(f"Results saved to {self.results_dir}")
    
    def create_results_table(self, results: Dict):
        
        table_data = []
        for trainer_type, result in results.items():
            table_data.append({
                'Model': trainer_type.replace('_', ' ').title(),
                'Clean acc (%)': f"{result['clean_acc']:.1f}",
                'Robust acc (%)': f"{result['robust_acc']:.1f}",
                'Training time (h)': f"{result['training_time_hours']:.1f}",
                'GPU energy (kWh)': f"{result['gpu_energy_kwh']:.2f}"
            })
        
        df = pd.DataFrame(table_data)
        
        df.to_csv(f"{self.results_dir}/results_table.csv", index=False)
        
        print("\n" + "="*80)
        print("EXPERIMENTAL RESULTS TABLE")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        return df

if __name__ == "__main__":
    print("Testing evaluation functionality...")
    
    from models import create_model
    from torch.utils.data import DataLoader, TensorDataset
    
    dummy_images = torch.randn(50, 3, 224, 224)
    dummy_labels = torch.randint(0, 5, (50,))
    test_dataset = TensorDataset(dummy_images, dummy_labels)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    model = create_model('dual_bn', num_classes=5)
    
    evaluator = ModelEvaluator()
    
    print("Testing clean accuracy evaluation...")
    clean_acc = evaluator.evaluate_clean_accuracy(model, test_loader)
    print(f"Clean accuracy: {clean_acc:.4f}")
    
    print("Testing robust accuracy evaluation...")
    robust_acc = evaluator.evaluate_robust_accuracy(model, test_loader, 'pgd1')
    print(f"Robust accuracy: {robust_acc:.4f}")
    
    print("Evaluation test completed successfully!")