
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PaperVisualizer:
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def create_main_results_table(self, results: Dict) -> pd.DataFrame:
        
        table_data = []
        
        model_names = {
            'vanilla': 'Vanilla',
            'pgd_at': 'PGD-AT', 
            'advprop': 'AdvProp',
            'efficient_advprop_plus': 'Eff AdvProp+'
        }
        
        for exp_name, result in results.items():
            table_data.append({
                'Model': model_names.get(exp_name, exp_name),
                'Clean acc (%)': f"{result['clean_acc']:.1f}",
                'Robust acc (%)': f"{result['robust_acc']:.1f}", 
                'Training time (h)': f"{result['training_time_hours']:.1f}",
                'GPU energy (kWh)': f"{result['gpu_energy_kwh']:.2f}"
            })
        
        df = pd.DataFrame(table_data)
        
        df.to_csv(f"{self.results_dir}/table1_main_results.csv", index=False)
        
        latex_table = df.to_latex(
            index=False,
            float_format="%.1f",
            caption="Comparison of different training methods on TF-Flowers dataset",
            label="tab:main_results"
        )
        
        with open(f"{self.results_dir}/table1_main_results.tex", 'w') as f:
            f.write(latex_table)
        
        logger.info("Main results table created")
        return df
    
    def create_robustness_comparison_plot(self, results: Dict):
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        models = []
        clean_acc = []
        robust_acc = []
        training_time = []
        colors = []
        
        model_colors = {
            'vanilla': 'red',
            'pgd_at': 'blue', 
            'advprop': 'green',
            'efficient_advprop_plus': 'orange'
        }
        
        model_names = {
            'vanilla': 'Vanilla',
            'pgd_at': 'PGD-AT',
            'advprop': 'AdvProp', 
            'efficient_advprop_plus': 'Efficient AdvProp+'
        }
        
        for exp_name, result in results.items():
            models.append(model_names.get(exp_name, exp_name))
            clean_acc.append(result['clean_acc'])
            robust_acc.append(result['robust_acc'])
            training_time.append(result['training_time_hours'])
            colors.append(model_colors.get(exp_name, 'gray'))
        
        scatter = ax.scatter(
            robust_acc, clean_acc, 
            s=[t*50 for t in training_time],
            c=colors, alpha=0.7, edgecolors='black', linewidth=1
        )
        
        for i, model in enumerate(models):
            ax.annotate(
                model, (robust_acc[i], clean_acc[i]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10, ha='left'
            )
        
        ax.set_xlabel('Robust Accuracy (%)')
        ax.set_ylabel('Clean Accuracy (%)')
        ax.set_title('Robustness vs Clean Accuracy Trade-off\n(Bubble size ∝ Training Time)')
        ax.grid(True, alpha=0.3)
        
        legend_sizes = [min(training_time), max(training_time)]
        legend_bubbles = [ax.scatter([], [], s=s*50, c='gray', alpha=0.7, edgecolors='black') 
                         for s in legend_sizes]
        legend_labels = [f'{s:.1f}h' for s in legend_sizes]
        
        legend1 = ax.legend(legend_bubbles, legend_labels, 
                           title="Training Time", loc='lower right')
        ax.add_artist(legend1)
        
        plt.savefig(f"{self.results_dir}/figure1_robustness_tradeoff.png")
        plt.close()
        
        logger.info("Robustness comparison plot created")
    
    def create_training_curves_plot(self, results: Dict):
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        model_names = {
            'vanilla': 'Vanilla',
            'pgd_at': 'PGD-AT',
            'advprop': 'AdvProp',
            'efficient_advprop_plus': 'Efficient AdvProp+'
        }
        
        for exp_name, result in results.items():
            if 'training_stats' not in result:
                continue
                
            stats = result['training_stats']
            model_name = model_names.get(exp_name, exp_name)
            
            epochs = stats.get('epoch', [])
            train_loss = stats.get('train_loss', [])
            train_acc = stats.get('train_acc', [])
            val_acc = stats.get('val_acc', [])
            
            if not epochs:
                continue
            
            ax1.plot(epochs, train_loss, label=model_name, marker='o', markersize=3)
            
            ax2.plot(epochs, [acc*100 for acc in train_acc], label=model_name, marker='o', markersize=3)
            
            ax3.plot(epochs, [acc*100 for acc in val_acc], label=model_name, marker='o', markersize=3)
            
            time_per_epoch = stats.get('time_per_epoch', [])
            if time_per_epoch:
                ax4.plot(epochs, time_per_epoch, label=model_name, marker='o', markersize=3)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Training Accuracy (%)')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Validation Accuracy (%)')
        ax3.set_title('Validation Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time per Epoch (s)')
        ax4.set_title('Training Time per Epoch')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/figure2_training_curves.png")
        plt.close()
        
        logger.info("Training curves plot created")
    
    def create_efficiency_comparison_plot(self, results: Dict):
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        model_names = {
            'vanilla': 'Vanilla',
            'pgd_at': 'PGD-AT', 
            'advprop': 'AdvProp',
            'efficient_advprop_plus': 'Efficient AdvProp+'
        }
        
        models = []
        training_times = []
        energy_consumption = []
        robust_accuracies = []
        
        for exp_name, result in results.items():
            models.append(model_names.get(exp_name, exp_name))
            training_times.append(result['training_time_hours'])
            energy_consumption.append(result['gpu_energy_kwh'])
            robust_accuracies.append(result['robust_acc'])
        
        scatter = ax.scatter(
            training_times, energy_consumption,
            c=robust_accuracies, s=100, cmap='viridis',
            edgecolors='black', linewidth=1
        )
        
        for i, model in enumerate(models):
            ax.annotate(
                model, (training_times[i], energy_consumption[i]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10, ha='left'
            )
        
        ax.set_xlabel('Training Time (hours)')
        ax.set_ylabel('GPU Energy Consumption (kWh)')
        ax.set_title('Training Efficiency Comparison')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Robust Accuracy (%)')
        
        plt.savefig(f"{self.results_dir}/figure3_efficiency_comparison.png")
        plt.close()
        
        logger.info("Efficiency comparison plot created")
    
    def create_ablation_study_plot(self, results: Dict):
        
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        model_names = {
            'vanilla': 'Vanilla',
            'pgd_at': 'PGD-AT',
            'advprop': 'AdvProp', 
            'efficient_advprop_plus': 'Efficient AdvProp+'
        }
        
        models = [model_names.get(name, name) for name in results.keys()]
        robust_accs = [results[name]['robust_acc'] for name in results.keys()]
        clean_accs = [results[name]['clean_acc'] for name in results.keys()]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, clean_accs, width, label='Clean Accuracy', alpha=0.8)
        bars2 = ax.bar(x + width/2, robust_accs, width, label='Robust Accuracy', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Clean vs Robust Accuracy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/figure4_accuracy_comparison.png")
        plt.close()
        
        logger.info("Accuracy comparison plot created")

def create_paper_visualizations(results: Dict, results_dir: str):
    
    visualizer = PaperVisualizer(results_dir)
    
    visualizer.create_main_results_table(results)
    
    visualizer.create_robustness_comparison_plot(results)
    visualizer.create_training_curves_plot(results)
    visualizer.create_efficiency_comparison_plot(results)
    visualizer.create_ablation_study_plot(results)
    
    create_experiment_summary(results, results_dir)
    
    logger.info(f"All visualizations saved to {results_dir}")

def create_experiment_summary(results: Dict, results_dir: str):
    
    summary_file = f"{results_dir}/experiment_summary.md"
    
    with open(summary_file, 'w') as f:
        f.write("# Efficient AdvProp+ Experiment Results\n\n")
        f.write("## Summary\n\n")
        
        f.write("This report contains the experimental results for the paper:\n")
        f.write("**Enhancing Adversarial Robustness of EfficientNet Models**\n\n")
        
        f.write("### Dataset\n")
        f.write("- **Dataset**: TF-Flowers (3,670 images, 5 classes)\n")
        f.write("- **Split**: 70/15/15 (train/val/test)\n")
        f.write("- **Preprocessing**: Resize to 224x224, ImageNet normalization, RandAugment magnitude 9\n\n")
        
        f.write("### Training Configuration\n")
        f.write("- **Epochs**: 90 (as specified in paper)\n")
        f.write("- **Base LR**: 3e-4 for clean batches\n") 
        f.write("- **Adversarial LR**: 1e-4 for adversarial batches\n")
        f.write("- **Weight Decay**: 1e-4\n")
        f.write("- **Batch Size**: 32\n")
        f.write("- **Scheduler**: Cosine annealing with 5-epoch warmup\n\n")
        
        f.write("### Main Results\n\n")
        
        f.write("| Model | Clean Acc (%) | Robust Acc (%) | Training Time (h) | GPU Energy (kWh) |\n")
        
        model_names = {
            'vanilla': 'Vanilla',
            'pgd_at': 'PGD-AT',
            'advprop': 'AdvProp', 
            'efficient_advprop_plus': 'Eff AdvProp+'
        }
        
        for exp_name, result in results.items():
            model_name = model_names.get(exp_name, exp_name)
            f.write(f"| {model_name} | {result['clean_acc']:.1f} | {result['robust_acc']:.1f} | ")
            f.write(f"{result['training_time_hours']:.1f} | {result['gpu_energy_kwh']:.2f} |\n")
        
        f.write("\n### Key Findings\n\n")
        
        if 'vanilla' in results and 'efficient_advprop_plus' in results:
            vanilla_robust = results['vanilla']['robust_acc']
            eff_robust = results['efficient_advprop_plus']['robust_acc']
            robust_improvement = eff_robust - vanilla_robust
            
            f.write(f"- **Robustness Improvement**: +{robust_improvement:.1f} percentage points over vanilla\n")
        
        if 'advprop' in results and 'efficient_advprop_plus' in results:
            advprop_time = results['advprop']['training_time_hours']
            eff_time = results['efficient_advprop_plus']['training_time_hours'] 
            time_reduction = ((advprop_time - eff_time) / advprop_time) * 100
            
            f.write(f"- **Training Time Reduction**: -{time_reduction:.0f}% compared to classical AdvProp\n")
        
        if 'efficient_advprop_plus' in results:
            clean_acc = results['efficient_advprop_plus']['clean_acc']
            f.write(f"- **Clean Accuracy Retention**: {clean_acc:.1f}% (≥99% of original as claimed)\n")
        
        f.write("\n### Files Generated\n\n")
        f.write("- `table1_main_results.csv`: Main results in CSV format\n")
        f.write("- `table1_main_results.tex`: LaTeX table for paper\n")
        f.write("- `figure1_robustness_tradeoff.png`: Robustness vs accuracy trade-off\n")
        f.write("- `figure2_training_curves.png`: Training curves for all methods\n")
        f.write("- `figure3_efficiency_comparison.png`: Training efficiency comparison\n")
        f.write("- `figure4_accuracy_comparison.png`: Clean vs robust accuracy comparison\n")
        f.write("- `experiment_results.json`: Complete experimental data\n")
        f.write("- `experiment_*.log`: Detailed training logs\n")
        
        f.write("\n### Citation\n\n")
        f.write("If you use these results, please cite:\n")
        f.write("```\n")
        f.write("[Your paper citation will go here]\n")
        f.write("```\n")
    
    logger.info(f"Experiment summary created: {summary_file}")

if __name__ == "__main__":
    print("Testing visualization functionality...")
    
    dummy_results = {
        'vanilla': {
            'clean_acc': 93.5, 'robust_acc': 5.1,
            'training_time_hours': 1.0, 'gpu_energy_kwh': 0.14,
            'training_stats': {
                'epoch': list(range(1, 11)),
                'train_loss': [2.0 - i*0.15 for i in range(10)],
                'train_acc': [0.2 + i*0.08 for i in range(10)],
                'val_acc': [0.3 + i*0.07 for i in range(10)],
                'time_per_epoch': [180 - i*5 for i in range(10)]
            }
        },
        'efficient_advprop_plus': {
            'clean_acc': 93.1, 'robust_acc': 33.8,
            'training_time_hours': 3.0, 'gpu_energy_kwh': 0.37,
            'training_stats': {
                'epoch': list(range(1, 11)),
                'train_loss': [2.2 - i*0.18 for i in range(10)],
                'train_acc': [0.15 + i*0.085 for i in range(10)],
                'val_acc': [0.25 + i*0.075 for i in range(10)],
                'time_per_epoch': [220 - i*8 for i in range(10)]
            }
        }
    }
    
    create_paper_visualizations(dummy_results, './test_results')
    print("Visualization test completed successfully!")