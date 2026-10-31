"""
Visualization module for CViTS-Net training results.
Generates training plots, ROC curves, and confusion matrices.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json

# Use non-interactive backend
matplotlib.use('Agg')


class TrainingVisualizer:
    """Generate visualizations for training results."""
    
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_history(self, history: dict, metric_name: str):
        """
        Plot training history for a single metric.
        
        Args:
            history: Dictionary with 'train' and 'val' keys
            metric_name: Name of the metric (e.g., 'loss', 'accuracy')
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(history['train']) + 1)
        
        ax.plot(epochs, history['train'], 'b-', label=f'Train {metric_name}', linewidth=2)
        ax.plot(epochs, history['val'], 'r-', label=f'Val {metric_name}', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name.capitalize(), fontsize=12)
        ax.set_title(f'{metric_name.capitalize()} vs Epoch', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        filename = self.output_dir / f'{metric_name}_vs_epoch.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
    
    def plot_all_metrics(self, history: dict):
        """
        Plot all metrics on a single figure.
        
        Args:
            history: History dictionary with multiple metrics
        """
        metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1_score']
        
        for metric in metrics:
            if metric in history:
                self.plot_training_history(history[metric], metric)
    
    def plot_roc_curve(self, fpr: dict, tpr: dict, roc_auc: dict):
        """
        Plot ROC curves for all classes.
        
        Args:
            fpr: False positive rates for each class
            tpr: True positive rates for each class
            roc_auc: AUC scores for each class
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(fpr)))
        
        for i, (fpr_class, tpr_class) in enumerate(zip(fpr.values(), tpr.values())):
            ax.plot(fpr_class, tpr_class, color=colors[i], lw=2,
                   label=f'Class {i} (AUC = {roc_auc[i]:.3f})')
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves for All Classes', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        filename = self.output_dir / 'roc_curve.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: list = None):
        """
        Plot confusion matrix as a heatmap.
        
        Args:
            cm: Confusion matrix
            class_names: Names of classes (optional)
        """
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Normalized Count'}, ax=ax)
        
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        filename = self.output_dir / 'confusion_matrix.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
    
    def plot_metrics_comparison(self, metrics_history: dict):
        """
        Plot multiple metrics in a grid layout.
        
        Args:
            metrics_history: Dictionary with metric histories
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'roc_auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            if metric in metrics_history:
                train_vals = metrics_history[metric]['train']
                val_vals = metrics_history[metric]['val']
                
                epochs = range(1, len(train_vals) + 1)
                
                axes[idx].plot(epochs, train_vals, 'b-', label=f'Train', linewidth=2)
                axes[idx].plot(epochs, val_vals, 'r-', label=f'Val', linewidth=2)
                axes[idx].set_xlabel('Epoch', fontsize=10)
                axes[idx].set_ylabel(metric.capitalize(), fontsize=10)
                axes[idx].set_title(f'{metric.capitalize()} vs Epoch', fontsize=11, fontweight='bold')
                axes[idx].legend(fontsize=9)
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = self.output_dir / 'all_metrics.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
    
    def save_metrics_json(self, metrics_data: dict, filename: str = 'metrics.json'):
        """
        Save metrics to JSON file.
        
        Args:
            metrics_data: Dictionary of metrics
            filename: Output filename
        """
        filepath = self.output_dir / filename
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj
        
        metrics_data = convert_types(metrics_data)
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        
        print(f"Saved: {filepath}")


class LivePlotter:
    """Real-time plotting during training (optional)."""
    
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_epoch_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict, output_file: str = None):
        """
        Plot metrics for current epoch.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics for this epoch
            val_metrics: Validation metrics for this epoch
            output_file: Optional output file path
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        metrics_names = list(train_metrics.keys())
        x_pos = np.arange(len(metrics_names))
        
        train_vals = list(train_metrics.values())
        val_vals = list(val_metrics.values())
        
        width = 0.35
        axes[0].bar(x_pos - width/2, train_vals, width, label='Train', alpha=0.8)
        axes[0].bar(x_pos + width/2, val_vals, width, label='Val', alpha=0.8)
        axes[0].set_xlabel('Metric', fontsize=11)
        axes[0].set_ylabel('Value', fontsize=11)
        axes[0].set_title(f'Epoch {epoch} - Metrics Comparison', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([m.capitalize() for m in metrics_names], rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Training info panel
        axes[1].axis('off')
        info_text = f"Epoch: {epoch}\n\nTrain Metrics:\n"
        for name, val in train_metrics.items():
            info_text += f"  {name}: {val:.4f}\n"
        info_text += f"\nValidation Metrics:\n"
        for name, val in val_metrics.items():
            info_text += f"  {name}: {val:.4f}\n"
        
        axes[1].text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if output_file is None:
            output_file = self.output_dir / f'epoch_{epoch:03d}.png'
        
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
