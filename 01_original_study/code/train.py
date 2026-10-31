"""
Complete CViTS-Net Training Pipeline (PyTorch)
Trains the model on APTOS2019 dataset with full metrics and visualizations.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import time
import traceback
from typing import Tuple

from data.dataset import DRDataset
from data.transforms import get_transforms
from models.lamca_net import LAMCANet
from losses.combined_loss import CombinedLoss
from utils.metrics import MetricsCalculator
from utils.visualize import TrainingVisualizer


# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LAMCANetTrainer:
    """Main training orchestrator for LAMCA-Net."""
    def __init__(self, 
                 train_csv: str = "APTOS2019/train.csv",
                 val_csv: str = "APTOS2019/valid.csv",
                 test_csv: str = "APTOS2019/test.csv",
                 img_dir: str = "APTOS2019/train_images",
                 output_dir: str = "results",
                 model_dir: str = "trained_model",
                 batch_size: int = 16,
                 epochs: int = 100,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0001,
                 num_workers: int = 4,
                 max_retries: int = 3):
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.img_dir = img_dir
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.max_retries = max_retries
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.logs_dir = self.output_dir / "logs"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.history = {}
        self.test_history = {}
        self.visualizer = TrainingVisualizer(str(self.plots_dir))
        
    def load_data_with_retry(self) -> bool:
        for attempt in range(self.max_retries):
            try:
                print(f"\n{'='*80}")
                print(f"Loading dataset (Attempt {attempt + 1}/{self.max_retries})")
                print(f"{'='*80}")
                train_dataset = DRDataset(self.train_csv, self.img_dir, transform=get_transforms('train'))
                val_dataset = DRDataset(self.val_csv, self.img_dir, transform=get_transforms('val'))
                test_dataset = DRDataset(self.test_csv, self.img_dir, transform=get_transforms('test'))
                self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
                self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
                self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
                print("Dataset loaded successfully!")
                return True
            except Exception as e:
                print(f"Data loading attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to load dataset after {self.max_retries} attempts")
                    traceback.print_exc()
                    return False
        return False
    
    def build_model(self) -> bool:
        try:
            print(f"\n{'='*80}")
            print("Building LAMCA-Net Model")
            print(f"{'='*80}")
            self.model = LAMCANet(num_classes=5).to(device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            self.criterion = CombinedLoss()
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"\nTotal Trainable Parameters: {total_params:,}")
            print(f"Device: {device}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            return True
        except Exception as e:
            print(f"Error building model: {str(e)}")
            traceback.print_exc()
            return False
    
    def train(self) -> bool:
        try:
            print(f"\n{'='*80}")
            print(f"Starting Training ({self.epochs} epochs)")
            print(f"{'='*80}\n")
            
            self.history = {
                'loss': {'train': [], 'val': []},
                'accuracy': {'train': [], 'val': []},
                'precision': {'train': [], 'val': []},
                'recall': {'train': [], 'val': []},
                'f1_score': {'train': [], 'val': []},
                'specificity': {'train': [], 'val': []},
                'roc_auc': {'train': [], 'val': []},
                'qwk': {'train': [], 'val': []}
            }
            
            for epoch in range(self.epochs):
                print(f"\nEpoch {epoch + 1}/{self.epochs}")
                print("-" * 80)
                
                # Training phase
                self.model.train()
                train_metrics_calc = MetricsCalculator(num_classes=5)
                train_loss = 0.0
                num_batches = 0
                
                for batch_images, batch_labels in self.train_loader:
                    batch_images = batch_images.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    predictions = self.model(batch_images)
                    loss = F.cross_entropy(predictions, batch_labels)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    # Track metrics
                    train_loss += loss.item()
                    pred_proba = F.softmax(predictions, dim=1).detach().cpu().numpy()
                    train_metrics_calc.update(batch_labels.cpu().numpy(), pred_proba)
                    num_batches += 1
                
                train_loss = train_loss / num_batches
                train_metrics = train_metrics_calc.compute_metrics()
                
                # Validation phase
                self.model.eval()
                val_metrics_calc = MetricsCalculator(num_classes=5)
                val_loss = 0.0
                num_val_batches = 0
                
                with torch.no_grad():
                    for batch_images, batch_labels in self.val_loader:
                        batch_images = batch_images.to(device)
                        batch_labels = batch_labels.to(device)
                        
                        predictions = self.model(batch_images)
                        loss = F.cross_entropy(predictions, batch_labels)
                        
                        val_loss += loss.item()
                        pred_proba = F.softmax(predictions, dim=1).cpu().numpy()
                        val_metrics_calc.update(batch_labels.cpu().numpy(), pred_proba)
                        num_val_batches += 1
                
                val_loss = val_loss / num_val_batches
                val_metrics = val_metrics_calc.compute_metrics()
                
                # Store history
                self.history['loss']['train'].append(float(train_loss))
                self.history['loss']['val'].append(float(val_loss))
                
                for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'roc_auc', 'qwk']:
                    if metric_name in train_metrics:
                        self.history[metric_name]['train'].append(train_metrics[metric_name])
                        self.history[metric_name]['val'].append(val_metrics[metric_name])
                
                # Print epoch results
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"Train Acc: {train_metrics.get('accuracy', 0):.4f} | Val Acc: {val_metrics.get('accuracy', 0):.4f}")
                print(f"Train F1: {train_metrics.get('f1_score', 0):.4f} | Val F1: {val_metrics.get('f1_score', 0):.4f}")
                print(f"Train ROC-AUC: {train_metrics.get('roc_auc', 0):.4f} | Val ROC-AUC: {val_metrics.get('roc_auc', 0):.4f}")
                print(f"Train QWK: {train_metrics.get('qwk', 0):.4f} | Val QWK: {val_metrics.get('qwk', 0):.4f}")
                
                # Save model periodically
                if (epoch + 1) % 10 == 0:
                    self._save_checkpoint(epoch + 1)
            
            print(f"\n{'='*80}")
            print("Training completed!")
            print(f"{'='*80}\n")
            
            return True
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            traceback.print_exc()
            return False
    
    def evaluate_test_set(self) -> bool:
        try:
            print(f"\n{'='*80}")
            print("Evaluating on Test Set")
            print(f"{'='*80}\n")
            
            self.model.eval()
            test_metrics_calc = MetricsCalculator(num_classes=5)
            test_loss = 0.0
            num_test_batches = 0
            
            with torch.no_grad():
                for batch_images, batch_labels in self.test_loader:
                    batch_images = batch_images.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    predictions = self.model(batch_images)
                    loss = F.cross_entropy(predictions, batch_labels)
                    
                    test_loss += loss.item()
                    pred_proba = F.softmax(predictions, dim=1).cpu().numpy()
                    test_metrics_calc.update(batch_labels.cpu().numpy(), pred_proba)
                    num_test_batches += 1
            
            test_loss = test_loss / num_test_batches
            test_metrics = test_metrics_calc.compute_metrics()
            
            self.test_history = {
                'loss': float(test_loss),
                'metrics': test_metrics
            }
            
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
            print(f"Test Precision: {test_metrics.get('precision', 0):.4f}")
            print(f"Test Recall: {test_metrics.get('recall', 0):.4f}")
            print(f"Test F1 Score: {test_metrics.get('f1_score', 0):.4f}")
            print(f"Test Specificity: {test_metrics.get('specificity', 0):.4f}")
            print(f"Test ROC-AUC: {test_metrics.get('roc_auc', 0):.4f}")
            print(f"Test QWK: {test_metrics.get('qwk', 0):.4f}")
            
            # Get confusion matrix and ROC curves
            cm = test_metrics_calc.get_confusion_matrix()
            roc_data = test_metrics_calc.get_roc_curves()
            
            self.test_history['confusion_matrix'] = cm.tolist()
            self.test_history['roc_data'] = {k: {kk: v.tolist() for kk, v in vv.items()} 
                                            for k, vv in roc_data.items()}
            
            return True, test_metrics_calc, cm, roc_data
            
        except Exception as e:
            print(f"Error during test evaluation: {str(e)}")
            traceback.print_exc()
            return False, None, None, None
    
    def save_model(self) -> bool:
        try:
            print(f"\n{'='*80}")
            print("Saving Model and Training History")
            print(f"{'='*80}\n")
            
            # Save model (PyTorch state_dict)
            model_path = self.model_dir / "cvitsnet_aptos2019.pth"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, str(model_path))
            print(f"Model saved: {model_path}")
            
            # Save training history
            def convert_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_types(item) for item in obj]
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                else:
                    return obj
            
            history_data = {
                'training_history': convert_types(self.history),
                'test_results': convert_types(self.test_history)
            }
            
            history_path = self.logs_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(history_data, f, indent=4)
            print(f"Training history saved: {history_path}")
            
            return True
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            traceback.print_exc()
            return False
    
    def generate_visualizations(self, test_metrics_calc=None, cm=None, roc_data=None) -> bool:
        try:
            print(f"\n{'='*80}")
            print("Generating Visualizations")
            print(f"{'='*80}\n")
            
            self.visualizer.plot_all_metrics(self.history)
            self.visualizer.plot_metrics_comparison(self.history)
            
            if cm is not None:
                class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
                self.visualizer.plot_confusion_matrix(cm, class_names)
            
            if roc_data is not None:
                fpr_dict = {}
                tpr_dict = {}
                roc_auc_dict = {}
                
                for i, (class_name, roc_vals) in enumerate(roc_data.items()):
                    fpr_dict[i] = roc_vals['fpr']
                    tpr_dict[i] = roc_vals['tpr']
                    from sklearn.metrics import auc
                    roc_auc_dict[i] = auc(roc_vals['fpr'], roc_vals['tpr'])
                
                self.visualizer.plot_roc_curve(fpr_dict, tpr_dict, roc_auc_dict)
            
            metrics_data = {
                'training_history': self.history,
                'test_results': self.test_history
            }
            self.visualizer.save_metrics_json(metrics_data, 'metrics.json')
            
            print("Visualizations generated successfully!")
            return True
            
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            traceback.print_exc()
            return False
    
    def _save_checkpoint(self, epoch: int):
        try:
            checkpoint_path = self.model_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, str(checkpoint_path))
            print(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
    
    def run_full_pipeline(self) -> bool:
        try:
            if not self.load_data_with_retry():
                return False
            if not self.build_model():
                return False
            if not self.train():
                return False
            
            result = self.evaluate_test_set()
            if result[0] is False:
                return False
            else:
                _, test_metrics_calc, cm, roc_data = result
            
            if not self.save_model():
                return False
            if not self.generate_visualizations(test_metrics_calc, cm, roc_data):
                return False
            
            print(f"\n{'='*80}")
            print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"{'='*80}\n")
            print(f"Results saved in: {self.output_dir}")
            print(f"Model saved in: {self.model_dir}")
            print(f"Plots saved in: {self.plots_dir}")
            print(f"Logs saved in: {self.logs_dir}")
            
            return True
            
        except Exception as e:
            print(f"Error in training pipeline: {str(e)}")
            traceback.print_exc()
            return False


def main():
    print(f"\n{'#'*80}")
    print("LAMCA-Net Training Pipeline for APTOS2019 Blindness Detection (PyTorch)")
    print(f"{'#'*80}\n")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    trainer = LAMCANetTrainer(
        train_csv="APTOS2019/train.csv",
        val_csv="APTOS2019/valid.csv",
        test_csv="APTOS2019/test.csv",
        img_dir="APTOS2019/train_images",
        output_dir="results",
        model_dir="trained_model",
        batch_size=16,
        epochs=100,
        learning_rate=0.001,
        weight_decay=0.0001,
        num_workers=4,
        max_retries=3
    )
    success = trainer.run_full_pipeline()
    if success:
        print("\n✓ Training completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
