"""
Training pipeline for 10 hybrid CNN+ViT research ideas.

Handles:
- Data loading (80/10/10 split)
- Model training with checkpointing
- Metrics computation (Accuracy, F1, QWK, Confusion Matrix)
- Visualization (Loss curves, metrics plots)
- Results aggregation
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from datetime import datetime

# Local imports
from models.hybrid_cnn_vit_base import HybridCNNViTBase
from model_configs import get_model_config, list_all_models
from dataset_loader import APTOS2019DatasetLoader, get_data_loaders
from metrics import compute_metrics
from sklearn.metrics import confusion_matrix, classification_report


class HybridModelTrainer:
    """Unified trainer for all 10 model variants."""
    
    def __init__(
        self,
        model_name: str,
        config: Dict,
        device: str = "cuda",
        num_epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        results_dir: str = "results/hybrid_cnn_vit"
    ):
        self.model_name = model_name
        self.config = config
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Model-specific directories
        self.model_dir = self.results_dir / model_name
        self.model_dir.mkdir(exist_ok=True)
        
        self.checkpoint_path = self.model_dir / "checkpoint_current.pth"
        self.best_model_path = self.model_dir / "best_model.pth"
        self.metrics_path = self.model_dir / "metrics.json"
        self.history_path = self.model_dir / "training_history.json"
        
        print(f"\n{'='*60}")
        print(f"Training Model: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"Output Dir: {self.model_dir}")
        print(f"{'='*60}\n")
        
        # Initialize model, optimizer, scheduler
        self.model = HybridCNNViTBase(num_classes=5, config=config).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        
        # Loss criterion
        self.criterion = nn.CrossEntropyLoss()
        
        # Load existing training state if available
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_qwk": []}
        self.best_qwk = -np.inf
        self.start_epoch = 0
        
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load training state from checkpoint if it exists."""
        if self.checkpoint_path.exists():
            print(f"Loading checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.history = checkpoint["history"]
            self.best_qwk = checkpoint["best_qwk"]
            self.start_epoch = checkpoint["epoch"] + 1
            
            print(f"Resumed from epoch {self.start_epoch} (Best QWK: {self.best_qwk:.4f})")
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint for resuming."""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "history": self.history,
            "best_qwk": self.best_qwk,
        }
        torch.save(checkpoint, self.checkpoint_path)
    
    def _save_best_model(self):
        """Save best model weights."""
        torch.save(self.model.state_dict(), self.best_model_path)
        print(f"✓ Best model saved (QWK: {self.best_qwk:.4f})")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Handle one-hot labels
            if labels.ndim > 1:
                label_indices = labels.argmax(1)
            else:
                label_indices = labels
            
            loss = self.criterion(outputs, label_indices)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            pbar.set_postfix({"loss": loss.item():.4f})
        
        return total_loss / len(train_loader.dataset)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validation", leave=False)
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                
                # Handle one-hot labels
                if labels.ndim > 1:
                    label_indices = labels.argmax(1)
                else:
                    label_indices = labels
                
                loss = self.criterion(outputs, label_indices)
                total_loss += loss.item() * images.size(0)
                
                preds = outputs.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(label_indices.cpu().numpy())
        
        val_loss = total_loss / len(val_loader.dataset)
        accuracy, f1, qwk = compute_metrics(all_labels, all_preds)
        
        return val_loss, accuracy, f1, qwk
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop with early stopping."""
        patience = 15
        patience_counter = 0
        
        for epoch in range(self.start_epoch, self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            
            # Validate
            val_loss, val_acc, val_f1, val_qwk = self.validate(val_loader)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["val_qwk"].append(val_qwk)
            
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Accuracy: {val_acc:.4f} | F1: {val_f1:.4f} | QWK: {val_qwk:.4f}")
            
            # Checkpoint
            self._save_checkpoint(epoch)
            
            # Early stopping based on QWK
            if val_qwk > self.best_qwk:
                self.best_qwk = val_qwk
                self._save_best_model()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping: No improvement for {patience} epochs")
                    break
            
            self.scheduler.step()
        
        # Save final history
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n✓ Training completed. Best QWK: {self.best_qwk:.4f}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate on test set."""
        print(f"\nEvaluating on test set...")
        
        # Load best model
        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                
                # Handle one-hot labels
                if labels.ndim > 1:
                    label_indices = labels.argmax(1)
                else:
                    label_indices = labels
                
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(1)
                
                all_probs.append(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label_indices.cpu().numpy())
        
        all_probs = np.vstack(all_probs)
        accuracy, f1, qwk = compute_metrics(all_labels, all_preds)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Classification report
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        
        results = {
            "accuracy": float(accuracy),
            "f1_macro": float(f1),
            "qwk": float(qwk),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs,
        }
        
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test F1: {f1:.4f}")
        print(f"  Test QWK: {qwk:.4f}")
        print(f"\nClassification Report:\n{classification_report(all_labels, all_preds, target_names=class_names)}")
        
        # Save results
        with open(self.metrics_path, 'w') as f:
            json.dump({
                "accuracy": float(accuracy),
                "f1_macro": float(f1),
                "qwk": float(qwk),
                "classification_report": report,
            }, f, indent=2)
        
        return results
    
    def plot_metrics(self):
        """Plot training history and test confusion matrix."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Training Metrics: {self.model_name}", fontsize=14, fontweight='bold')
        
        # Loss curve
        axes[0, 0].plot(self.history["train_loss"], label="Train Loss")
        axes[0, 0].plot(self.history["val_loss"], label="Val Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training & Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history["val_acc"], label="Val Accuracy", color='green')
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_title("Validation Accuracy")
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 0].plot(self.history["val_f1"], label="Val F1", color='orange')
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("F1 Score")
        axes[1, 0].set_title("Validation F1 Score")
        axes[1, 0].grid(True, alpha=0.3)
        
        # QWK
        axes[1, 1].plot(self.history["val_qwk"], label="Val QWK", color='red')
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("QWK")
        axes[1, 1].set_title("Validation QWK (Primary Metric)")
        axes[1, 1].axhline(y=self.best_qwk, color='darkred', linestyle='--', label=f"Best: {self.best_qwk:.4f}")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / "training_metrics.png", dpi=150, bbox_inches='tight')
        print(f"✓ Metrics plot saved: {self.model_dir / 'training_metrics.png'}")
        plt.close()
    
    def plot_confusion_matrix(self, cm: np.ndarray):
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f"Confusion Matrix: {self.model_name}")
        
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        # Add text annotations
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(self.model_dir / "confusion_matrix.png", dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {self.model_dir / 'confusion_matrix.png'}")
        plt.close()


def run_all_training():
    """Execute training for all 10 model variants."""
    
    print("\n" + "="*60)
    print("HYBRID CNN+ViT: 10 RESEARCH IDEAS TRAINING PIPELINE")
    print("="*60)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load data (once)
    print("\nLoading APTOS2019 dataset...")
    train_loader, val_loader, test_loader, class_weights, (X_train, y_train, X_val, y_val, X_test, y_test) = \
        get_data_loaders(dataset_path="APTOS2019", batch_size=16)
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Results aggregation
    all_results = {}
    
    # Train each model
    for idx, model_name in enumerate(list_all_models(), 1):
        print(f"\n\n{'#'*60}")
        print(f"MODEL {idx}/10: {model_name}")
        print(f"{'#'*60}")
        
        config = get_model_config(model_name)
        
        try:
            trainer = HybridModelTrainer(
                model_name=model_name,
                config=config,
                device=device,
                num_epochs=100,
                batch_size=16,
                learning_rate=1e-3,
                results_dir="results/hybrid_cnn_vit"
            )
            
            # Train
            trainer.train(train_loader, val_loader)
            
            # Evaluate on test set
            test_results = trainer.evaluate(test_loader)
            
            # Plot metrics
            trainer.plot_metrics()
            trainer.plot_confusion_matrix(np.array(test_results["confusion_matrix"]))
            
            # Store results
            all_results[model_name] = {
                "accuracy": test_results["accuracy"],
                "f1_macro": test_results["f1_macro"],
                "qwk": test_results["qwk"],
                "best_epoch_qwk": float(trainer.best_qwk),
            }
            
        except Exception as e:
            print(f"✗ Error training {model_name}: {str(e)}")
            all_results[model_name] = {"error": str(e)}
    
    # Summary results
    print("\n\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    results_df = pd.DataFrame(all_results).T
    results_df = results_df.sort_values("qwk", ascending=False)
    
    print("\n" + str(results_df))
    
    results_df.to_csv("results/hybrid_cnn_vit/all_models_comparison.csv")
    print("\n✓ Results saved to: results/hybrid_cnn_vit/all_models_comparison.csv")
    
    return all_results


if __name__ == "__main__":
    run_all_training()
