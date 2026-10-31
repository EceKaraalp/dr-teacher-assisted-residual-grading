"""
Metrics calculation module for CViTS-Net training.
Calculates: Accuracy, Precision, Recall, F1 Score, Specificity, ROC-AUC, QWK
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, cohen_kappa_score
)


class MetricsCalculator:
    """Calculate and track various metrics during training."""
    
    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.y_true = []
        self.y_pred = []
        self.y_pred_proba = []
    
    def update(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """
        Update metrics with batch results.
        
        Args:
            y_true: True labels (one-hot encoded or class indices)
            y_pred_proba: Predicted probabilities
        """
        # Convert one-hot to class indices if needed
        if len(y_true.shape) > 1 and y_true.shape[1] == self.num_classes:
            y_true = np.argmax(y_true, axis=1)
        
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        self.y_true.extend(y_true.flatten())
        self.y_pred.extend(y_pred.flatten())
        self.y_pred_proba.extend(y_pred_proba)
    
    def compute_metrics(self) -> dict:
        """
        Compute all metrics.
        
        Returns:
            dict: Dictionary with all metric values
        """
        if len(self.y_true) == 0:
            return {}
        
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        y_pred_proba = np.array(self.y_pred_proba)
        
        # Accuracy
        acc = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 (weighted for multi-class)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Specificity (average of per-class specificity)
        specificity = self._compute_specificity(y_true, y_pred)
        
        # ROC-AUC (one-vs-rest for multi-class)
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            roc_auc = 0.0
        
        # Quadratic Weighted Kappa (QWK) - primary metric for ordinal DR grading
        try:
            qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        except:
            qwk = 0.0
        
        metrics = {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'roc_auc': float(roc_auc),
            'qwk': float(qwk)
        }
        
        return metrics
    
    def _compute_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute weighted average specificity for multi-class classification.
        
        Specificity = TN / (TN + FP)
        """
        specificities = []
        
        for class_idx in range(self.num_classes):
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            
            if (tn + fp) > 0:
                spec = tn / (tn + fp)
            else:
                spec = 0.0
            
            specificities.append(spec)
        
        # Weighted average
        class_counts = np.bincount(y_true, minlength=self.num_classes)
        weights = class_counts / np.sum(class_counts)
        weighted_spec = np.mean(specificities)
        
        return weighted_spec
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.
        
        Returns:
            np.ndarray: Confusion matrix
        """
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        return confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
    
    def get_roc_curves(self) -> dict:
        """
        Get ROC curve data for each class (one-vs-rest).
        
        Returns:
            dict: Dictionary with FPR and TPR for each class
        """
        y_true = np.array(self.y_true)
        y_pred_proba = np.array(self.y_pred_proba)
        
        roc_data = {}
        
        for class_idx in range(self.num_classes):
            y_true_binary = (y_true == class_idx).astype(int)
            
            try:
                fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, class_idx])
                roc_data[f'class_{class_idx}'] = {'fpr': fpr, 'tpr': tpr}
            except:
                pass
        
        return roc_data
    
    def get_class_metrics(self) -> dict:
        """
        Get per-class metrics (precision, recall, F1).
        
        Returns:
            dict: Per-class metrics
        """
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        
        class_metrics = {}
        
        for class_idx in range(self.num_classes):
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            class_metrics[f'class_{class_idx}'] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
        
        return class_metrics


def compute_metrics(y_true, y_pred):
    """
    Standalone function to compute key metrics.
    
    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
    
    Returns:
        tuple: (accuracy, f1_score, qwk_score)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # F1 Score (weighted for multi-class)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Quadratic Weighted Kappa (QWK)
    try:
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    except:
        qwk = 0.0
    
    return float(acc), float(f1), float(qwk)



