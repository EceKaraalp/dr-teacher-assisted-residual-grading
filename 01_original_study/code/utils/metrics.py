"""
Metrics for DR classification
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    return acc, f1, qwk

def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    if labels is not None:
        plt.xticks(np.arange(len(labels)), labels)
        plt.yticks(np.arange(len(labels)), labels)
    plt.show()

def plot_metrics(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1, 3, 2)
    plt.plot(history['val_acc'], label='Val Acc')
    plt.plot(history['val_f1'], label='Val F1')
    plt.legend()
    plt.title('Accuracy / F1')
    plt.subplot(1, 3, 3)
    plt.plot(history['val_qwk'], label='Val QWK')
    plt.legend()
    plt.title('QWK')
    plt.tight_layout()
    plt.show()
