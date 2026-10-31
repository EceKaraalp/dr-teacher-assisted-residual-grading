"""
Training Visualizer for LAMCA-Net
"""
import matplotlib.pyplot as plt

class TrainingVisualizer:
    def __init__(self, save_dir=None):
        self.save_dir = save_dir

    def plot_all_metrics(self, history):
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

    def plot_confusion_matrix(self, cm, class_names=None):
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.colorbar()
        if class_names is not None:
            plt.xticks(range(len(class_names)), class_names)
            plt.yticks(range(len(class_names)), class_names)
        plt.show()
