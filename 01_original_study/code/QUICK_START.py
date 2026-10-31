"""
CViTS-Net Quick Start Guide
============================

This guide walks you through running the CViTS-Net training pipeline.

STEP 1: Verify Environment
---------------------------
python verify_model_fixed.py

Expected output:
  - Model builds successfully
  - All 7 verification tests pass
  - Ready for training

STEP 2: Run Training
-------------------
python train.py

The training pipeline will:
1. Load APTOS2019 dataset (with automatic retry)
2. Build CViTS-Net model (46M parameters)
3. Train for 100 epochs with progress updates
4. Calculate all metrics (accuracy, precision, recall, F1, specificity, ROC-AUC)
5. Generate visualizations
6. Save model and training history

Approximate time: 10-15 minutes per epoch on modern GPU

STEP 3: Examine Results
-----------------------
Check the following output directories:

trained_model/
  ├── cvitsnet_aptos2019.h5        <- Final trained model
  └── checkpoint_epoch_*.h5        <- Checkpoint files

results/
  ├── plots/
  │   ├── loss_vs_epoch.png
  │   ├── accuracy_vs_epoch.png
  │   ├── precision_vs_epoch.png
  │   ├── recall_vs_epoch.png
  │   ├── f1_score_vs_epoch.png
  │   ├── roc_curve.png
  │   ├── confusion_matrix.png
  │   ├── all_metrics.png
  │   └── metrics.json
  └── logs/
      └── training_history.json    <- Complete training metrics

STEP 4: Use Trained Model
-------------------------
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model('trained_model/cvitsnet_aptos2019.h5')

# Load and preprocess image
image = ... # Load your 224x224 RGB image (uint8, values 0-255)
if len(image.shape) == 3:
    image = image[np.newaxis, ...]  # Add batch dimension

# Get predictions
predictions = model.predict(image)
class_prob = predictions[0]
class_idx = np.argmax(class_prob)
confidence = class_prob[class_idx]

# DR levels:
dr_levels = {
    0: 'No DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferative'
}

print(f"Prediction: {dr_levels[class_idx]}")
print(f"Confidence: {confidence:.4f}")

TROUBLESHOOTING
---------------

1. "ModuleNotFoundError: No module named 'tensorflow'"
   -> Run: pip install -r requirements.txt

2. "Dataset not found"
   -> Ensure APTOS2019/ folder exists in current directory with train.csv and images

3. "CUDA out of memory"
   -> Reduce batch size: Edit train.py and change batch_size=32 to batch_size=16

4. "FileNotFoundError: train.csv"
   -> Check APTOS2019 folder structure - should contain train.csv and train_images/

5. "Import error for modules"
   -> Run: python verify_model_fixed.py to diagnose

CUSTOMIZATION
------------

To modify training parameters, edit train.py:

    trainer = CVitsNetTrainer(
        dataset_path="APTOS2019",    # Path to dataset
        output_dir="results",         # Output directory
        model_dir="trained_model",    # Model save directory
        batch_size=32,                # Batch size (reduce if OOM)
        epochs=100,                   # Number of epochs
        learning_rate=0.001,          # Learning rate
        weight_decay=0.0001,          # AdamW weight decay
        max_retries=3                 # Retry attempts for data loading
    )

KEY FEATURES
-----------
✓ Exact CViTS-Net architecture implementation
✓ No preprocessing (raw images only)
✓ All metrics calculated (accuracy, precision, recall, F1, specificity, ROC-AUC)
✓ Automatic visualizations
✓ Automatic retry on failure
✓ GPU memory optimization
✓ Full training history logging
✓ Model checkpointing

FILES
-----
dataset_loader.py         - Dataset loading with retry logic
cvitsnet_model.py         - CViTS-Net architecture
metrics.py                - Metric calculations
visualize.py              - Visualization utilities
train.py                  - Main training script
verify_model_fixed.py     - Model verification
requirements.txt          - Python dependencies
CVITSNET_README.md        - Full documentation

TRAINING TIMELINE
-----------------
Epoch 1:   2-3 minutes (model compilation + first pass)
Epoch 2-10: 1-2 minutes each
Epoch 11+: 1 minute each (stabilized)

Total for 100 epochs: ~120-150 minutes (~2-2.5 hours) on modern GPU

SUPPORT
-------
For issues, check:
1. CVITSNET_README.md for detailed documentation
2. verify_model_fixed.py to ensure model builds
3. Training logs in results/logs/training_history.json
4. Training plots in results/plots/

Ready to train? Run: python train.py
"""

# This is a documentation file. You can view it with:
# python -m pydoc THIS_FILE
# or just read it in any text editor
