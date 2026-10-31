# 🚀 CViTS-Net Training: Complete Execution Guide

## Pre-Training Checklist ✅

Before running training, verify all prerequisites:

### 1. Dataset Structure
```
✅ VERIFY: d:\Ece_DR\APTOS2019\
    ├── train.csv (contains: id_code, diagnosis)
    ├── test.csv
    ├── train_images/ (containing .png files)
    ├── test_images/ (containing .png files)
    └── val_images/ (containing .png files)
```

**Check**: Run in PowerShell:
```powershell
Get-Item "d:\Ece_DR\APTOS2019\train.csv" -Force
Get-ChildItem "d:\Ece_DR\APTOS2019\train_images\" | Measure-Object | Select-Object Count
```

### 2. Python Environment
```
✅ REQUIRED:
    - Python 3.10+
    - TensorFlow 2.15.0
    - NumPy, Pandas, Matplotlib, Scikit-learn
```

**Check**: Run in PowerShell:
```powershell
python --version
pip list | grep -E "tensorflow|numpy|pandas"
```

### 3. All Code Files Present
```
✅ VERIFY:
    ├── cvitsnet_model.py
    ├── dataset_loader.py
    ├── metrics.py
    ├── visualize.py
    ├── train.py
    ├── verify_model_fixed.py
    └── train_cvitsnet.ipynb
```

**Check**: All files exist with sizes:
- `cvitsnet_model.py`: ~12-15 KB
- `train.py`: ~20-25 KB
- `train_cvitsnet.ipynb`: ~30-40 KB

### 4. GPU Availability (Recommended)
```
✅ CHECK GPU:
    - NVIDIA GPU (if available)
    - CUDA support (if using GPU)
```

**Check**: Run in PowerShell:
```powershell
python -c "import tensorflow as tf; print(f'GPUs Available: {len(tf.config.list_physical_devices(\"GPU\"))}')"
```

---

## Execution Paths

### Path 1: Python Script (Recommended for Batch)

**Step 1**: Navigate to project directory
```powershell
cd d:\Ece_DR
```

**Step 2**: Verify environment is activated
```powershell
python --version  # Should show 3.10+
```

**Step 3**: Run training
```powershell
python train.py
```

**Expected Output**:
```
2026-03-10 14:23:45 - INFO - Loading APTOS2019 dataset...
2026-03-10 14:23:47 - INFO - Dataset loaded: Train=3662, Val=523, Test=1499
2026-03-10 14:23:50 - INFO - Building CViTS-Net model...
2026-03-10 14:23:52 - INFO - Model Summary:
  Total params: 30,338,053
  Trainable params: 30,338,053
2026-03-10 14:23:52 - INFO - Training for 100 epochs...
Epoch 1/100
115/115 [==============================] - 65s 563ms/step - loss: 1.5632 - accuracy: 0.4521 - val_loss: 1.2841 - val_accuracy: 0.5823
Epoch 2/100
115/115 [==============================] - 52s 452ms/step - loss: 1.1234 - accuracy: 0.6234 - val_loss: 1.0123 - val_accuracy: 0.6891
...
Epoch 100/100
115/115 [==============================] - 51s 441ms/step - loss: 0.4532 - accuracy: 0.8234 - val_loss: 0.6234 - val_accuracy: 0.7734
2026-03-10 17:35:12 - INFO - Training completed!
2026-03-10 17:35:12 - INFO - Evaluating on test set...
Test Accuracy: 0.7512
Test F1-Score: 0.7234
...
2026-03-10 17:35:45 - INFO - Model saved to: trained_model/cvitsnet_aptos2019.h5
2026-03-10 17:35:45 - INFO - Visualizations saved to: results/plots/
```

**Duration**: 100-200 minutes (1.5-3.5 hours)

**Step 4**: Verify completion
```powershell
Test-Path "trained_model/cvitsnet_aptos2019.h5"
Get-ChildItem "results/plots/" | Measure-Object | Select-Object Count
```

---

### Path 2: Jupyter Notebook (Interactive)

**Step 1**: Launch Jupyter
```powershell
cd d:\Ece_DR
jupyter notebook train_cvitsnet.ipynb
```

Browser will open to: `http://localhost:8888/notebooks/train_cvitsnet.ipynb`

**Step 2**: Execute cells in order

Cell structure:
```
1. [Markdown] CViTS-Net Training Notebook
2. [Markdown] Setup Instructions
3. [Code] Imports
4. [Markdown] Dataset Loading
5. [Code] Load Data
6. [Markdown] Model Building
7. [Code] Build Model
8. [Markdown] Create Output Directories
9. [Code] Create Directories
10. [Markdown] Training Loop
11. [Code] Training (⏱ This takes 1.5-3.5 hours)
12. [Markdown] Evaluate Test Set
13. [Code] Test Evaluation
14. [Markdown] Save Model
15. [Code] Save Model
16. [Markdown] Generate Visualizations
17. [Code] Generate Plots
18. [Markdown] Summary
19. [Code] Summary Stats
...
```

**Execute Workflow**:
- Click "Run All" button (⏯ in toolbar)
- OR click each cell → press Shift+Enter
- Monitor progress in output
- Stop kernel if needed (Kernel → Interrupt)

**Step 3**: Monitor progress
- Watch loss curve in cell outputs
- Check accuracy metrics per epoch
- Verify GPU usage (if applicable)

**Step 4**: Review results
- Scroll to visualization cells
- View plots inline
- Download model from output

---

## Real-Time Monitoring

### During Training

**Monitor Loss**:
```
Epoch 1: loss ~1.5-1.8 (starting high)
Epoch 10: loss ~1.0-1.2 (improving)
Epoch 50: loss ~0.5-0.8 (converging)
Epoch 100: loss ~0.3-0.7 (final)
```

**Monitor Accuracy**:
```
Epoch 1: accuracy ~40-50% (random)
Epoch 10: accuracy ~60-70% (learning)
Epoch 50: accuracy ~75-85% (converging)
Epoch 100: accuracy ~78-88% (final)
```

**GPU Memory**:
- Typical usage: 4-6 GB during training
- If OOM error: Reduce batch_size to 16 or 8

### After Each Epoch
- Loss ratio: val_loss / train_loss should be ~1-1.5
- If > 2: Model may be overfitting
- If < 0.8: Model may be underfitting

---

## Output Files

### Automatically Created After Training

```
trained_model/
├── cvitsnet_aptos2019.h5          ← Final model (116 MB)
├── checkpoint_epoch_010.h5        ← Checkpoint at epoch 10
├── checkpoint_epoch_020.h5        ← Checkpoint at epoch 20
├── checkpoint_epoch_030.h5        ← Checkpoint at epoch 30
├── checkpoint_epoch_040.h5        ← Checkpoint at epoch 40
├── checkpoint_epoch_050.h5        ← Checkpoint at epoch 50
├── checkpoint_epoch_060.h5        ← Checkpoint at epoch 60
├── checkpoint_epoch_070.h5        ← Checkpoint at epoch 70
├── checkpoint_epoch_080.h5        ← Checkpoint at epoch 80
├── checkpoint_epoch_090.h5        ← Checkpoint at epoch 90
└── checkpoint_epoch_100.h5        ← Checkpoint at epoch 100

results/
├── plots/
│   ├── loss_vs_epoch.png           (300 DPI - publication quality)
│   ├── accuracy_vs_epoch.png
│   ├── precision_vs_epoch.png
│   ├── recall_vs_epoch.png
│   ├── f1_score_vs_epoch.png
│   ├── roc_curve.png               (5 DR classes)
│   ├── confusion_matrix.png        (5×5 classes)
│   ├── all_metrics.png             (combined grid)
│   └── metrics.json                (raw metric values)
└── logs/
    └── training_history.json       (complete epoch-by-epoch history)
```

### File Descriptions

**cvitsnet_aptos2019.h5** (Final Model)
- Size: ~116 MB
- Format: Keras SavedModel format
- Contains: Weights + architecture + optimizer state
- Usage: Load for inference on new images
- Load code:
  ```python
  model = tf.keras.models.load_model('trained_model/cvitsnet_aptos2019.h5')
  ```

**Checkpoint Files** (Resume Training)
- Save model state every 10 epochs
- Use to resume if interrupted:
  ```python
  model = tf.keras.models.load_model('trained_model/checkpoint_epoch_050.h5')
  # Continue training from epoch 50
  ```

**Visualizations** (Analysis)
- PNG files: View in any image viewer
- All curves: Training (blue) vs Validation (orange)
- ROC Curve: One line per DR class
- Confusion Matrix: Shows classification errors
- Combined Grid: All metrics together

**metrics.json** (Raw Values)
```json
{
  "training_history": {
    "loss": {"train": [...], "val": [...]},
    "accuracy": {"train": [...], "val": [...]},
    ...
  },
  "test_results": {
    "metrics": {
      "accuracy": 0.7512,
      "precision": 0.7623,
      "recall": 0.7512,
      "f1_score": 0.7234,
      "specificity": 0.8921,
      "roc_auc": 0.8456
    },
    "per_class": {...}
  }
}
```

---

## Verification After Training

### Step 1: Check Model Size
```powershell
$model = Get-Item "trained_model/cvitsnet_aptos2019.h5"
$model.Length / 1MB  # Should show ~116 MB
```

### Step 2: Load and Verify Model
```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('trained_model/cvitsnet_aptos2019.h5')

# Check shape
print(model.input_shape)      # Should be (None, 224, 224, 3)
print(model.output_shape)     # Should be (None, 5)

# Count parameters
print(model.count_params())   # Should be 30,338,053
```

### Step 3: Verify Visualizations Exist
```powershell
Get-ChildItem "results/plots/" -Filter "*.png" | ForEach-Object { Write-Host $_.Name }
```

Should list:
- loss_vs_epoch.png
- accuracy_vs_epoch.png
- precision_vs_epoch.png
- recall_vs_epoch.png
- f1_score_vs_epoch.png
- roc_curve.png
- confusion_matrix.png
- all_metrics.png

### Step 4: Check Training History
```powershell
Get-Content "results/logs/training_history.json" | ConvertFrom-Json | Get-Member
```

---

## Model Inference (After Training)

### Quick Prediction
```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model('trained_model/cvitsnet_aptos2019.h5')

# Load image
img = Image.open('APTOS2019/test_images/test_image_001.png')
img = img.resize((224, 224))
img_array = np.array(img)  # Shape: (224, 224, 3), dtype: uint8

# Add batch dimension
img_batch = img_array[np.newaxis, ...]  # Shape: (1, 224, 224, 3)

# Predict
predictions = model.predict(img_batch)
class_idx = np.argmax(predictions[0])
confidence = predictions[0][class_idx]

# DR labels
dr_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
print(f"Diagnosis: {dr_labels[class_idx]}")
print(f"Confidence: {confidence:.2%}")
print(f"\nAll class probabilities:")
for i, label in enumerate(dr_labels):
    print(f"  {label}: {predictions[0][i]:.4f}")
```

### Batch Prediction
```python
import tensorflow as tf
import numpy as np
from PIL import Image
import os

model = tf.keras.models.load_model('trained_model/cvitsnet_aptos2019.h5')
dr_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

# Load multiple images
image_dir = 'APTOS2019/test_images/'
images = []

for i, filename in enumerate(os.listdir(image_dir)[:10]):  # First 10 images
    img_path = os.path.join(image_dir, filename)
    img = Image.open(img_path).resize((224, 224))
    images.append(np.array(img))

# Batch prediction
images = np.array(images)  # Shape: (N, 224, 224, 3)
predictions = model.predict(images)

# Display results
for i, pred in enumerate(predictions):
    class_idx = np.argmax(pred)
    confidence = pred[class_idx]
    print(f"Image {i+1}: {dr_labels[class_idx]} ({confidence:.2%})")
```

---

## Performance Benchmarks

### Expected Metrics (After Training)
```
Accuracy:    72-82%
Precision:   70-80%
Recall:      72-82%
F1 Score:    71-81%
Specificity: 85-95%
ROC-AUC:     80-90%
```

### Per-Class Performance
```
No DR:           ~85-95% accuracy (most samples, easiest)
Mild:            ~70-80% accuracy
Moderate:        ~60-75% accuracy
Severe:          ~50-70% accuracy (hard to distinguish from proliferative)
Proliferative:   ~40-65% accuracy (fewest samples, hardest)
```

### Training Speed
```
Epoch 1:         150-180 seconds (includes compilation)
Epoch 2-10:      60-90 seconds
Epoch 11+:       50-80 seconds
Total 100 epochs: 1.5-3.5 hours (depending on GPU)
```

---

## Troubleshooting

### Problem: Training crashes with CUDA error
**Solution**:
```python
# In train.py, uncomment:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
```

### Problem: "Out of memory" error
**Solution**: Reduce batch size
```python
# In train.py:
batch_size = 16  # Instead of 32
```

### Problem: Training is very slow (2+ min per epoch)
**Solution**: 
- Check GPU is being used: `nvidia-smi`
- Reduce batch size may help
- Ensure no other processes using GPU

### Problem: Loss not decreasing (stuck after epoch 5)
**Solution**: 
- Check learning rate (default 0.001 should work)
- Dataset may be problematic
- Try reducing to batch_size=16

### Problem: Model saves but can't load
**Solution**: Ensure TensorFlow 2.15+
```bash
pip install --upgrade tensorflow==2.15.0
```

---

## Next Steps After Training

### 1. Analyze Results
- Open `results/plots/all_metrics.png` in image viewer
- Check confusion matrix to see which classes are confused
- Review ROC curves to see per-class performance

### 2. Export Model for Deployment
```python
# Convert to TensorFlow Lite (mobile)
model = tf.keras.models.load_model('trained_model/cvitsnet_aptos2019.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('cvitsnet_aptos2019.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 3. Deploy for Web
```python
# Convert to JSON format
model.export('cvitsnet_model_web')  # Exports as SavedModel for TensorFlow.js
```

### 4. Advanced Analysis
- Plot training curves per class
- Analyze misclassifications
- Study worst-performing samples

---

## Quick Reference Commands

```powershell
# Check GPU availability
python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))"

# Run training
python train.py

# Launch Jupyter
jupyter notebook train_cvitsnet.ipynb

# Verify model
python verify_model_fixed.py

# Check output files
Get-ChildItem trained_model/ | Measure-Object -Sum { $_.Length/1MB }
Get-ChildItem results/plots/ | Measure-Object
```

---

## Success Indicators ✅

After training completes successfully, you should see:

✅ Model file: `trained_model/cvitsnet_aptos2019.h5` (~116 MB)
✅ 10 checkpoint files saved
✅ 8+ visualization PNG files in `results/plots/`
✅ `training_history.json` with complete metrics
✅ Test accuracy reported (typically 70-82%)
✅ No error messages in console
✅ All plots display correctly

---

**Ready to Train!** 🎉

Choose your execution path above and follow the steps.
Expected time: 1.5-3.5 hours for full training with 100 epochs.

Good luck! 🚀
