# ✅ Pre-Training Checklist & Quick Reference

**Print this page or check off items before running training!**

---

## 🔴 CRITICAL: Must Have Before Training

### Dataset Files
- [ ] `APTOS2019/train.csv` exists and contains 3,662 rows
- [ ] `APTOS2019/test.csv` exists and contains 1,499 rows
- [ ] `APTOS2019/train_images/` folder exists with ~3,662 PNG files
- [ ] `APTOS2019/test_images/` folder exists with ~1,499 PNG files
- [ ] `APTOS2019/val_images/` folder exists with ~523 PNG files

**Verify with**:
```powershell
Test-Path "APTOS2019/train.csv"
(Get-ChildItem "APTOS2019/train_images/").Count  # Should be ~3662
```

### Python Environment
- [ ] Python version 3.10 or higher
- [ ] TensorFlow 2.15.0 installed
- [ ] NumPy, Pandas, Matplotlib, Scikit-learn installed
- [ ] All packages from requirements.txt installed

**Verify with**:
```powershell
python --version  # Should show 3.10+
pip list | findstr tensorflow  # Should show 2.15.0
```

### Code Files
- [ ] train.py
- [ ] train_cvitsnet.ipynb
- [ ] cvitsnet_model.py
- [ ] dataset_loader.py
- [ ] metrics.py
- [ ] visualize.py
- [ ] verify_model_fixed.py
- [ ] requirements.txt

**Verify with**:
```powershell
Get-Item train.py, cvitsnet_model.py, dataset_loader.py, metrics.py, visualize.py, verify_model_fixed.py
```

---

## 🟡 RECOMMENDED: Should Check

### System Resources
- [ ] At least 8 GB RAM available
- [ ] GPU available (optional but recommended)
- [ ] 50 GB free disk space (for model + outputs)
- [ ] Stable internet (for dependencies if needed)

**Check GPU**:
```powershell
nvidia-smi  # If available, should show GPU memory
python -c "import tensorflow as tf; print(f'GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"
```

### Dependencies Verified
- [ ] TensorFlow imports without errors
- [ ] All required packages available
- [ ] No conflicting versions

**Verify with**:
```python
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
print("All imports successful!")
```

---

## 🟢 OPTIONAL: Nice to Have

- [ ] GPU with CUDA support (training 3x faster)
- [ ] SSD storage (faster loading than HDD)
- [ ] Display monitor (for viewing plots while training)
- [ ] Second monitor (to watch training while doing other work)

---

## 🚀 Quick Start Commands

### Verify Model Works (30 seconds)
```powershell
python verify_model_fixed.py
```
**Expected output**: ✅ All 7 tests pass

### Run Full Training (1.5-3.5 hours)

**Option A - Python Script**:
```powershell
python train.py
```

**Option B - Jupyter Notebook**:
```powershell
jupyter notebook train_cvitsnet.ipynb
```

---

## 📊 What Gets Created

After training completes, you'll have:

### Model Files (~116 MB each)
```
trained_model/
├── cvitsnet_aptos2019.h5           ← Use this for inference
├── checkpoint_epoch_010.h5
├── checkpoint_epoch_020.h5
├── ... (one every 10 epochs)
└── checkpoint_epoch_100.h5
```

### Visualization Files (~1-5 MB each)
```
results/plots/
├── loss_vs_epoch.png               ← Training curves
├── accuracy_vs_epoch.png
├── precision_vs_epoch.png
├── recall_vs_epoch.png
├── f1_score_vs_epoch.png
├── roc_curve.png                   ← Per-class analysis
├── confusion_matrix.png            ← Error analysis
├── all_metrics.png                 ← Combined view
└── metrics.json                    ← Raw values
```

### Training History (~1-2 MB)
```
results/logs/
└── training_history.json           ← Complete epoch-by-epoch history
```

---

## ⏱ Timeline Expectations

| Phase | Time | What's Happening |
|-------|------|-----------------|
| Setup | 1 min | Loading data |
| Model Build | 1 min | Compiling model |
| **Training** | 100-200 min | Epoch 1-100 |
| Evaluation | 5 min | Test set metrics |
| Visualization | 5 min | Creating plots |
| Save | 1 min | Saving model |
| **Total** | **110-220 min** | **1.5-3.5 hours** |

---

## 📈 Expected Results

### Metrics Range
```
Accuracy:    72-82%
Precision:   70-80%
Recall:      72-82%
F1 Score:    71-81%
Specificity: 85-95%
ROC-AUC:     80-90%
```

### Training Curve Pattern
```
Epoch 1:   Loss ~ 1.5-1.8, Accuracy ~ 40-50%
Epoch 10:  Loss ~ 1.0-1.2, Accuracy ~ 60-70%
Epoch 50:  Loss ~ 0.5-0.8, Accuracy ~ 75-85%
Epoch 100: Loss ~ 0.3-0.7, Accuracy ~ 78-88%
```

### Loss Should Decrease
```
✅ Epoch 1  > Epoch 10  > Epoch 50  > Epoch 100
✅ First 10 epochs: steep decrease
✅ After epoch 50: gradual improvement
✅ No sudden spikes (indicates good learning)
```

---

## 🚨 Troubleshooting Quick Fixes

| Error | Quick Fix | Full Guide |
|-------|-----------|-----------|
| `ModuleNotFoundError: tensorflow` | `pip install -r requirements.txt` | See TRAINING_EXECUTION_GUIDE.md |
| `CUDA out of memory` | Reduce batch_size to 16 or 8 | See TRAINING_EXECUTION_GUIDE.md |
| `Dataset not found` | Verify APTOS2019/ folder structure | See TRAINING_EXECUTION_GUIDE.md |
| `FileNotFoundError: train.csv` | Check file exists and has data | See TRAINING_EXECUTION_GUIDE.md |
| Training is slow (2+ min/epoch) | Check nvidia-smi for GPU use | See TRAINING_EXECUTION_GUIDE.md |

---

## 📖 Quick Reference Links

- **START_HERE.md** - Getting started guide
- **TRAINING_EXECUTION_GUIDE.md** - Detailed step-by-step instructions
- **CVITSNET_PAPER_IMPLEMENTATION.md** - Architecture details
- **PAPER_IMPLEMENTATION_NOTES.md** - Implementation changes
- **MASTER_INDEX.md** - Complete project index

---

## 🎨 Model Architecture (Quick View)

```
Input 224×224×3 (raw RGB)
    ↓
Patch Embedding (196 patches, 768-dim)
    ↓
MSF Block (multi-scale features)
    ↓
4× Encoder Blocks
├─ DGL Block (Depthwise Sep Conv + 4-Head Attention)
├─ MLP Block (Dense 128→64→768)
└─ Skip Connections (ME1→ME3, ME2→ME4)
    ↓
Classification Head (5-class softmax)
    ↓
Output: 5 DR class probabilities

Model Size: 30.3M parameters (~116 MB)
```

---

## 🔧 Configuration (If You Need to Change It)

### Edit training config in `train.py`:
```python
# Line ~150
batch_size = 32      # Reduce to 16 or 8 if OOM
epochs = 100         # Can reduce to 50 for testing
learning_rate = 0.001
weight_decay = 0.0001
```

### GPU control:
```python
# To force CPU:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# To limit GPU memory:
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

---

## ✅ Pre-Training Verification Checklist

Print this section and check off before running training:

**Dataset Ready**
- [ ] APTOS2019/train.csv exists
- [ ] APTOS2019/train_images/ has files
- [ ] APTOS2019/test_images/ has files
- [ ] APTOS2019/val_images/ has files

**Python Ready**
- [ ] Python 3.10+ installed
- [ ] TensorFlow 2.15+ installed
- [ ] All packages installed: `pip install -r requirements.txt`

**Code Ready**
- [ ] All 8 code files present
- [ ] verify_model_fixed.py runs successfully
- [ ] No import errors when running verify

**System Ready**
- [ ] At least 8 GB RAM available
- [ ] 50 GB free disk space
- [ ] GPU available (optional but nice)

**Everything Checked?**
- [ ] YES - Ready to run `python train.py`!

---

## 🎯 One-Minute Startup

```powershell
# 1. Navigate to project
cd d:\Ece_DR

# 2. Quick verify (30 sec)
python verify_model_fixed.py

# 3. Start training (1.5-3.5 hours)
python train.py

# That's it! Training will:
# ✓ Load dataset
# ✓ Build model
# ✓ Train for 100 epochs
# ✓ Evaluate on test set
# ✓ Save model to trained_model/
# ✓ Generate visualizations to results/
# ✓ Export metrics to JSON
```

---

## 📞 Quick Support

**Problem**: Training crashes  
**Solution**: Run `python verify_model_fixed.py` first to diagnose

**Problem**: Very slow training  
**Solution**: Check `nvidia-smi` for GPU usage

**Problem**: Out of memory  
**Solution**: Edit train.py, change `batch_size = 16`

**Problem**: Can't find dataset  
**Solution**: Verify APTOS2019/ folder structure

**Problem**: Import errors  
**Solution**: Run `pip install -r requirements.txt`

---

## 🎉 Success Indicators

Training is working when you see:

✅ Output starts with "Loading APTOS2019 dataset..."  
✅ Epoch 1-5 loss values decrease from ~1.5 to ~1.0  
✅ Accuracy increases from ~40% to ~60% by epoch 10  
✅ No errors in console output  
✅ GPU memory usage shows in nvidia-smi (if GPU available)  

---

## 🏁 Final Checklist

Before clicking "Run Training":

- [ ] All datasets verified ✓
- [ ] Python environment ready ✓
- [ ] All code files present ✓
- [ ] verify_model_fixed.py passes ✓
- [ ] System has enough resources ✓
- [ ] Read TRAINING_EXECUTION_GUIDE.md ✓

**✅ Ready to go!**

```powershell
python train.py
```

Expected: Complete training in 1.5-3.5 hours  
Result: Model saved, visualizations created, metrics logged  

---

**Date**: March 10, 2026  
**Model**: CViTS-Net (30.3M params)  
**Dataset**: APTOS2019 (5-class DR)  
**Status**: ✅ Ready for Training
