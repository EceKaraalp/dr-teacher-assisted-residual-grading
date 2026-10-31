# 10 Hybrid CNN+ViT Models - Training & Execution Guide

## 📌 Overview

This pipeline trains **10 novel CNN+ViT hybrid architecture variants** on the APTOS 2019 diabetic retinopathy dataset with:
- **Checkpointing**: Resume training if interrupted
- **Comprehensive Metrics**: Accuracy, F1-score, QWK, Confusion Matrices
- **Automatic Visualization**: Loss curves, metric evolution, per-class heatmaps
- **Best Model Tracking**: Saves top performer for each variant

---

## 🗂️ Project Structure

```
d:\Ece_DR\
├── models/
│   ├── hybrid_cnn_vit_base.py      # Base architecture with all components
│   ├── cnn_branch.py               # CNN encoder
│   ├── transformer_branch.py       # ViT encoder
│   └── ...
├── model_configs.py                 # 10 model configurations
├── train_all_hybrid_models.py       # Main training script
├── analyze_hybrid_models_results.ipynb  # Results analysis & comparison
├── novel_hybrid_cnn_vit_research_ideas.ipynb  # Research idea documentation
├── dataset_loader.py                # Data loading (APTOS2019)
├── metrics.py                       # Metric computations
├── APTOS2019/                       # Dataset directory
│   ├── train.csv
│   ├── test.csv
│   ├── valid.csv
│   ├── train_images/
│   ├── test_images/
│   └── val_images/
└── results/hybrid_cnn_vit/          # Generated results
    ├── Idea_1_ConfidenceGatedFusion/
    ├── Idea_2_MultiScaleLesionAttention/
    ├── ... (10 models)
    ├── 00_ranking_comparison.png
    ├── 01_loss_curves_comparison.png
    ├── 02_validation_metrics_evolution.png
    ├── 03_best_model_confusion_matrix.png
    ├── 04_per_class_f1_heatmap.png
    ├── all_models_comparison.csv
    └── model_comparison_sorted.csv
```

---

## 🚀 Quick Start

### Option 1: Full Training (Recommended)

Run all 10 models with checkpointing:

```bash
python train_all_hybrid_models.py
```

**Expected output:**
- Trains each model for up to 100 epochs
- Early stopping: stops if no QWK improvement for 15 epochs
- Saves checkpoint every epoch (resume if interrupted)
- Generates loss/accuracy plots and confusion matrices
- Results saved in `results/hybrid_cnn_vit/`

**Execution Time:** ~4-6 hours per model (GPU), ~24-36 hours for all 10

### Option 2: Analyze Results

After training completes, analyze all models:

```bash
jupyter notebook analyze_hybrid_models_results.ipynb
```

This notebook provides:
- Performance rankings (QWK, Accuracy, F1)
- Training loss curve comparisons
- Validation metric evolution
- Per-class F1-score heatmap
- Best model confusion matrix analysis
- Statistical summary

---

## 📊 Model Configurations

### 10 Research Ideas:

| # | Idea | Description | Key Features |
|---|------|-------------|--------------|
| 1 | **Confidence-Gated Fusion** | Per-sample branch weighting | Gate mechanism + entropy regularization |
| 2 | **Lesion-Scale Attention** | Multi-scale spatial-channel attention | Pyramid co-attention |
| 3 | **Uncertainty Token Refinement** | Uncertainty-driven token pruning | Soft selection in ViT |
| 4 | **Ordinal QWK Optimization** | Hybrid loss + ordinal labels | Distance-aware penalties |
| 5 | **Dual-Stream Cross-Attention** | Prototype-guided branch alignment | Memory bank + relation modeling |
| 6 | **Topology-Aware Graph** | Retinal anatomy relations (GNN) | Lesion connectivity graph |
| 7 | **Frequency-Spatial Dual** | Wavelet + spatial tokens | Dual domain attention |
| 8 | **Mixture-of-Experts** | Severity-aware routing | 3 expert networks + load balancing |
| 9 | **Causal Counterfactual** | Lesion-focused consistency | Counterfactual augmentation + invariance |
| 10 | **Tri-Level Distillation** | Self-distillation + calibration | Feature/token/logit levels + uncertainty weighting |

---

## 🔄 Resuming Interrupted Training

If training interrupts (power loss, timeout, etc.):

1. **Checkpoint exists**: Run the same command again
   ```bash
   python train_all_hybrid_models.py
   ```
   
2. **Checkpoint auto-loading**: Each model automatically detects and loads its latest checkpoint

3. **Epoch tracking**: Resumes from where it stopped

4. **Results preservation**: All previously saved outputs retained

---

## 📈 Metrics Explained

### Primary Metric: **QWK** (Quadratic Weighted Kappa)
- Measures agreement between predicted and true DR grades
- Accounts for ordinal nature of severity
- **Range**: -1 to 1 (1 = perfect)
- **Why**: Prioritizes clinically meaningful errors

### Secondary Metrics:
- **Accuracy**: % correct predictions
- **F1-Score (Macro)**: Mean F1 across all classes (handles imbalance)
- **Per-Class Metrics**: Precision, Recall, F1 per DR grade

### Confusion Matrix:
- Shows classification patterns per class
- Reveals if model confuses Mild↔Moderate or other patterns
- Saved as `.png` for each model variant

---

## 🎯 Output Files

### Per-Model (`results/hybrid_cnn_vit/<Model_Name>/`):

| File | Description |
|------|-------------|
| `best_model.pth` | Best weights (highest QWK) |
| `checkpoint_current.pth` | Latest epoch checkpoint (for resuming) |
| `metrics.json` | Test set performance (Acc, F1, QWK, reports) |
| `training_history.json` | All epochs' train/val losses & metrics |
| `training_metrics.png` | 4-panel plot (loss, acc, F1, QWK curves) |
| `confusion_matrix.png` | Test set confusion matrix heatmap |

### Global (`results/hybrid_cnn_vit/`):

| File | Description |
|------|-------------|
| `all_models_comparison.csv` | Metrics for all 10 models |
| `model_comparison_sorted.csv` | Same, sorted by QWK descending |
| `00_ranking_comparison.png` | Bar charts for Acc/F1/QWK rankings |
| `01_loss_curves_comparison.png` | 10-panel loss curves (all models) |
| `02_validation_metrics_evolution.png` | 10-panel QWK/Acc evolution |
| `03_best_model_confusion_matrix.png` | Detailed CM for top-performing model |
| `04_per_class_f1_heatmap.png` | F1-score heatmap across classes |

---

## 💾 Checkpoint Details

Each checkpoint saves:
```
{
  "epoch": 45,                          # Last completed epoch
  "model_state": {...},                 # Model weights
  "optimizer_state": {...},             # Adam state for resuming
  "scheduler_state": {...},             # LR scheduler state
  "history": {                          # Training history up to epoch
    "train_loss": [...],
    "val_loss": [...],
    "val_acc": [...],
    "val_f1": [...],
    "val_qwk": [...]
  },
  "best_qwk": 0.7234                    # Best QWK so far
}
```

---

## ⚙️ Configuration

Edit `model_configs.py` to modify any model:

```python
"Idea_1_ConfidenceGatedFusion": {
    "fusion_method": "gate",            # or "concat"
    "use_uncertainty_refinement": False,
    "use_ordinal_head": False,
    "loss_weights": {"ce": 1.0, ...},   # Adjust loss weights
    "hyperparam": {
        "entropy_reg": 0.01,            # Tune hyperparameters
    },
}
```

---

## 🔧 Hyperparameters (in `train_all_hybrid_models.py`)

```python
HybridModelTrainer(
    num_epochs=100,           # Max training epochs
    batch_size=16,            # Batch size (smaller = more stable)
    learning_rate=1e-3,       # AdamW learning rate
    results_dir="results/hybrid_cnn_vit"
)
```

Early stopping settings:
```python
patience = 15               # Stop if no QWK improvement for 15 epochs
```

---

## 📋 Dataset Info

**APTOS 2019**: 5-class Diabetic Retinopathy classification

| Split | Images | Classes |
|-------|--------|---------|
| Train | 2877 (80%) | 0→4 |
| Valid | 360 (10%)  | 0→4 |
| Test  | 360 (10%)  | 0→4 |

**Classes:**
- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

**Imbalance:** Classes 1-4 underrepresented; models account for this via class weighting

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
```python
batch_size = 8  # Reduce batch size (slower but fits)
```

### Training too slow
- Check GPU availability: `torch.cuda.is_available()` should be `True`
- Reduce `num_epochs` for quick testing

### Checkpoint not loading
- Delete `.pth` files in `trained_model/` and restart
- Or remove individual model checkpoints in `results/hybrid_cnn_vit/<Model>/`

### Missing results
- Ensure training completed (check stdout for "Training completed" message)
- Results only appear after evaluation phase

---

## 📚 Paper Contribution Mapping

Each model addresses different research gaps:

| Model | Primary Contribution |
|-------|---------------------|
| Idea 1 | Uncertainty-aware fusion |
| Idea 2 | Multi-scale lesion detection |
| Idea 3 | Token quality control in ViT |
| **Idea 4** | **Ordinal geometry + QWK optimization** ✱ |
| **Idea 5** | **Prototype-guided contrastive learning** ✱ |
| Idea 6 | Anatomical relation modeling |
| Idea 7 | Spectral clue integration |
| **Idea 8** | **Severity-aware mixture of experts** ✱ |
| **Idea 9** | **Causal invariance learning** ✱ |
| **Idea 10** | **Tri-level distillation + calibration** ✱ Publication-level |

✱ = Particularly novel/complex (suitable for publication)

---

## 📞 Quick Reference

```bash
# Train all models
python train_all_hybrid_models.py

# Analyze results
jupyter notebook analyze_hybrid_models_results.ipynb

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# View best results
cat results/hybrid_cnn_vit/model_comparison_sorted.csv
```

---

**Last Updated**: April 2026
**Device**: GPU recommended (CUDA support)
**Python**: 3.8+, PyTorch 1.10+
