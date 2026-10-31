import matplotlib
matplotlib.use('Agg')
# ---- notebook cell 1 ----
# 1) Imports
import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    cohen_kappa_score
)

warnings.filterwarnings('ignore')

sns.set_palette('husl')

print('Libraries imported')
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))

# ---- notebook cell 2 ----
# 2) Setup and Module Imports
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results_dir = Path(os.environ.get('RESULTS_DIR', 'results/improved_unique_anchored_hybrid'))
results_dir.mkdir(parents=True, exist_ok=True)

required_files = [
    'models/hybrid_cnn_vit_base.py',
    'models/advanced_hybrid_models.py',
    'advanced_model_configs.py',
    'model_configs.py',
    'dataset_loader.py'
 ]
for f in required_files:
    if not Path(f).exists():
        raise FileNotFoundError(f'Missing required file: {f}')

import importlib
for m in [
    'models.hybrid_cnn_vit_base',
    'models.advanced_hybrid_models',
    'advanced_model_configs',
    'dataset_loader',
    'model_configs'
 ]:
    if m in sys.modules:
        importlib.reload(sys.modules[m])

from models.hybrid_cnn_vit_base import HybridCNNViTBase
from models.advanced_hybrid_models import AdvancedHybridModel, ContrastiveLoss, PrototypicalLoss
from advanced_model_configs import get_advanced_model_config
from model_configs import get_model_config
from dataset_loader import get_data_loaders

print('Device:', device)
print('Results dir:', results_dir)

# ---- notebook cell 3 ----
# 3) Load Data (80-10-10)
batch_size = 16
train_loader, val_loader, test_loader, class_weights, split_data = get_data_loaders(
    dataset_path='APTOS2019',
    batch_size=batch_size
)

X_train, y_train, X_val, y_val, X_test, y_test = split_data
total = len(X_train) + len(X_val) + len(X_test)

print('Dataset loaded')
print(f'Train: {len(X_train)} ({len(X_train)/total*100:.1f}%)')
print(f'Val:   {len(X_val)} ({len(X_val)/total*100:.1f}%)')
print(f'Test:  {len(X_test)} ({len(X_test)/total*100:.1f}%)')
print('Class weights:', class_weights)

# ---- notebook cell 4 ----
# 4) Metric Helpers
def _mean_specificity(y_true, y_pred, num_classes=5):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specs = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        denom = tn + fp
        specs.append((tn / denom) if denom > 0 else 0.0)
    return float(np.mean(specs))

def compute_full_metrics(y_true, y_pred, y_prob, num_classes=5):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['specificity'] = _mean_specificity(y_true, y_pred, num_classes=num_classes)
    metrics['qwk'] = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))

    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted'))
    except Exception:
        metrics['roc_auc'] = 0.0

    return metrics

def class_weight_tensor_from_dict(class_weights_dict, device, n_classes=5):
    if isinstance(class_weights_dict, dict):
        arr = [float(class_weights_dict.get(i, 1.0)) for i in range(n_classes)]
    else:
        arr = [1.0] * n_classes
    return torch.tensor(arr, dtype=torch.float32, device=device)

print('Metric helpers ready')

# ---- notebook cell 5 ----
# 5) Models: Improved Baseline-Anchored Ensemble
class ImprovedBaselineAnchoredEnsemble(nn.Module):
    """
    Improved safety layer:
    - Baseline branch is frozen (stable anchor)
    - Feature-aware gate uses probabilities + uncertainty features
    - Final mix: p_final = (1-g)*p_base + g*p_adv
    """
    def __init__(self, baseline_model: nn.Module, adv_model: nn.Module, gate_dropout: float = 0.1):
        super().__init__()
        self.baseline_model = baseline_model
        self.adv_model = adv_model

        for p in self.baseline_model.parameters():
            p.requires_grad = False

        # gate input: [p_base(5), p_adv(5), entropy_base(1), entropy_adv(1), conf_gap(1)] => 13
        self.gate = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(gate_dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def _entropy(p: torch.Tensor) -> torch.Tensor:
        p = p.clamp_min(1e-8)
        return -(p * torch.log(p)).sum(dim=1, keepdim=True)

    def forward(self, x):
        with torch.no_grad():
            base_logits = self.baseline_model(x)
            p_base = F.softmax(base_logits, dim=1)

        adv_logits, adv_extra = self.adv_model(x)
        p_adv = F.softmax(adv_logits, dim=1)

        ent_base = self._entropy(p_base)
        ent_adv = self._entropy(p_adv)
        conf_gap = (p_base.max(dim=1, keepdim=True).values - p_adv.max(dim=1, keepdim=True).values).abs()

        gate_in = torch.cat([p_base, p_adv, ent_base, ent_adv, conf_gap], dim=1)
        g = self.gate(gate_in)

        p_final = (1.0 - g) * p_base + g * p_adv
        log_p_final = torch.log(p_final.clamp_min(1e-8))

        extra = {
            'gate': g,
            'p_base': p_base,
            'p_adv': p_adv,
            'ent_base': ent_base,
            'ent_adv': ent_adv,
            'adv_extra': adv_extra
        }
        return log_p_final, extra

print('ImprovedBaselineAnchoredEnsemble ready')

# ---- notebook cell 6 ----
# 6) Trainers (Checkpoint-Resume Enabled, Improved Constraints)
class BaselineTrainer:
    def __init__(self, model_name: str, config: Dict, num_epochs=80, lr=1e-3):
        self.model_name = model_name
        self.config = config
        self.num_epochs = num_epochs
        self.device = device

        self.model_dir = results_dir / model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.model_dir / 'checkpoint_current.pth'
        self.best_path = self.model_dir / 'best_model.pth'
        self.hist_path = self.model_dir / 'training_history.json'
        self.metrics_path = self.model_dir / 'test_metrics.json'

        self.model = HybridCNNViTBase(num_classes=5, config=config).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.class_w = class_weight_tensor_from_dict(class_weights, self.device, n_classes=5)
        self.ce = nn.CrossEntropyLoss(weight=self.class_w)

        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_qwk': []}
        self.best_qwk = -1e9
        self.start_epoch = 0
        self._load_checkpoint()

    def _load_checkpoint(self):
        if self.ckpt_path.exists():
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'])
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
            self.scheduler.load_state_dict(ckpt['scheduler_state'])
            self.history = ckpt['history']
            self.best_qwk = ckpt['best_qwk']
            self.start_epoch = int(ckpt['epoch']) + 1
            print(f'[Baseline] Resume epoch {self.start_epoch}')

    def _save_checkpoint(self, epoch):
        ckpt = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'history': self.history,
            'best_qwk': self.best_qwk
        }
        torch.save(ckpt, self.ckpt_path)

    def train_epoch(self, loader):
        self.model.train()
        total = 0.0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            y = labels.argmax(1) if labels.ndim > 1 else labels

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.ce(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += float(loss.item())

        return total / max(1, len(loader))

    def validate(self, loader):
        self.model.eval()
        total = 0.0
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                y = labels.argmax(1) if labels.ndim > 1 else labels

                logits = self.model(images)
                loss = self.ce(logits, y)
                total += float(loss.item())

                prob = F.softmax(logits, dim=1)
                pred = prob.argmax(1)
                y_true.extend(y.detach().cpu().numpy().tolist())
                y_pred.extend(pred.detach().cpu().numpy().tolist())
                y_prob.extend(prob.detach().cpu().numpy().tolist())

        val_loss = total / max(1, len(loader))
        m = compute_full_metrics(y_true, y_pred, y_prob, num_classes=5)
        return val_loss, m

    def fit(self, train_loader, val_loader, patience=15):
        wait = 0
        for epoch in range(self.start_epoch, self.num_epochs):
            tr_loss = self.train_epoch(train_loader)
            val_loss, vm = self.validate(val_loader)

            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(vm['accuracy'])
            self.history['val_qwk'].append(vm['qwk'])

            self._save_checkpoint(epoch)

            if vm['qwk'] > self.best_qwk:
                self.best_qwk = vm['qwk']
                torch.save(self.model.state_dict(), self.best_path)
                wait = 0
            else:
                wait += 1

            self.scheduler.step()
            print(f"[Baseline] Epoch {epoch+1:03d} | tr={tr_loss:.4f} | val={val_loss:.4f} | acc={vm['accuracy']:.4f} | qwk={vm['qwk']:.4f}")

            if wait >= patience:
                print(f'[Baseline] Early stopping at epoch {epoch+1}')
                break

        with open(self.hist_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)

    def evaluate_test(self, test_loader):
        self.model.load_state_dict(torch.load(self.best_path, map_location=self.device))
        self.model.eval()

        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                y = labels.argmax(1) if labels.ndim > 1 else labels

                logits = self.model(images)
                prob = F.softmax(logits, dim=1)
                pred = prob.argmax(1)

                y_true.extend(y.detach().cpu().numpy().tolist())
                y_pred.extend(pred.detach().cpu().numpy().tolist())
                y_prob.extend(prob.detach().cpu().numpy().tolist())

        m = compute_full_metrics(y_true, y_pred, y_prob, num_classes=5)
        cm = confusion_matrix(y_true, y_pred).tolist()
        report = classification_report(
            y_true, y_pred,
            target_names=['No DR','Mild','Moderate','Severe','Proliferative'],
            output_dict=True
        )

        result = {'model_name': self.model_name, **m, 'confusion_matrix': cm, 'classification_report': report}
        with open(self.metrics_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        return result


class ImprovedAnchoredTrainer:
    def __init__(
        self,
        model_name: str,
        baseline_config: Dict,
        baseline_best_path: Path,
        adv_config: Dict,
        num_epochs=80,
        lr=1e-4,
        lambda_start=1.5,
        lambda_end=0.6,
        margin_start=0.02,
        margin_end=0.005,
        gate_entropy_beta=0.02
    ):
        self.model_name = model_name
        self.baseline_config = baseline_config
        self.adv_config = adv_config
        self.num_epochs = int(num_epochs)
        self.device = device

        self.lambda_start = float(lambda_start)
        self.lambda_end = float(lambda_end)
        self.margin_start = float(margin_start)
        self.margin_end = float(margin_end)
        self.gate_entropy_beta = float(gate_entropy_beta)

        self.model_dir = results_dir / model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.model_dir / 'checkpoint_current.pth'
        self.best_path = self.model_dir / 'best_model.pth'
        self.hist_path = self.model_dir / 'training_history.json'
        self.metrics_path = self.model_dir / 'test_metrics.json'

        base_model = HybridCNNViTBase(num_classes=5, config=baseline_config).to(self.device)
        base_model.load_state_dict(torch.load(baseline_best_path, map_location=self.device))

        adv_model = AdvancedHybridModel(num_classes=5, config=adv_config).to(self.device)
        self.model = ImprovedBaselineAnchoredEnsemble(base_model, adv_model).to(self.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)

        self.class_w = class_weight_tensor_from_dict(class_weights, self.device, n_classes=5)
        self.nll_sample = nn.NLLLoss(weight=self.class_w, reduction='none')
        self.nll_mean = nn.NLLLoss(weight=self.class_w)

        self.aux_loss_fns = {}
        aux_cfg = self.adv_config.get('auxiliary_losses', {})
        for aux_name, aux_params in aux_cfg.items():
            t = aux_params.get('type', '')
            w = float(aux_params.get('weight', 0.0))
            if t == 'contrastive':
                self.aux_loss_fns[aux_name] = (ContrastiveLoss(temperature=0.1), w)
            elif t == 'prototypical':
                self.aux_loss_fns[aux_name] = (PrototypicalLoss(), w)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_qwk': [],
            'gate_mean': [],
            'lambda_noninf': [],
            'margin_noninf': []
        }
        self.best_qwk = -1e9
        self.start_epoch = 0
        self._load_checkpoint()

    def _load_checkpoint(self):
        if self.ckpt_path.exists():
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'])
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
            self.scheduler.load_state_dict(ckpt['scheduler_state'])
            self.history = ckpt['history']
            self.best_qwk = ckpt['best_qwk']
            self.start_epoch = int(ckpt['epoch']) + 1
            print(f'[{self.model_name}] Resume epoch {self.start_epoch}')

    def _save_checkpoint(self, epoch):
        ckpt = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'history': self.history,
            'best_qwk': self.best_qwk
        }
        torch.save(ckpt, self.ckpt_path)

    def _schedule_noninf(self, epoch: int):
        if self.num_epochs <= 1:
            return self.lambda_end, self.margin_end
        progress = float(epoch) / float(self.num_epochs - 1)
        cur_lambda = self.lambda_start + progress * (self.lambda_end - self.lambda_start)
        cur_margin = self.margin_start + progress * (self.margin_end - self.margin_start)
        return float(cur_lambda), float(cur_margin)

    def _aux_loss(self, adv_extra, y):
        total_aux = torch.tensor(0.0, device=self.device)
        for aux_name, (fn, w) in self.aux_loss_fns.items():
            if aux_name == 'contrastive' and 'cnn_proj' in adv_extra and 'vit_proj' in adv_extra:
                total_aux = total_aux + w * fn(adv_extra['cnn_proj'], adv_extra['vit_proj'], y)
            elif aux_name == 'prototypical' and 'final_feat' in adv_extra and 'prototypes' in adv_extra:
                total_aux = total_aux + w * fn(adv_extra['final_feat'], adv_extra['prototypes'], y)
        return total_aux

    def train_epoch(self, loader, cur_lambda, cur_margin):
        self.model.train()
        total = 0.0
        gate_vals = []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            y = labels.argmax(1) if labels.ndim > 1 else labels

            self.optimizer.zero_grad()
            log_p_final, extra = self.model(images)

            loss_main_vec = self.nll_sample(log_p_final, y)
            loss_main = loss_main_vec.mean()

            base_log = torch.log(extra['p_base'].clamp_min(1e-8))
            loss_base_vec = self.nll_sample(base_log, y)
            loss_noninf = torch.relu(loss_main_vec - loss_base_vec + cur_margin).mean()

            adv_extra = extra.get('adv_extra', {})
            loss_aux = self._aux_loss(adv_extra, y)

            g = extra['gate'].clamp(1e-6, 1 - 1e-6)
            gate_entropy = -(g * torch.log(g) + (1 - g) * torch.log(1 - g)).mean()

            loss = loss_main + cur_lambda * loss_noninf + loss_aux - self.gate_entropy_beta * gate_entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total += float(loss.item())
            gate_vals.append(float(g.mean().detach().cpu().item()))

        gate_mean = float(np.mean(gate_vals)) if gate_vals else 0.0
        return total / max(1, len(loader)), gate_mean

    def validate(self, loader):
        self.model.eval()
        total = 0.0
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                y = labels.argmax(1) if labels.ndim > 1 else labels

                log_p_final, _ = self.model(images)
                loss = self.nll_mean(log_p_final, y)
                total += float(loss.item())

                prob = torch.exp(log_p_final)
                pred = prob.argmax(1)
                y_true.extend(y.detach().cpu().numpy().tolist())
                y_pred.extend(pred.detach().cpu().numpy().tolist())
                y_prob.extend(prob.detach().cpu().numpy().tolist())

        val_loss = total / max(1, len(loader))
        m = compute_full_metrics(y_true, y_pred, y_prob, num_classes=5)
        return val_loss, m

    def fit(self, train_loader, val_loader, patience=15):
        wait = 0
        for epoch in range(self.start_epoch, self.num_epochs):
            cur_lambda, cur_margin = self._schedule_noninf(epoch)
            tr_loss, gate_mean = self.train_epoch(train_loader, cur_lambda, cur_margin)
            val_loss, vm = self.validate(val_loader)

            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(vm['accuracy'])
            self.history['val_qwk'].append(vm['qwk'])
            self.history['gate_mean'].append(gate_mean)
            self.history['lambda_noninf'].append(cur_lambda)
            self.history['margin_noninf'].append(cur_margin)

            self._save_checkpoint(epoch)

            if vm['qwk'] > self.best_qwk:
                self.best_qwk = vm['qwk']
                torch.save(self.model.state_dict(), self.best_path)
                wait = 0
            else:
                wait += 1

            self.scheduler.step()
            print(
                f"[{self.model_name}] Epoch {epoch+1:03d} | tr={tr_loss:.4f} | val={val_loss:.4f} | "
                f"acc={vm['accuracy']:.4f} | qwk={vm['qwk']:.4f} | gate={gate_mean:.3f} | "
                f"lambda={cur_lambda:.3f} | margin={cur_margin:.4f}"
            )

            if wait >= patience:
                print(f'[{self.model_name}] Early stopping at epoch {epoch+1}')
                break

        with open(self.hist_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)

    def evaluate_test(self, test_loader):
        self.model.load_state_dict(torch.load(self.best_path, map_location=self.device))
        self.model.eval()

        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                y = labels.argmax(1) if labels.ndim > 1 else labels

                log_p_final, _ = self.model(images)
                prob = torch.exp(log_p_final)
                pred = prob.argmax(1)

                y_true.extend(y.detach().cpu().numpy().tolist())
                y_pred.extend(pred.detach().cpu().numpy().tolist())
                y_prob.extend(prob.detach().cpu().numpy().tolist())

        m = compute_full_metrics(y_true, y_pred, y_prob, num_classes=5)
        cm = confusion_matrix(y_true, y_pred).tolist()
        report = classification_report(
            y_true, y_pred,
            target_names=['No DR','Mild','Moderate','Severe','Proliferative'],
            output_dict=True
        )

        result = {'model_name': self.model_name, **m, 'confusion_matrix': cm, 'classification_report': report}
        with open(self.metrics_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        return result

print('Trainers defined: BaselineTrainer + ImprovedAnchoredTrainer')

# ---- notebook cell 7 (only baseline + SpectralNorm teachers) ----
# 7) Train Baseline + Improved Anchored Models (Resume-safe)
baseline_name = 'Baseline_HybridCNNViT_Gate_Improved'
baseline_config = {
    'fusion_method': 'gate',
    'use_uncertainty_refinement': False,
    'use_ordinal_head': False,
    'use_prototype_memory': False
}

print('='*100)
print('STEP 1: BASELINE TRAINING (ANCHOR MODEL)')
print('='*100)

baseline_trainer = BaselineTrainer(
    model_name=baseline_name,
    config=baseline_config,
    num_epochs=110,
    lr=1e-3
)
if (results_dir / baseline_name / 'test_metrics.json').exists() and baseline_trainer.best_path.exists():
    print('Baseline already finished; skipping training (re-evaluating saved best model).')
else:
    baseline_trainer.fit(train_loader, val_loader, patience=15)
baseline_test = baseline_trainer.evaluate_test(test_loader)

all_results = {baseline_name: baseline_test}
summary_rows = [{
    'Model': baseline_name,
    'Accuracy': baseline_test['accuracy'],
    'F1_Macro': baseline_test['f1_macro'],
    'QWK': baseline_test['qwk'],
    'ROC_AUC': baseline_test['roc_auc'],
    'Specificity': baseline_test['specificity']
}]

print(f"Baseline Test -> Acc={baseline_test['accuracy']:.4f}, F1={baseline_test['f1_macro']:.4f}, QWK={baseline_test['qwk']:.4f}")

print('\n' + '='*100)
print('STEP 2: IMPROVED ANCHORED MODELS')
print('='*100)

# Focus on most promising advanced configurations
anchored_experiments = [
    ('Improved_Anchored_SpectralNorm', get_advanced_model_config('Adv_Idea_5_SpectralNorm'))
]

for exp_name, adv_cfg in anchored_experiments:
    print('\n' + '#'*100)
    print('Training:', exp_name)
    print('#'*100)

    try:
        trainer = ImprovedAnchoredTrainer(
            model_name=exp_name,
            baseline_config=baseline_config,
            baseline_best_path=baseline_trainer.best_path,
            adv_config=adv_cfg,
            num_epochs=120,
            lr=1e-4,
            lambda_start=1.5,
            lambda_end=0.6,
            margin_start=0.02,
            margin_end=0.005,
            gate_entropy_beta=0.02
        )

        trainer.fit(train_loader, val_loader, patience=18)
        test_result = trainer.evaluate_test(test_loader)

        all_results[exp_name] = test_result
        summary_rows.append({
            'Model': exp_name,
            'Accuracy': test_result['accuracy'],
            'F1_Macro': test_result['f1_macro'],
            'QWK': test_result['qwk'],
            'ROC_AUC': test_result['roc_auc'],
            'Specificity': test_result['specificity']
        })

        print(f"{exp_name} Test -> Acc={test_result['accuracy']:.4f}, F1={test_result['f1_macro']:.4f}, QWK={test_result['qwk']:.4f}")

    except Exception as e:
        print('Error on', exp_name, ':', str(e))
        all_results[exp_name] = {'error': str(e)}

print('\nAll trainings finished')