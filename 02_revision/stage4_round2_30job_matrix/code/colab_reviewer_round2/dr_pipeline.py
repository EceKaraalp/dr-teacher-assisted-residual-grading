"""Training / evaluation pipeline for the DR knowledge-distillation revision.

Architectures, losses, schedules and hyper-parameters are copied from the original
notebooks (`revision_experiments/run_teachers.py` for the teachers and
`export/TS_ConvNeXtTiny_Residual_Package/train_lowdata_hybrid_distill.ipynb` for the student).
Anything that is NEW relative to those files is marked `# NEW`.
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import convnext_tiny

from dr_common import (CLASS_NAMES, NUM_CLASSES, atomic_torch_save, class_weight_tensor,
                       compute_full_metrics, get_device, set_seed, write_json)
from models.hybrid_cnn_vit_base import HybridCNNViTBase
from models.advanced_hybrid_models import AdvancedHybridModel, ContrastiveLoss, PrototypicalLoss
from advanced_model_configs import get_advanced_model_config

device = get_device()

# ORIGINAL code left the frozen teacher(s) in train mode (BatchNorm statistics update, dropout active) while
# training the student / teacher gate. False = reproduce the original behaviour exactly. Set True only for a
# separate, clearly labelled ablation.
FREEZE_TEACHER_MODE = False

BASELINE_NAME = "Baseline_HybridCNNViT_Gate_Improved"
TEACHER_NAME = "Improved_Anchored_SpectralNorm"
BASELINE_CONFIG = {"fusion_method": "gate", "use_uncertainty_refinement": False,
                   "use_ordinal_head": False, "use_prototype_memory": False}

# Original hyper-parameters (run_teachers.py, cell 7)
BASELINE_HP = dict(num_epochs=110, lr=1e-3, patience=15)
ANCHORED_HP = dict(num_epochs=120, lr=1e-4, patience=18, lambda_start=1.5, lambda_end=0.6,
                   margin_start=0.02, margin_end=0.005, gate_entropy_beta=0.02)

# Original student recipe for TS_ConvNeXtTiny_Residual (train_lowdata_hybrid_distill.ipynb, cell 7)
STUDENT_HP = dict(num_epochs=70, lr=2e-4, patience=10)
STUDENT_MODEL_KW = dict(residual_scale=0.70, use_conflict_brake=True, conflict_threshold=0.82, conflict_gmax=0.66)
STUDENT_LOSS_CFG = dict(use_noninf=True, lambda_noninf_start=0.8, lambda_noninf_end=0.25,
                        margin_start=0.010, margin_end=0.003,
                        use_distill=True, lambda_distill=0.30, distill_temps=[2.0, 4.0], distill_weights=[0.6, 0.4],
                        use_ordinal=True, lambda_ordinal=0.12, beta_gate_entropy=0.01)


def _entropy(p):
    p = p.clamp_min(1e-8)
    return -(p * torch.log(p)).sum(dim=1, keepdim=True)


def count_params(m):
    tot = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return tot, tr, tot - tr


# ================================================================== models
class HeavyTeacherEnsemble(nn.Module):
    """Frozen baseline (Teacher 1) + Advanced SpectralNorm model (Teacher 2) + learned 13->64->32->1 gate.
    p = (1-g) p_base + g p_adv; returns log p. Equal-weight teacher analysis averages probabilities."""

    def __init__(self, baseline_model, adv_model, gate_dropout=0.1, fusion_mode="gate"):
        super().__init__()
        if fusion_mode not in {"gate", "equal_prob", "equal_logits", "t1", "t2"}:
            raise ValueError(f"Unknown teacher fusion mode: {fusion_mode}")
        self.baseline_model, self.adv_model = baseline_model, adv_model
        self.fusion_mode = fusion_mode
        for p in self.baseline_model.parameters():
            p.requires_grad = False
        self.gate = nn.Sequential(nn.Linear(13, 64), nn.ReLU(inplace=True), nn.Dropout(gate_dropout),
                                  nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        with torch.no_grad():
            base_logits = self.baseline_model(x)
            p_base = F.softmax(base_logits, dim=1)
        adv_logits, adv_extra = self.adv_model(x)
        p_adv = F.softmax(adv_logits, dim=1)
        conf_gap = (p_base.max(1, keepdim=True).values - p_adv.max(1, keepdim=True).values).abs()
        g = self.gate(torch.cat([p_base, p_adv, _entropy(p_base), _entropy(p_adv), conf_gap], dim=1))
        if self.fusion_mode == "gate":
            p_final = (1.0 - g) * p_base + g * p_adv
            z = torch.log(p_final.clamp_min(1e-8))
        elif self.fusion_mode == "equal_prob":
            p_final = (p_base + p_adv) / 2.0
            z = torch.log(p_final.clamp_min(1e-8))
        elif self.fusion_mode == "equal_logits":
            z = (base_logits + adv_logits) / 2.0
        elif self.fusion_mode == "t1":
            z = base_logits
        else:
            z = adv_logits
        return z, {"gate": g, "p_base": p_base, "p_adv": p_adv, "adv_extra": adv_extra,
                   "base_logits": base_logits, "adv_logits": adv_logits, "fusion_mode": self.fusion_mode}


class ConvNeXtTinyStudent(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = None
        if pretrained:
            from torchvision.models import ConvNeXt_Tiny_Weights
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1  # fail loudly: silent random init would invalidate results
        self.backbone = convnext_tiny(weights=weights)
        self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features, num_classes)

    def forward(self, x):
        return self.backbone(x), {}


class ResidualTeacherStudentModel(nn.Module):
    """mix='gated'  : z = z_t + a * g * (z_s - mean z_s), g learned + conflict brake  (ORIGINAL proposed model)
       mix='equal'  : z = (z_t + z_s) / 2, no student gate                            # NEW (advisor ablation)"""

    def __init__(self, teacher_model, student_model, residual_scale=0.8, use_conflict_brake=True,
                 conflict_threshold=0.80, conflict_gmax=0.65, mix="gated", teacher_mode="gate"):
        super().__init__()
        assert mix in ("gated", "equal")
        self.teacher_model, self.student_model, self.mix = teacher_model, student_model, mix
        if teacher_mode not in {"gate", "equal_prob", "equal_logits", "t1", "t2"}:
            raise ValueError(f"Unknown teacher fusion mode: {teacher_mode}")
        self.teacher_model.fusion_mode = teacher_mode
        self.teacher_mode = teacher_mode
        self.residual_scale = float(residual_scale)
        self.use_conflict_brake = bool(use_conflict_brake)
        self.conflict_threshold, self.conflict_gmax = float(conflict_threshold), float(conflict_gmax)
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.gate = nn.Sequential(nn.Linear(13, 32), nn.ReLU(inplace=True), nn.Linear(32, 1), nn.Sigmoid())

    def train(self, mode=True):
        super().train(mode)
        if FREEZE_TEACHER_MODE:
            self.teacher_model.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            z_teacher, t_extra = self.teacher_model(x)
            p_teacher = F.softmax(z_teacher, dim=1)
        z_student, _ = self.student_model(x)
        p_student = F.softmax(z_student, dim=1)

        if self.mix == "equal":
            z_final = (z_teacher + z_student) / 2.0
            g = torch.full((x.size(0), 1), 0.5, device=x.device)
            return z_final, {"gate": g, "z_teacher": z_teacher, "z_student": z_student,
                             "teacher_gate": t_extra["gate"], "conflict_rate": torch.tensor(0.0, device=x.device)}

        conf_gap = (p_teacher.max(1, keepdim=True).values - p_student.max(1, keepdim=True).values).abs()
        g = self.gate(torch.cat([p_teacher, p_student, _entropy(p_teacher), _entropy(p_student), conf_gap], dim=1))
        conflict_rate = torch.tensor(0.0, device=x.device)
        if self.use_conflict_brake:
            t_conf, t_pred = p_teacher.max(dim=1)
            mask = (t_conf > self.conflict_threshold) & (t_pred != p_student.argmax(dim=1))
            if mask.any():
                cap = (self.conflict_gmax * (1.0 - t_conf)).unsqueeze(1)
                g = torch.where(mask.unsqueeze(1), torch.minimum(g, cap), g)
            conflict_rate = mask.float().mean()
        delta = z_student - z_student.mean(dim=1, keepdim=True)
        z_final = z_teacher + self.residual_scale * g * delta
        return z_final, {"gate": g, "z_teacher": z_teacher, "z_student": z_student,
                         "teacher_gate": t_extra["gate"], "conflict_rate": conflict_rate}


# ================================================================ trainers
class _Base:
    """Resume-safe fit loop shared by all trainers. Early stopping / model selection on VALIDATION QWK."""
    name = "run"

    def _init_paths(self, run_dir):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path, self.best_path = self.dir / "checkpoint_current.pth", self.dir / "best_model.pth"
        self.hist_path, self.metrics_path = self.dir / "training_history.json", self.dir / "test_metrics.json"

    def _model_state_for_checkpoint(self):
        return self.model.state_dict()

    def _load_model_checkpoint_state(self, state):
        self.model.load_state_dict(state)

    def _save_ckpt(self, epoch):
        atomic_torch_save({"epoch": epoch, "model_state": self._model_state_for_checkpoint(),
                           "optimizer_state": self.optimizer.state_dict(),
                           "scheduler_state": self.scheduler.state_dict(),
                           "history": self.history, "best_qwk": self.best_qwk, "wait": self.wait,
                           "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                           "rng_torch": torch.get_rng_state(),
                           "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                          self.ckpt_path)

    def _load_ckpt(self):
        if self.ckpt_path.exists():
            c = torch.load(self.ckpt_path, map_location=device, weights_only=False)
            self._load_model_checkpoint_state(c["model_state"])
            self.optimizer.load_state_dict(c["optimizer_state"])
            self.scheduler.load_state_dict(c["scheduler_state"])
            self.history, self.best_qwk, self.start_epoch = c["history"], c["best_qwk"], int(c["epoch"]) + 1
            self.wait = int(c.get("wait", 0))
            if "rng_python" in c:
                random.setstate(c["rng_python"])
                np.random.set_state(c["rng_numpy"])
                torch.set_rng_state(c["rng_torch"].cpu())
                if torch.cuda.is_available() and c["rng_cuda"] is not None:
                    torch.cuda.set_rng_state_all([state.cpu() for state in c["rng_cuda"]])
            print(f"[{self.name}] resumed at epoch {self.start_epoch}")

    def is_done(self):
        return self.metrics_path.exists() and self.best_path.exists()

    def fit(self, train_loader, val_loader, max_epochs=None):
        n_ep = self.num_epochs if max_epochs is None else min(self.num_epochs, max_epochs)
        for epoch in range(self.start_epoch, n_ep):
            t0 = time.time()
            tr = self.train_epoch(train_loader, epoch)
            val_loss, vm = self.validate(val_loader)
            for k, v in {**tr, "val_loss": val_loss, "val_acc": vm["accuracy"], "val_qwk": vm["qwk"]}.items():
                self.history.setdefault(k, []).append(v)
            if vm["qwk"] > self.best_qwk:
                self.best_qwk, self.wait = vm["qwk"], 0
                atomic_torch_save(self._model_state_for_checkpoint(), self.best_path)
            else:
                self.wait += 1
            self.scheduler.step()
            self._save_ckpt(epoch)
            print(f"[{self.name}] ep {epoch+1:03d}/{n_ep} tr={tr['train_loss']:.4f} val={val_loss:.4f} "
                  f"acc={vm['accuracy']:.4f} qwk={vm['qwk']:.4f} best={self.best_qwk:.4f} "
                  f"wait={self.wait}/{self.patience} {time.time()-t0:.0f}s")
            self.history.setdefault("epoch_seconds", []).append(float(time.time() - t0))
            if self.wait >= self.patience:
                print(f"[{self.name}] early stop at epoch {epoch+1}")
                break
        write_json(self.history, self.hist_path)

    @torch.no_grad()
    def _predict_logits(self, loader):
        self.model.eval()
        zs, ys = [], []
        for x, y in loader:
            z = self.model(x.to(device))
            z = z[0] if isinstance(z, (tuple, list)) else z
            zs.append(z.float().cpu())
            ys.append((y.argmax(1) if y.ndim > 1 else y).cpu())
        return torch.cat(zs).numpy(), torch.cat(ys).numpy()

    def validate(self, loader):
        z, y = self._predict_logits(loader)
        zt = torch.as_tensor(z)
        loss = float(F.cross_entropy(zt, torch.as_tensor(y), weight=self.class_w.cpu()))
        p = F.softmax(zt, 1).numpy()
        return loss, compute_full_metrics(y, p.argmax(1), p)


class BaselineTrainer(_Base):
    """Teacher 1: frozen-later HybridCNNViTBase (gate fusion)."""

    def __init__(self, run_dir, class_weights, seed=42, **hp):
        hp = {**BASELINE_HP, **hp}
        set_seed(seed)
        self.name, self.num_epochs, self.patience = BASELINE_NAME, hp["num_epochs"], hp["patience"]
        self._init_paths(run_dir)
        self.model = HybridCNNViTBase(num_classes=5, config=BASELINE_CONFIG).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=hp["lr"], weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.class_w = class_weight_tensor(class_weights, device)
        self.ce = nn.CrossEntropyLoss(weight=self.class_w)
        self.history, self.best_qwk, self.start_epoch, self.wait = {}, -1e9, 0, 0
        self._load_ckpt()

    def train_epoch(self, loader, epoch):
        self.model.train()
        tot = 0.0
        for x, y in loader:
            x, y = x.to(device), (y.argmax(1) if y.ndim > 1 else y).to(device)
            self.optimizer.zero_grad()
            loss = self.ce(self.model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            tot += float(loss.item())
        return {"train_loss": tot / max(1, len(loader))}


class AnchoredTeacherTrainer(_Base):
    """Retrain the advanced teacher under a specified teacher-fusion objective."""

    def __init__(self, run_dir, baseline_best_path, class_weights, seed=42, fusion_mode="gate", **hp):
        hp = {**ANCHORED_HP, **hp}
        if fusion_mode not in {"gate", "equal_prob", "equal_logits"}:
            raise ValueError(f"Unsupported anchored teacher fusion mode: {fusion_mode}")
        set_seed(seed)
        self.name, self.num_epochs, self.patience = f"{TEACHER_NAME}_{fusion_mode}", hp["num_epochs"], hp["patience"]
        self.fusion_mode = fusion_mode
        self.hp = hp
        self._init_paths(run_dir)
        base = HybridCNNViTBase(num_classes=5, config=BASELINE_CONFIG).to(device)
        base.load_state_dict(torch.load(baseline_best_path, map_location=device, weights_only=True))
        adv_cfg = get_advanced_model_config("Adv_Idea_5_SpectralNorm")
        adv = AdvancedHybridModel(num_classes=5, config=adv_cfg).to(device)
        self.model = HeavyTeacherEnsemble(base, adv, fusion_mode=fusion_mode).to(device)
        if fusion_mode != "gate":
            for parameter in self.model.gate.parameters():
                parameter.requires_grad = False
        params = [
            p for name, p in self.model.named_parameters()
            if p.requires_grad and (fusion_mode == "gate" or not name.startswith("gate."))
        ]
        self.optimizer = optim.AdamW(params, lr=hp["lr"], weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.class_w = class_weight_tensor(class_weights, device)
        self.nll_sample = nn.NLLLoss(weight=self.class_w, reduction="none")
        self.ce_sample = nn.CrossEntropyLoss(weight=self.class_w, reduction="none")
        self.aux = {}
        for k, v in adv_cfg.get("auxiliary_losses", {}).items():
            t, w = v.get("type", ""), float(v.get("weight", 0.0))
            if t == "contrastive":
                self.aux[k] = (ContrastiveLoss(temperature=0.1), w)
            elif t == "prototypical":
                self.aux[k] = (PrototypicalLoss(), w)
        self.history, self.best_qwk, self.start_epoch, self.wait = {}, -1e9, 0, 0
        self._load_ckpt()

    def _sched(self, epoch):
        p = epoch / max(1, self.num_epochs - 1)
        h = self.hp
        return (h["lambda_start"] + p * (h["lambda_end"] - h["lambda_start"]),
                h["margin_start"] + p * (h["margin_end"] - h["margin_start"]))

    def _aux_loss(self, ex, y):
        tot = torch.tensor(0.0, device=device)
        for k, (fn, w) in self.aux.items():
            if k == "contrastive" and "cnn_proj" in ex and "vit_proj" in ex:
                tot = tot + w * fn(ex["cnn_proj"], ex["vit_proj"], y)
            elif k == "prototypical" and "final_feat" in ex and "prototypes" in ex:
                tot = tot + w * fn(ex["final_feat"], ex["prototypes"], y)
        return tot

    def train_epoch(self, loader, epoch):
        self.model.train()
        if FREEZE_TEACHER_MODE:
            self.model.baseline_model.eval()
        lam, mar = self._sched(epoch)
        tot, gv = 0.0, []
        for x, y in loader:
            x, y = x.to(device), (y.argmax(1) if y.ndim > 1 else y).to(device)
            self.optimizer.zero_grad()
            logp, ex = self.model(x)
            main_vec = (self.ce_sample(logp, y) if self.fusion_mode == "equal_logits"
                        else self.nll_sample(logp, y))
            base_vec = self.nll_sample(torch.log(ex["p_base"].clamp_min(1e-8)), y)
            noninf = torch.relu(main_vec - base_vec + mar).mean()
            g = ex["gate"].clamp(1e-6, 1 - 1e-6)
            gate_ent = -(g * torch.log(g) + (1 - g) * torch.log(1 - g)).mean()
            loss = main_vec.mean() + lam * noninf + self._aux_loss(ex["adv_extra"], y)
            if self.fusion_mode == "gate":
                loss = loss - self.hp["gate_entropy_beta"] * gate_ent
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            tot += float(loss.item())
            if self.fusion_mode == "gate":
                gv.append(float(g.mean().item()))
        return {"train_loss": tot / max(1, len(loader)),
                "gate_mean": float(np.mean(gv)) if gv else None,
                "lambda_noninf": lam, "margin_noninf": mar}

    def validate(self, loader):
        z, y = self._predict_logits(loader)
        zt = torch.as_tensor(z)
        if self.fusion_mode == "equal_logits":
            loss = float(F.cross_entropy(zt, torch.as_tensor(y), weight=self.class_w.cpu()))
            p = F.softmax(zt, dim=1).numpy()
        else:
            loss = float(F.nll_loss(zt, torch.as_tensor(y), weight=self.class_w.cpu()))
            p = np.exp(z)
        return loss, compute_full_metrics(y, p.argmax(1), p)


def load_teacher(baseline_ckpt, teacher_ckpt, fusion_mode="gate"):
    base = HybridCNNViTBase(num_classes=5, config=BASELINE_CONFIG).to(device)
    base.load_state_dict(torch.load(baseline_ckpt, map_location=device, weights_only=True))
    baseline_state = {key: value.clone() for key, value in base.state_dict().items()}
    adv = AdvancedHybridModel(num_classes=5, config=get_advanced_model_config("Adv_Idea_5_SpectralNorm")).to(device)
    t = HeavyTeacherEnsemble(base, adv).to(device)
    t.load_state_dict(torch.load(teacher_ckpt, map_location=device, weights_only=True))
    if fusion_mode == "t1":
        t.baseline_model.load_state_dict(baseline_state)
    t.fusion_mode = fusion_mode
    for p in t.parameters():
        p.requires_grad = False
    return t.eval()


def load_student_checkpoint(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    merged = model.state_dict()
    unexpected = sorted(set(state) - set(merged))
    if unexpected:
        raise RuntimeError(f"Student checkpoint has unexpected keys: {unexpected}")
    merged.update(state)
    model.load_state_dict(merged, strict=True)
    return model


class StudentTrainer(_Base):
    """Residual/equal teacher-student training with non-inferiority + KD + ordinal loss (original recipe)."""

    def __init__(self, run_dir, name, baseline_ckpt, teacher_ckpt, class_weights, seed, mix="gated",
                 teacher_mode="gate", loss_cfg=None, **hp):
        hp = {**STUDENT_HP, **hp}
        set_seed(seed)
        self.name, self.num_epochs, self.patience, self.mix = name, hp["num_epochs"], hp["patience"], mix
        self.teacher_mode = teacher_mode
        self.cfg = {**STUDENT_LOSS_CFG, **(loss_cfg or {})}
        self._init_paths(run_dir)
        teacher = load_teacher(baseline_ckpt, teacher_ckpt, fusion_mode=teacher_mode)
        self.model = ResidualTeacherStudentModel(teacher, ConvNeXtTinyStudent(5), mix=mix,
                                                 teacher_mode=teacher_mode, **STUDENT_MODEL_KW).to(device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(params, lr=hp["lr"], weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.class_w = class_weight_tensor(class_weights, device)
        self.ce_sample = nn.CrossEntropyLoss(weight=self.class_w, reduction="none")
        self.history, self.best_qwk, self.start_epoch, self.wait = {}, -1e9, 0, 0
        self._load_ckpt()

    def _model_state_for_checkpoint(self):
        teacher_parameters = {name for name, _ in self.model.named_parameters()
                              if name.startswith("teacher_model.")}
        return {
            key: value for key, value in self.model.state_dict().items()
            if not key.startswith("teacher_model.") or key not in teacher_parameters
        }

    def _load_model_checkpoint_state(self, state):
        merged = self.model.state_dict()
        unexpected = sorted(set(state) - set(merged))
        if unexpected:
            raise RuntimeError(f"Student checkpoint has unexpected keys: {unexpected}")
        merged.update(state)
        self.model.load_state_dict(merged, strict=True)

    def _sched(self, epoch):
        c = self.cfg
        p = epoch / max(1, self.num_epochs - 1)
        return (c["lambda_noninf_start"] + p * (c["lambda_noninf_end"] - c["lambda_noninf_start"]),
                c["margin_start"] + p * (c["margin_end"] - c["margin_start"]))

    @staticmethod
    def _ordinal(logits, y):
        cdf_p = torch.cumsum(F.softmax(logits, 1), 1)
        cdf_t = torch.cumsum(F.one_hot(y, NUM_CLASSES).float(), 1)
        return F.mse_loss(cdf_p, cdf_t)

    def train_epoch(self, loader, epoch):
        self.model.train()
        lam, mar = self._sched(epoch)
        c = self.cfg
        tot, gv, ni, di, od = 0.0, [], [], [], []
        for x, y in loader:
            x, y = x.to(device), (y.argmax(1) if y.ndim > 1 else y).to(device)
            self.optimizer.zero_grad()
            logits, ex = self.model(x)
            main_vec = self.ce_sample(logits, y)
            with torch.no_grad():
                t_vec = self.ce_sample(ex["z_teacher"], y)
            noninf = torch.relu(main_vec - t_vec + mar).mean() if c["use_noninf"] else torch.tensor(0.0, device=device)
            distill = torch.tensor(0.0, device=device)
            if c["use_distill"]:
                zt = ex["z_teacher"].detach()
                for T, w in zip(c["distill_temps"], c["distill_weights"]):
                    distill = distill + w * F.kl_div(F.log_softmax(logits / T, 1), F.softmax(zt / T, 1),
                                                     reduction="batchmean") * (T * T)
            ordl = self._ordinal(logits, y) if c["use_ordinal"] else torch.tensor(0.0, device=device)
            loss = main_vec.mean() + lam * noninf + c["lambda_distill"] * distill + c["lambda_ordinal"] * ordl
            if self.mix == "gated":  # original gate-entropy regulariser; meaningless for the fixed equal mix
                g = ex["gate"].clamp(1e-6, 1 - 1e-6)
                loss = loss - c["beta_gate_entropy"] * (-(g * torch.log(g) + (1 - g) * torch.log(1 - g)).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            tot += float(loss.item())
            gv.append(float(ex["gate"].mean().item()))
            ni.append(float(noninf.item())); di.append(float(distill.item())); od.append(float(ordl.item()))
        return {"train_loss": tot / max(1, len(loader)), "gate_mean": float(np.mean(gv)),
                "loss_noninf": float(np.mean(ni)), "loss_distill": float(np.mean(di)),
                "loss_ordinal": float(np.mean(od)), "lambda_noninf": lam, "margin_noninf": mar}


# ============================================================ evaluation
@torch.no_grad()
def collect_outputs(model, loader, kind):
    """Per-sample outputs of a trained model on a loader.
    kind='baseline' -> {z_base}; 'teacher' -> {z_t1, z_t2, z_teacher, z_equal_teacher, g_teacher};
    'student' -> final, teacher, student logits and both gate values."""
    model.eval()
    acc, ys = {}, []

    def add(k, v):
        acc.setdefault(k, []).append(v.float().cpu())

    for x, y in loader:
        x = x.to(device)
        ys.append((y.argmax(1) if y.ndim > 1 else y).cpu())
        if kind == "baseline":
            add("z_base", model(x))
        elif kind == "teacher":
            z, ex = model(x)
            add("z_t1", ex["base_logits"]); add("z_t2", ex["adv_logits"])
            add("z_teacher", z); add("g_teacher", ex["gate"])
            p_equal = (F.softmax(ex["base_logits"], dim=1) + F.softmax(ex["adv_logits"], dim=1)) / 2.0
            add("z_equal_teacher", torch.log(p_equal.clamp_min(1e-8)))
        else:
            z, ex = model(x)
            add("z_final", z); add("z_student", ex["z_student"]); add("z_teacher", ex["z_teacher"])
            add("g_student", ex["gate"]); add("g_teacher", ex["teacher_gate"])
    out = {k: torch.cat(v).numpy() for k, v in acc.items()}
    out["y"] = torch.cat(ys).numpy()
    return out


@torch.no_grad()
def time_model(model, input_shape=(1, 3, 224, 224), warmup=20, iters=100):
    """Latency of one forward pass at batch size 1 (ms) - includes teachers if they are inside `model`."""
    model.eval()
    x = torch.rand(*input_shape, device=device) * 255.0
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = float(np.mean(ts))
    return {
        "batch_size": int(input_shape[0]),
        "mean_ms_per_batch": mean_ms,
        "median_ms_per_batch": float(np.median(ts)),
        "p95_ms_per_batch": float(np.percentile(ts, 95)),
        "std_ms_per_batch": float(np.std(ts)),
        "mean_ms_per_image": mean_ms / int(input_shape[0]),
        "images_per_second": (int(input_shape[0]) * 1000.0 / mean_ms) if mean_ms > 0 else None,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "warmup_batches": warmup,
        "timed_batches": iters,
    }
