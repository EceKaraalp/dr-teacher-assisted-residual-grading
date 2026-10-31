"""Manuscript-faithful retraining of ConvNeXtTiny_Residual (Section II of manuscript 2020440).

Implements the method exactly as written in the original manuscript, not the historical notebook code:
  * Teacher 1 and Teacher 2 are trained independently (no joint mixture, no frozen-T1 anchoring).
  * Teacher signal z_T = (z_T1 + z_T2) / 2 (Eq. 2); no learned teacher gate in the proposed system.
  * The ConvNeXt Tiny student is trained on its OWN logits z_S with
    weighted CE + lambda_NI(e) * non-inferiority + 0.30 * multi-T KD (T=2/4, 0.6/0.4) + 0.12 * ordinal CDF (Eq. 3).
  * Teachers are pre-trained and frozen in eval mode during distillation (standard KD soft targets).
  * Inference: z = (z_T + z_S) / 2 (Eq. 8); student-only softmax(z_S) is reported as the compact mode.
  * Entropy-gated teacher fusion (Table 2 "Gated") is trained on frozen teacher logits for comparison only.

Selection uses validation QWK only. The test split is evaluated in a separate stage after every job is complete.
"""
import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision.models import convnext_tiny

from data import file_hash, load_data, preprocess
from dr_common import compute_full_metrics, set_seed
from models.hybrid_cnn_vit_base import HybridCNNViTBase
from models.advanced_hybrid_models import AdvancedHybridModel
from advanced_model_configs import get_advanced_model_config


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (42, 52, 62)
BASELINE_CONFIG = {"fusion_method": "gate", "use_uncertainty_refinement": False,
                   "use_ordinal_head": False, "use_prototype_memory": False}
STUDENT_ARMS = ("full", "ce_only", "no_kd", "no_ordinal", "no_noninf", "single_temp")
HYPER = {
    # Teacher settings are not given in the manuscript; the historical teacher recipe is reused.
    "teacher1": {"epochs": 110, "patience": 15, "lr": 1e-3, "batch": 16},
    "teacher2": {"epochs": 120, "patience": 18, "lr": 1e-4, "batch": 16},
    "teacher_gate": {"epochs": 200, "patience": 20, "lr": 1e-3, "batch": 16},
    # Manuscript Section II.H.3.
    "student": {"epochs": 70, "patience": 10, "lr": 2e-4, "batch": 16},
}
WEIGHT_DECAY = 1e-4


@dataclass(frozen=True)
class Job:
    seed: int
    stage: str
    arm: str = ""

    @property
    def key(self):
        return f"s{self.seed}/{self.stage}" + (f"/{self.arm}" if self.arm else "")


def jobs():
    result = []
    for seed in SEEDS:
        result += [Job(seed, "teacher1"), Job(seed, "teacher2"), Job(seed, "teacher_gate")]
        result += [Job(seed, "student", arm) for arm in STUDENT_ARMS]
    return result


def loss_config(arm):
    if arm not in STUDENT_ARMS:
        raise ValueError(arm)
    return {
        "kd": arm not in {"ce_only", "no_kd"},
        "ordinal": arm not in {"ce_only", "no_ordinal"},
        "noninf": arm not in {"ce_only", "no_noninf"},
        "temperatures": [2.0] if arm == "single_temp" else [2.0, 4.0],
        "temperature_weights": [1.0] if arm == "single_temp" else [0.6, 0.4],
        "lambda_distill": 0.30, "lambda_ordinal": 0.12,
        "lambda_noninf": [0.80, 0.25], "margin": [0.010, 0.003],
        # ce_only has no teacher during training; it is selected on its own logits.
        "selection": "student" if arm == "ce_only" else "system",
    }


def protocol():
    return {
        "name": "manuscript_faithful_v1", "seeds": SEEDS,
        "split": "frozen historical 2928/367/367 (split_manifest.json); test IDs exposed in earlier work",
        "input": "RGB bilinear 224, float32 0..255, no normalization, no augmentation",
        "teachers": "T1 HybridCNNViTBase(gate) and T2 ResNet50+ViT-B/16 spectral-norm head trained independently "
                    "with class-weighted CE; selection on own validation QWK; retrained for every seed",
        "teacher_signal": "z_T = (z_T1 + z_T2)/2 raw logits; teachers frozen in eval mode during distillation",
        "student_loss": "mean_i[w_y CE_i] + lambda(e)*mean_i relu(CE_S,i - CE_T,i + m(e)) "
                        "+ 0.30*sum_k a_k T_k^2 KL(softmax(z_T/T_k)||softmax(z_S/T_k)) + 0.12*MSE(CDF_S, CDF_y)",
        "schedules": "lambda 0.80->0.25 and margin 0.010->0.003 linear over epochs 1..70",
        "inference": "full system (z_T + z_S)/2; student-only z_S",
        "teacher_gate": "MLP 13->64->1 sigmoid on [p1,p2,H1,H2,|conf1-conf2|]; z=(1-g)z_T1+g*z_T2; "
                        "trained on frozen train-set teacher logits with class-weighted CE",
        "student_arms": {arm: loss_config(arm) for arm in STUDENT_ARMS},
        "hyper": HYPER, "optimizer": "AdamW wd 1e-4; grad clip 1.0; CosineAnnealingWarmRestarts(T_0=10, T_mult=2)",
        "selection": "validation QWK, patience as listed; test never used for any choice",
        "jobs": [asdict(j) for j in jobs()],
    }


def fingerprint():
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ models
class ConvNeXtTinyStudent(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = None
        if pretrained:
            from torchvision.models import ConvNeXt_Tiny_Weights
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.backbone = convnext_tiny(weights=weights)
        self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features, 5)

    def forward(self, x):
        return self.backbone(x)


class AdvancedTeacher(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = AdvancedHybridModel(num_classes=5, config=get_advanced_model_config("Adv_Idea_5_SpectralNorm"),
                                         pretrained=pretrained)

    def forward(self, x):
        return self.model(x)[0]


def _entropy(p):
    p = p.clamp_min(1e-8)
    return -(p * p.log()).sum(1, keepdim=True)


class EntropyGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(13, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, z1, z2):
        p1, p2 = z1.softmax(1), z2.softmax(1)
        gap = (p1.max(1, keepdim=True).values - p2.max(1, keepdim=True).values).abs()
        g = self.net(torch.cat([p1, p2, _entropy(p1), _entropy(p2), gap], 1))
        return (1 - g) * z1 + g * z2, g


def build_model(job, pretrained=True):
    set_seed(job.seed)
    if job.stage == "teacher1":
        return HybridCNNViTBase(num_classes=5, config=BASELINE_CONFIG)
    if job.stage == "teacher2":
        return AdvancedTeacher(pretrained)
    if job.stage == "teacher_gate":
        return EntropyGate()
    return ConvNeXtTinyStudent(pretrained)


# ------------------------------------------------------------------ losses
def schedule(epoch, start, end, epochs=70):
    progress = min(max(epoch, 0), epochs - 1) / (epochs - 1)
    return start + progress * (end - start)


def ordinal_cdf_loss(z, y):
    return F.mse_loss(z.softmax(1).cumsum(1), F.one_hot(y, 5).float().cumsum(1))


def student_loss(z_s, z_t, y, weights, arm, epoch):
    config = loss_config(arm)
    ce_s = F.cross_entropy(z_s, y, weight=weights, reduction="none")
    parts = {"ce": ce_s.mean()}
    if config["noninf"]:
        ce_t = F.cross_entropy(z_t, y, weight=weights, reduction="none")
        margin = schedule(epoch, *config["margin"])
        parts["noninf"] = schedule(epoch, *config["lambda_noninf"]) * F.relu(ce_s - ce_t + margin).mean()
    if config["kd"]:
        kd = 0.0
        for temperature, weight in zip(config["temperatures"], config["temperature_weights"]):
            kd = kd + weight * temperature ** 2 * F.kl_div(
                F.log_softmax(z_s / temperature, 1), F.softmax(z_t / temperature, 1), reduction="batchmean")
        parts["kd"] = config["lambda_distill"] * kd
    if config["ordinal"]:
        parts["ordinal"] = config["lambda_ordinal"] * ordinal_cdf_loss(z_s, y)
    return sum(parts.values()), {k: float(v) for k, v in parts.items()}


# -------------------------------------------------------------------- data
class Indexed(Dataset):
    def __init__(self, images, labels, indices):
        self.images, self.labels, self.indices = images, labels, np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        position = int(self.indices[i])
        return preprocess(self.images[position], "raw"), int(self.labels[position]), position


def make_loaders(bundle, seed, batch=16):
    _, images, labels, splits = bundle
    counts = np.bincount(labels[splits["train"]], minlength=5)
    weights = torch.tensor(len(splits["train"]) / (5 * counts), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    result = {split: DataLoader(Indexed(images, labels, splits[split]), batch_size=batch, shuffle=split == "train",
                                generator=generator if split == "train" else None, num_workers=0,
                                pin_memory=torch.cuda.is_available())
              for split in ("train", "val", "test")}
    return result, weights


# ------------------------------------------------------------------- utils
def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_save(path, value):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def synchronize(folder, remote):
    if remote is None:
        return
    remote.mkdir(parents=True, exist_ok=True)
    for source in folder.iterdir():
        if source.is_file() and source.suffix in {".pth", ".json", ".npz"}:
            target = remote / source.name
            if target.exists() and target.stat().st_size == source.stat().st_size and file_hash(source) == file_hash(target):
                continue
            temporary = remote / (source.name + ".uploading")
            shutil.copy2(source, temporary)
            if file_hash(source) != file_hash(temporary):
                raise OSError(f"Drive copy hash mismatch: {temporary}")
            temporary.replace(target)


def restore(folder, remote):
    if remote is None or not remote.exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    for source in remote.iterdir():
        if source.is_file() and source.suffix in {".pth", ".json", ".npz"} and not (folder / source.name).exists():
            temporary = folder / (source.name + ".restoring")
            shutil.copy2(source, temporary)
            if file_hash(source) != file_hash(temporary):
                raise OSError(f"Restore failed: {source}")
            temporary.replace(folder / source.name)


def code_fingerprint():
    files = sorted(p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts and not p.name.startswith("test_"))
    return hashlib.sha256("".join(f"{p.relative_to(ROOT).as_posix()}:{file_hash(p)}" for p in files).encode()).hexdigest()


def lock_protocol(args):
    value = {"protocol": protocol(), "protocol_sha256": fingerprint(), "code_sha256": code_fingerprint(),
             "manifest_sha256": file_hash(ROOT / "split_manifest.json"),
             "environment": {"python": sys.version, "platform": platform.platform(), "cuda": torch.version.cuda,
                             "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                             "packages": {n: version(n) for n in ("torch", "torchvision", "timm", "numpy", "scikit-learn")}}}
    target = args.output / "protocol.json"
    restore(args.output, args.drive)
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        for key in ("protocol_sha256", "code_sha256", "manifest_sha256"):
            if old[key] != value[key]:
                raise ValueError(f"{key} differs from the locked run. Use a new RUN_ID; do not mix results.")
        if old["environment"]["packages"] != value["environment"]["packages"]:
            raise ValueError("Package versions changed; restore the locked versions before resuming.")
    else:
        write_json(target, value)
    synchronize(args.output, args.drive)


def load_best(folder, name="best.pth"):
    complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    if file_hash(folder / name) != complete["sha256"][name]:
        raise ValueError(f"Checkpoint hash mismatch: {folder / name}")
    return torch.load(folder / name, map_location="cpu", weights_only=False)


@torch.no_grad()
def logits_of(model, loader):
    model.eval()
    z, y, positions = [], [], []
    for x, labels, pos in loader:
        out = model(x.to(DEVICE))
        z.append(out.float().cpu())
        y.append(labels)
        positions.append(pos)
    return torch.cat(z), torch.cat(y), torch.cat(positions)


def teacher_logits(args, seed, bundle):
    """Frozen eval-mode T1/T2 logits for every image (exactly equal to on-the-fly logits: no augmentation)."""
    target = args.output / f"s{seed}" / "teacher_logits.npz"
    restore(target.parent, args.drive / f"s{seed}" if args.drive else None)
    hashes = {stage: json.loads((args.output / f"s{seed}" / stage / "complete.json").read_text())["sha256"]["best.pth"]
              for stage in ("teacher1", "teacher2")}
    if target.exists():
        cached = np.load(target, allow_pickle=False)
        if str(cached["teacher1_sha256"]) == hashes["teacher1"] and str(cached["teacher2_sha256"]) == hashes["teacher2"]:
            return torch.from_numpy(cached["z1"]), torch.from_numpy(cached["z2"])
        raise ValueError(f"Stale teacher logit cache: {target}")
    n = len(bundle[0])
    loader = DataLoader(Indexed(bundle[1], bundle[2], np.arange(n)), batch_size=64, shuffle=False, num_workers=0)
    z_all = {}
    for stage in ("teacher1", "teacher2"):
        model = build_model(Job(seed, stage), pretrained=False).to(DEVICE)
        model.load_state_dict(load_best(args.output / f"s{seed}" / stage)["model"], strict=True)
        full = torch.full((n, 5), float("nan"))
        z, _, positions = logits_of(model, loader)
        full[positions] = z
        if torch.isnan(full).any():
            raise ValueError("Teacher logits missing for some images.")
        z_all[stage] = full
        del model
        torch.cuda.empty_cache()
    np.savez_compressed(target, z1=z_all["teacher1"].numpy(), z2=z_all["teacher2"].numpy(),
                        teacher1_sha256=hashes["teacher1"], teacher2_sha256=hashes["teacher2"])
    synchronize(target.parent, args.drive / f"s{seed}" if args.drive else None)
    return z_all["teacher1"], z_all["teacher2"]


def qwk(y, z):
    return compute_full_metrics(y, z.argmax(1).numpy(), z.softmax(1).numpy())["qwk"]


# ---------------------------------------------------------------- training
def train(job, args, bundle):
    folder = args.output / job.key
    remote = args.drive / job.key if args.drive else None
    restore(folder, remote)
    if (folder / "complete.json").exists():
        print("SKIP", job.key, flush=True)
        return False
    if job.stage in {"teacher_gate", "student"}:
        for stage in ("teacher1", "teacher2"):
            if not (args.output / f"s{job.seed}" / stage / "complete.json").exists():
                raise FileNotFoundError(f"Dependency not complete: s{job.seed}/{stage}")
    folder.mkdir(parents=True, exist_ok=True)
    hyper = HYPER["student" if job.stage == "student" else job.stage]
    set_seed(job.seed)
    data, weights = make_loaders(bundle, job.seed, hyper["batch"])
    weights = weights.to(DEVICE)
    z1 = z2 = z_teacher = None
    if job.stage in {"teacher_gate", "student"}:
        z1, z2 = teacher_logits(args, job.seed, bundle)
        z_teacher = ((z1 + z2) / 2).to(DEVICE)
        z1, z2 = z1.to(DEVICE), z2.to(DEVICE)
    config = {**asdict(job), "protocol_sha256": fingerprint(), "hyper": hyper, "class_weights": weights.cpu().tolist(),
              "loss": loss_config(job.arm) if job.stage == "student" else "class-weighted CE"}
    if (folder / "config.json").exists() and json.loads((folder / "config.json").read_text()) != config:
        raise ValueError(f"Configuration mismatch: {job.key}")
    write_json(folder / "config.json", config)
    checkpoint = torch.load(folder / "last.pth", map_location="cpu", weights_only=False) if (folder / "last.pth").exists() else None
    model = build_model(job, pretrained=checkpoint is None).to(DEVICE)
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyper["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    start, best, best_student, wait, history = 0, -float("inf"), -float("inf"), 0, []
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best, best_student = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["best_student"]
        wait, history = checkpoint["wait"], checkpoint["history"]
        random.setstate(checkpoint["rng_python"])
        np.random.set_state(checkpoint["rng_numpy"])
        torch.set_rng_state(checkpoint["rng_torch"].cpu())
        if torch.cuda.is_available() and checkpoint["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint["rng_cuda"]])
        data["train"].generator.set_state(checkpoint["rng_loader"])
    del checkpoint
    selection = loss_config(job.arm)["selection"] if job.stage == "student" else "own"
    for epoch in range(start, hyper["epochs"]):
        if wait >= hyper["patience"]:
            break
        started = time.monotonic()
        model.train()
        losses, components = [], {}
        for x, y, pos in data["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            if job.stage == "teacher_gate":
                z, _ = model(z1[pos.to(DEVICE)], z2[pos.to(DEVICE)])
                loss = F.cross_entropy(z, y, weight=weights, reduction="none").mean()
            elif job.stage == "student":
                loss, parts = student_loss(model(x), z_teacher[pos.to(DEVICE)], y, weights, job.arm, epoch)
                for k, v in parts.items():
                    components.setdefault(k, []).append(v)
            else:
                loss = F.cross_entropy(model(x), y, weight=weights, reduction="none").mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in {job.key}, epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            if job.stage == "teacher_gate":
                idx = torch.as_tensor(bundle[3]["val"], device=DEVICE)
                z_val, gate = model(z1[idx], z2[idx])
                y_val = torch.as_tensor(bundle[2][bundle[3]["val"]])
                scores = {"own": qwk(y_val, z_val.cpu()), "gate_mean": float(gate.mean())}
            else:
                z_val, y_val, pos_val = logits_of(model, data["val"])
                scores = {"own": qwk(y_val, z_val)}
                if job.stage == "student":
                    scores["student"] = scores.pop("own")
                    scores["system"] = qwk(y_val, (z_teacher[pos_val.to(DEVICE)].cpu() + z_val) / 2)
        current = scores[selection]
        if not np.isfinite(current):
            raise FloatingPointError("Validation QWK is nonfinite.")
        if current > best:
            best, wait = current, 0
            atomic_save(folder / "best.pth", {"model": model.state_dict(), "epoch": epoch + 1, "val_qwk": best})
        else:
            wait += 1
        if job.stage == "student" and scores["student"] > best_student:
            best_student = scores["student"]
            atomic_save(folder / "best_student.pth", {"model": model.state_dict(), "epoch": epoch + 1,
                                                      "val_qwk_student": best_student})
        scheduler.step()
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_qwk": scores,
                        "components": {k: float(np.mean(v)) for k, v in components.items()},
                        "lr": optimizer.param_groups[0]["lr"], "seconds": time.monotonic() - started})
        atomic_save(folder / "last.pth", {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "best": best, "best_student": best_student, "wait": wait, "history": history,
            "rng_python": random.getstate(), "rng_numpy": np.random.get_state(), "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rng_loader": data["train"].generator.get_state(),
        })
        write_json(folder / "history.json", history)
        synchronize(folder, remote)
        print(job.key, history[-1], flush=True)
    names = ["best.pth"] + (["best_student.pth"] if job.stage == "student" else [])
    write_json(folder / "complete.json", {"sha256": {n: file_hash(folder / n) for n in names}, "best_val_qwk": best,
                                          "selection": selection, "epochs_completed": len(history)})
    (folder / "last.pth").unlink(missing_ok=True)
    if remote is not None:
        (remote / "last.pth").unlink(missing_ok=True)
    synchronize(folder, remote)
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


# -------------------------------------------------------------- evaluation
def evaluate(args, bundle):
    missing = [j.key for j in jobs() if not (args.output / j.key / "complete.json").exists()]
    if missing:
        raise RuntimeError(f"Finish all training before test evaluation. Missing: {missing}")
    ids, _, labels, splits = bundle
    results = []
    for seed in SEEDS:
        data, _ = make_loaders(bundle, seed)
        z1, z2 = teacher_logits(args, seed, bundle)
        gate = EntropyGate()
        gate.load_state_dict(load_best(args.output / f"s{seed}" / "teacher_gate")["model"])
        gate.eval()
        for split in ("val", "test"):
            idx = torch.as_tensor(splits[split])
            y = labels[splits[split]]
            with torch.no_grad():
                z_gated, g = gate(z1[idx], z2[idx])
            z_t = (z1[idx] + z2[idx]) / 2
            paths = {"teacher1": z1[idx], "teacher2": z2[idx], "teacher_equal": z_t, "teacher_gated": z_gated}
            for arm in STUDENT_ARMS:
                folder = args.output / f"s{seed}" / "student" / arm
                for name, suffix in (("best.pth", ""), ("best_student.pth", "@student_selected")):
                    model = ConvNeXtTinyStudent(pretrained=False).to(DEVICE)
                    model.load_state_dict(load_best(folder, name)["model"], strict=True)
                    z_s, y_check, positions = logits_of(model, data[split])
                    if not np.array_equal(positions.numpy(), splits[split]) or not np.array_equal(y_check.numpy(), y):
                        raise ValueError("Evaluation order mismatch.")
                    paths[f"{arm}/student{suffix}"] = z_s
                    paths[f"{arm}/system{suffix}"] = (z_t + z_s) / 2
                    del model
            np.savez_compressed(args.output / f"s{seed}" / f"{split}_logits.npz", ids=ids[splits[split]], labels=y,
                                gate=g.numpy(), **{k.replace("/", "__").replace("@", "_at_"): v.numpy() for k, v in paths.items()})
            for name, z in paths.items():
                p = z.double().softmax(1).numpy()
                m = compute_full_metrics(y, p.argmax(1), p)
                m["confusion_matrix"] = confusion_matrix(y, p.argmax(1), labels=list(range(5))).tolist()
                results.append({"seed": seed, "split": split, "path": name, **m})
            if split == "test":
                results.append({"seed": seed, "split": split, "path": "teacher_gate_stats",
                                "gate_mean": float(g.mean()), "gate_lt_0.2": float((g < 0.2).float().mean())})
        synchronize(args.output / f"s{seed}", args.drive / f"s{seed}" if args.drive else None)
    write_json(args.output / "evaluation.json", results)
    synchronize(args.output, args.drive)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["plan", "train", "evaluate"], required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--drive", type=Path)
    parser.add_argument("--limit", type=int, help="Maximum unfinished jobs in this invocation; rerun to resume.")
    args = parser.parse_args()
    if args.stage == "plan":
        print(json.dumps({"jobs": len(jobs()), "protocol_sha256": fingerprint(), **protocol()}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime.")
    bundle = load_data(args.data, ROOT / "split_manifest.json")
    args.output.mkdir(parents=True, exist_ok=True)
    lock_protocol(args)
    if args.stage == "train":
        done = 0
        for job in jobs():
            if args.limit is not None and done >= args.limit:
                print("Limit reached; rerun the same cell to continue.")
                break
            done += int(train(job, args, bundle))
        remaining = [j.key for j in jobs() if not (args.output / j.key / "complete.json").exists()]
        print("REMAINING JOBS:", len(remaining), remaining)
    else:
        evaluate(args, bundle)


if __name__ == "__main__":
    main()
