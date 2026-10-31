"""Sequential, epoch-resumable reviewer experiments. Tests are a separate stage."""
import argparse
from dataclasses import asdict
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import dr_pipeline as P
from data import file_hash, load_data, loaders
from plan import fingerprint, loss_config, matrix, protocol


ROOT = Path(__file__).resolve().parent
DEVICE = P.device


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_save(path, value):
    def tensor_bytes(item):
        if torch.is_tensor(item):
            return item.numel() * item.element_size()
        if isinstance(item, dict):
            return sum(tensor_bytes(v) for v in item.values())
        if isinstance(item, (list, tuple)):
            return sum(tensor_bytes(v) for v in item)
        return 0
    if shutil.disk_usage(path.parent).free < tensor_bytes(value) * 1.1 + 1024**3:
        raise OSError(f"Insufficient local disk for atomic checkpoint: {path}. Last completed epoch remains intact.")
    temporary = path.with_suffix(".pth.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def code_fingerprint():
    files = sorted(ROOT.rglob("*.py"))
    return hashlib.sha256("".join(f"{p.relative_to(ROOT).as_posix()}:{file_hash(p)}" for p in files).encode()).hexdigest()


def synchronize(folder, remote):
    if remote is None:
        return
    remote.mkdir(parents=True, exist_ok=True)
    for source in folder.iterdir():
        if source.is_file() and source.suffix in {".pth", ".json", ".npz"}:
            target = remote / source.name
            if target.exists() and file_hash(source) == file_hash(target):
                continue
            if shutil.disk_usage(remote).free < source.stat().st_size * 1.1:
                raise OSError(f"Insufficient Drive space to persist {source.name}; training stopped.")
            temporary = target.with_suffix(target.suffix + ".uploading")
            shutil.copy2(source, temporary)
            if file_hash(source) != file_hash(temporary):
                raise OSError(f"Drive copy hash mismatch: {temporary}")
            temporary.replace(target)


def lock_protocol(args):
    target = args.output / "protocol.json"
    value = {
        "protocol": protocol(), "protocol_sha256": fingerprint(), "code_sha256": code_fingerprint(),
        "csv_sha256": file_hash(args.data / "train.csv"),
        "cache_sha256": file_hash(args.data / "cache_224.npz"),
        "manifest_sha256": file_hash(ROOT / "split_manifest.json"),
        "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
                        "packages": {name: version(name) for name in ("torch", "torchvision", "timm", "numpy", "pandas", "scikit-learn", "Pillow", "matplotlib")},
                        "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"},
    }
    remote = args.drive / "protocol.json" if args.drive else None
    if not target.exists() and remote and remote.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote, target)
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        for key in ("protocol_sha256", "code_sha256", "csv_sha256", "cache_sha256", "manifest_sha256"):
            if old[key] != value[key]:
                raise ValueError(f"Existing experiment {key} differs. Use a new run ID; do not mix results.")
        if old["environment"]["packages"] != value["environment"]["packages"]:
            raise ValueError("Dependency versions changed. Restore the original versions before resuming this run.")
    else:
        write_json(target, value)
    synchronize(args.output, args.drive)
    return value


def load_best(folder):
    completed = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    if file_hash(folder / "best.pth") != completed["best_sha256"]:
        raise ValueError(f"Best checkpoint integrity check failed: {folder}")
    return torch.load(folder / "best.pth", map_location="cpu", weights_only=False)


def teacher_modules(seed, pretrained=True):
    P.set_seed(seed)
    base = P.HybridCNNViTBase(num_classes=5, config=P.BASELINE_CONFIG)
    advanced = P.AdvancedHybridModel(num_classes=5, config=P.get_advanced_model_config("Adv_Idea_5_SpectralNorm"), pretrained=pretrained)
    return P.HeavyTeacherEnsemble(base, advanced, fusion_mode="gate")


def restore_state(model, state):
    complete = model.state_dict()
    if set(state) - set(complete):
        raise ValueError("Unexpected checkpoint state keys.")
    allowed_missing = {name for name, _ in model.named_parameters() if name.startswith("teacher_model.")}
    if set(complete) - set(state) - allowed_missing:
        raise ValueError("Checkpoint is missing model parameters or buffers.")
    complete.update(state)
    model.load_state_dict(complete, strict=True)


def model_state(model, stage):
    if stage != "student":
        return model.state_dict()
    parameters = {name for name, _ in model.named_parameters() if name.startswith("teacher_model.")}
    return {key: value for key, value in model.state_dict().items() if key not in parameters}


def build_model(run, output, checkpoint=None):
    P.set_seed(run.seed)
    if run.stage == "teacher1":
        model = P.HybridCNNViTBase(num_classes=5, config=P.BASELINE_CONFIG)
    elif run.stage == "teacher2":
        model = teacher_modules(run.seed, pretrained=checkpoint is None)
        baseline = load_best(output / run.dependency)
        model.baseline_model.load_state_dict(baseline["model"], strict=True)
    elif run.arm == "student_ce":
        model = P.ConvNeXtTinyStudent(5, pretrained=checkpoint is None)
    else:
        teacher = teacher_modules(run.seed, pretrained=False)
        state = load_best(output / run.dependency)
        teacher.load_state_dict(state["model"], strict=True)
        # The same student/gate initialization is reused across matched objective arms.
        P.set_seed(run.seed)
        student = P.ConvNeXtTinyStudent(5, pretrained=checkpoint is None)
        mode = "t2" if run.arm == "teacher2_only" else "gate"
        model = P.ResidualTeacherStudentModel(teacher, student, mix="gated", teacher_mode=mode, **P.STUDENT_MODEL_KW)
    model.to(DEVICE)
    if checkpoint is not None:
        restore_state(model, checkpoint["model"])
    return model


def compute_loss(logits, extra, labels, weights, run, epoch):
    ce = F.cross_entropy(logits, labels, weight=weights, reduction="none")
    if run.stage == "teacher1" or run.arm == "student_ce":
        # Preserve historical teacher-1 weighted-mean reduction; student uses arithmetic sample mean.
        return ce.sum() / weights[labels].sum() if run.stage == "teacher1" else ce.mean()
    if run.stage == "teacher2":
        progress = epoch / 119
        lam, margin = 1.5 + progress * (0.6 - 1.5), 0.02 + progress * (0.005 - 0.02)
        reference = F.cross_entropy(extra["base_logits"].detach(), labels, weight=weights, reduction="none")
        gate = extra["gate"].clamp(1e-6, 1 - 1e-6)
        entropy = -(gate * gate.log() + (1 - gate) * (1 - gate).log()).mean()
        return ce.mean() + lam * F.relu(ce - reference + margin).mean() - 0.02 * entropy
    config = loss_config(run.arm)
    loss = ce.mean()
    reference_logits = extra["z_teacher"].detach()
    if config["kd"]:
        for temperature, weight in zip(config["temperatures"], config["temperature_weights"]):
            loss = loss + config["kd_coefficient"] * weight * temperature ** 2 * F.kl_div(
                F.log_softmax(logits / temperature, dim=1),
                F.softmax(reference_logits / temperature, dim=1), reduction="batchmean",
            )
    if config["ordinal"]:
        target = F.one_hot(labels, 5).float().cumsum(1)
        loss = loss + config["ordinal_coefficient"] * F.mse_loss(logits.softmax(1).cumsum(1), target)
    if config["hinge"]:
        progress = epoch / 69
        lam, margin = 0.8 + progress * (0.25 - 0.8), 0.010 + progress * (0.003 - 0.010)
        reference = F.cross_entropy(reference_logits, labels, weight=weights, reduction="none")
        loss = loss + lam * F.relu(ce - reference + margin).mean()
    if config["entropy_coefficient"]:
        gate = extra["gate"].clamp(1e-6, 1 - 1e-6)
        entropy = -(gate * gate.log() + (1 - gate) * (1 - gate).log()).mean()
        loss = loss - config["entropy_coefficient"] * entropy
    return loss


def forward_paths(model, x, run):
    output = model(x)
    logits, extra = output if isinstance(output, tuple) else (output, {})
    if run.stage != "student" or run.arm == "student_ce":
        return {"primary": logits}, extra
    return {
        "primary": logits, "student": extra["z_student"], "teacher": extra["z_teacher"],
        "equal": (extra["z_teacher"] + extra["z_student"]) / 2,
    }, extra


@torch.no_grad()
def predict(model, loader, run):
    model.eval()
    values, labels = {}, []
    for x, y in loader:
        paths, extra = forward_paths(model, x.to(DEVICE), run)
        if run.stage == "teacher2":
            paths.update({"teacher1": extra["base_logits"], "teacher2": extra["adv_logits"],
                          "teacher_raw_average": (extra["base_logits"] + extra["adv_logits"]) / 2,
                          "teacher_probability_average": ((extra["base_logits"].softmax(1)
                                                          + extra["adv_logits"].softmax(1)) / 2).clamp_min(1e-12).log()})
        labels.append(y.numpy())
        for name, z in paths.items():
            values.setdefault(name, []).append(z.float().cpu().numpy())
    return np.concatenate(labels), {k: np.concatenate(v) for k, v in values.items()}


def train(run, args, bundle):
    folder = args.output / run.key
    remote = args.drive / run.key if args.drive else None
    folder.mkdir(parents=True, exist_ok=True)
    if remote and remote.exists():
        for source in remote.iterdir():
            if source.is_file() and source.suffix in {".pth", ".json", ".npz"} and not (folder / source.name).exists():
                temporary = folder / (source.name + ".restoring")
                shutil.copy2(source, temporary)
                if file_hash(source) != file_hash(temporary):
                    raise OSError(f"Restore failed: {source}")
                temporary.replace(folder / source.name)
    if (folder / "complete.json").exists():
        completed = json.loads((folder / "complete.json").read_text())
        if file_hash(folder / "best.pth") != completed["best_sha256"]:
            raise ValueError("Completed run checkpoint hash mismatch.")
        print("SKIP", run.key, flush=True)
        return False
    if run.dependency and not (args.output / run.dependency / "complete.json").exists():
        raise FileNotFoundError(f"Dependency not complete: {run.dependency}")
    P.set_seed(run.seed)
    data, weights, train_ids = loaders(bundle, run)
    weights = weights.to(DEVICE)
    dependency_hash = file_hash(args.output / run.dependency / "best.pth") if run.dependency else None
    configuration = {
        **asdict(run), "protocol_sha256": fingerprint(), "dependency_sha256": dependency_hash,
        "train_ids": train_ids, "class_weights": weights.cpu().tolist(),
        "training_size": len(train_ids), "training_class_counts": np.bincount(
            bundle[2][np.isin(bundle[0], train_ids)], minlength=5).tolist(),
        "loss": loss_config(run.arm) if run.stage == "student" else run.stage,
        "selection_output": "standalone student" if run.arm == "student_ce" else "primary training output",
    }
    config_path = folder / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError(f"Run configuration mismatch: {run.key}")
    write_json(config_path, configuration)
    checkpoint = torch.load(folder / "last.pth", map_location="cpu", weights_only=False) if (folder / "last.pth").exists() else None
    model = build_model(run, args.output, checkpoint)
    hyper = protocol()[run.stage if run.stage != "student" else "student"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=hyper["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    start, best, wait, history = 0, -float("inf"), 0, []
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best, wait, history = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["wait"], checkpoint["history"]
        random.setstate(checkpoint["rng_python"])
        np.random.set_state(checkpoint["rng_numpy"])
        torch.set_rng_state(checkpoint["rng_torch"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint["rng_cuda"]])
    del checkpoint
    for epoch in range(start, hyper["epochs"]):
        if wait >= hyper["patience"]:
            break
        started = time.monotonic()
        model.train()
        if run.stage == "student" and run.arm == "teacher_eval":
            model.teacher_model.eval()
        losses = []
        for x, y in data["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            paths, extra = forward_paths(model, x, run)
            loss = compute_loss(paths["primary"], extra, y, weights, run, epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in {run.key}, epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        y, z = predict(model, data["val"], run)
        qwk = P.compute_full_metrics(y, z["primary"].argmax(1), torch.tensor(z["primary"]).softmax(1).numpy())["qwk"]
        if not np.isfinite(qwk):
            raise FloatingPointError("Validation QWK is nonfinite.")
        improved = qwk > best
        if improved:
            best, wait = qwk, 0
            atomic_save(folder / "best.pth", {"model": model_state(model, run.stage), "epoch": epoch, "qwk": best})
        else:
            wait += 1
        scheduler.step()
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_qwk": float(qwk),
                        "seconds": time.monotonic() - started})
        atomic_save(folder / "last.pth", {
            "model": model_state(model, run.stage), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "best": best, "wait": wait, "history": history,
            "rng_python": random.getstate(), "rng_numpy": np.random.get_state(), "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        })
        write_json(folder / "history.json", history)
        synchronize(folder, remote)
        print(run.key, history[-1], "SAVED", flush=True)
    write_json(folder / "complete.json", {"best_sha256": file_hash(folder / "best.pth"), "best_val_qwk": best,
                                        "epochs_completed": len(history), "test_evaluated": False})
    synchronize(folder, remote)
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def evaluate(args, bundle):
    missing = [r.key for r in matrix() if not (args.output / r.key / "complete.json").exists()]
    if missing:
        raise RuntimeError(f"Finish the frozen training matrix before test evaluation. Missing {len(missing)} jobs.")
    for run in matrix():
        folder = args.output / run.key
        data, _, _ = loaders(bundle, run)
        checkpoint = load_best(folder)
        model = build_model(run, args.output, checkpoint)
        del checkpoint
        metrics = {}
        for split in ("val", "test"):
            y, paths = predict(model, data[split], run)
            payload = {"labels": y, "ids": np.asarray(bundle[0][bundle[3][split]], dtype=str), **paths}
            np.savez_compressed(folder / f"{split}_logits.npz", **payload)
            metrics[split] = {name: P.compute_full_metrics(y, z.argmax(1), torch.tensor(z).softmax(1).numpy())
                              for name, z in paths.items()}
        write_json(folder / "metrics.json", metrics)
        synchronize(folder, args.drive / run.key if args.drive else None)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preflight", "train", "evaluate"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drive", type=Path)
    parser.add_argument("--limit", type=int, help="Maximum unfinished jobs this invocation; resume later.")
    args = parser.parse_args()
    if args.stage != "preflight" and not torch.cuda.is_available():
        raise RuntimeError("Training/evaluation requires a Colab GPU; CPU is reserved for tests/preflight.")
    bundle = load_data(args.data, ROOT / "split_manifest.json")
    args.output.mkdir(parents=True, exist_ok=True)
    lock_protocol(args)
    budget = {"training_jobs": len(matrix()), "recommended_storage_gib": 85,
              "output_free_gib": shutil.disk_usage(args.output).free / 2**30,
              "drive_reported_free_gib": shutil.disk_usage(args.drive).free / 2**30 if args.drive else None,
              "warning": "Drive FUSE free space may not reflect account quota. Check Drive account storage manually.",
              "persistent_files": "best + optimizer/RNG last checkpoint per run; approximately 60-75 GiB total"}
    write_json(args.output / "storage.json", budget)
    synchronize(args.output, args.drive)
    print("PREFLIGHT OK", len(matrix()), "jobs; fixed split 2928/367/367", budget, flush=True)
    if args.stage == "train":
        count = 0
        for run in matrix():
            if train(run, args, bundle):
                count += 1
            if args.limit and count >= args.limit:
                break
    elif args.stage == "evaluate":
        evaluate(args, bundle)


if __name__ == "__main__":
    main()
