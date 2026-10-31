"""Round-2 reviewer experiments: sequential, epoch-resumable, Drive-persistent. Reuses unchanged round-1 code."""
import argparse
from dataclasses import asdict
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
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dr_pipeline as P
import run as runner
from data import file_hash, load_data, loaders
from fusion_baseline import FeatureFusionModel, historical_loaders
from plan2 import FAMILIES, ROUND1_RUN_ID, fingerprint, hyper, jobs, protocol


ROOT = Path(__file__).resolve().parent
# False only in CPU smoke/tests (no weight download); Colab training always uses ImageNet weights.
PRETRAINED = True
REFERENCE_FILES = ("best.pth", "complete.json", "config.json", "history.json", "metrics.json",
                   "val_logits.npz", "test_logits.npz")


def device():
    return runner.DEVICE


class SingleTeacher(nn.Module):
    """Independently trained Teacher 2 used alone; returns its raw logits like round-1 teacher_mode='t2'."""

    def __init__(self, adv_model):
        super().__init__()
        self.adv_model = adv_model
        self.fusion_mode = "t2"

    def forward(self, x):
        logits, extra = self.adv_model(x)
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=logits.dtype)
        return logits, {"gate": ones, "adv_logits": logits, "adv_extra": extra}


def advanced_config(spectral_norm=True):
    config = dict(P.get_advanced_model_config("Adv_Idea_5_SpectralNorm"))
    if not spectral_norm:
        config.update(name="Adv_Idea_5_NoSpectralNorm", use_spectral_norm=False)
    return config


def nosn_teacher_modules(seed, pretrained=True):
    P.set_seed(seed)
    base = P.HybridCNNViTBase(num_classes=5, config=P.BASELINE_CONFIG)
    advanced = P.AdvancedHybridModel(num_classes=5, config=advanced_config(False), pretrained=pretrained)
    return P.HeavyTeacherEnsemble(base, advanced, fusion_mode="gate")


def teacher_for_student(job, output):
    state = runner.load_best(output / job.dependency)["model"]
    if job.family == "t2indep":
        P.set_seed(job.seed)
        adv = P.AdvancedHybridModel(num_classes=5, config=advanced_config(True), pretrained=False)
        adv.load_state_dict(state, strict=True)
        return SingleTeacher(adv), "t2"
    teacher = (nosn_teacher_modules if job.family == "nosn" else runner.teacher_modules)(job.seed, pretrained=False)
    teacher.load_state_dict(state, strict=True)
    return teacher, "gate"


def build(job, output, checkpoint=None):
    fresh = checkpoint is None and PRETRAINED
    if job.family == "reference":
        return runner.build_model(job.legacy, output / "reference", checkpoint)
    P.set_seed(job.seed)
    if job.stage == "teacher1":
        model = P.HybridCNNViTBase(num_classes=5, config=P.BASELINE_CONFIG)
    elif job.stage == "teacher2_indep":
        model = P.AdvancedHybridModel(num_classes=5, config=advanced_config(True), pretrained=fresh)
    elif job.stage == "teacher2":
        factory = nosn_teacher_modules if job.family == "nosn" else runner.teacher_modules
        model = factory(job.seed, pretrained=fresh)
        model.baseline_model.load_state_dict(runner.load_best(output / job.dependency)["model"], strict=True)
    elif job.stage == "fusion":
        model = FeatureFusionModel(5, pretrained=fresh)
    elif job.arm == "student_ce":
        model = P.ConvNeXtTinyStudent(5, pretrained=fresh)
    else:
        teacher, mode = teacher_for_student(job, output)
        P.set_seed(job.seed)
        student = P.ConvNeXtTinyStudent(5, pretrained=fresh)
        model = P.ResidualTeacherStudentModel(teacher, student, mix="gated", teacher_mode=mode, **P.STUDENT_MODEL_KW)
    model.to(device())
    if checkpoint is not None:
        runner.restore_state(model, checkpoint["model"])
    return model


def job_loss(job, logits, extra, labels, weights, epoch):
    if job.stage == "teacher2_indep":
        return F.cross_entropy(logits, labels, weight=weights, reduction="none").mean()
    if job.stage == "fusion" and job.arm == "historical":
        return F.cross_entropy(logits, labels)
    return runner.compute_loss(logits, extra, labels, weights, job.legacy, epoch)


def job_loaders(job, bundle):
    if job.stage == "fusion" and job.arm == "historical":
        return historical_loaders(bundle, job.seed, hyper(job)["batch"])
    legacy = job.legacy
    view = SimpleNamespace(fraction=1.0, seed=job.seed, preprocessing=legacy.preprocessing,
                           arm="augmentation" if job.augment else legacy.arm)
    return loaders(bundle, view)


def scheduler_for(job, optimizer):
    if hyper(job)["scheduler"] == "cosine_annealing_tmax100":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)


def checkpoint_stage(job):
    return "student" if job.stage == "student" and job.arm != "student_ce" else "other"


def copy_verified(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".copying")
    shutil.copy2(source, temporary)
    if file_hash(source) != file_hash(temporary):
        raise OSError(f"Copy hash mismatch: {source}")
    temporary.replace(target)


def sync_files(folder, remote, names):
    if remote is None:
        return
    for name in names:
        source = folder / name
        if source.is_file():
            target = remote / name
            if not (target.exists() and file_hash(target) == file_hash(source)):
                copy_verified(source, target)


def code_fingerprint():
    files = sorted(ROOT.rglob("*.py"))
    return hashlib.sha256("".join(f"{p.relative_to(ROOT).as_posix()}:{file_hash(p)}" for p in files).encode()).hexdigest()


def lock_protocol(args):
    target = args.output / "protocol2.json"
    value = {
        "protocol": protocol(), "protocol_sha256": fingerprint(), "code_sha256": code_fingerprint(),
        "csv_sha256": file_hash(args.data / "train.csv"), "cache_sha256": file_hash(args.data / "cache_224.npz"),
        "manifest_sha256": file_hash(ROOT / "split_manifest.json"),
        "environment": {"python": sys.version, "platform": platform.platform(), "cuda": torch.version.cuda,
                        "packages": {n: version(n) for n in ("torch", "torchvision", "timm", "numpy", "pandas",
                                                             "scikit-learn", "Pillow", "matplotlib")},
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"},
    }
    remote = args.drive / "protocol2.json" if args.drive else None
    if not target.exists() and remote and remote.exists():
        copy_verified(remote, target)
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        for key in ("protocol_sha256", "code_sha256", "csv_sha256", "cache_sha256", "manifest_sha256"):
            if old[key] != value[key]:
                raise ValueError(f"Existing round-2 experiment {key} differs. Use a new RUN_ID; do not mix results.")
        if old["environment"]["packages"] != value["environment"]["packages"]:
            raise ValueError("Dependency versions changed. Restore the original versions before resuming.")
    else:
        runner.write_json(target, value)
    sync_files(args.output, args.drive, ["protocol2.json"])
    return value


def reference_import(args):
    """Copy round-1 reference checkpoints read-only (SHA-256 verified). Never writes to the round-1 folder."""
    manifest_path = args.output / "reference_manifest.json"
    if not manifest_path.exists() and args.drive and (args.drive / "reference_manifest.json").exists():
        copy_verified(args.drive / "reference_manifest.json", manifest_path)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for job in (j for j in jobs() if j.family == "reference"):
        folder = args.output / job.key
        if (folder / "complete.json").exists():
            continue
        source = args.round1 / job.legacy.key if args.round1 else None
        entry = {"round1_key": job.legacy.key, "source": str(source) if source else None}
        try:
            completed = json.loads((source / "complete.json").read_text())
            if file_hash(source / "best.pth") != completed["best_sha256"]:
                raise ValueError("round-1 best.pth hash differs from complete.json")
        except (OSError, ValueError, KeyError, TypeError) as error:
            if manifest.get(job.key, {}).get("mode") != "retrained":
                manifest[job.key] = {**entry, "mode": "unavailable_retrain_required", "reason": str(error)}
            continue
        for name in REFERENCE_FILES:
            if (source / name).is_file():
                copy_verified(source / name, folder / name)
        manifest[job.key] = {**entry, "mode": "imported", "best_sha256": completed["best_sha256"],
                             "files": sorted(p.name for p in folder.iterdir())}
    runner.write_json(manifest_path, manifest)
    sync_files(args.output, args.drive, ["reference_manifest.json"])
    return manifest


def round1_inventory(args):
    from plan import matrix
    report = {"round1_root": str(args.round1), "runs": {}}
    for run in matrix():
        folder = args.round1 / run.key if args.round1 else None
        report["runs"][run.key] = {name: bool(folder and (folder / name).is_file())
                                   for name in ("complete.json", "best.pth", "val_logits.npz", "test_logits.npz")}
    values = list(report["runs"].values())
    report["counts"] = {name: sum(v[name] for v in values) for name in values[0]}
    runner.write_json(args.output / "round1_inventory.json", report)
    sync_files(args.output, args.drive, ["round1_inventory.json"])
    return report


def train_reference(job, args, bundle):
    view = SimpleNamespace(output=args.output / "reference", drive=args.drive / "reference" if args.drive else None)
    trained = runner.train(job.legacy, view, bundle)
    manifest_path = args.output / "reference_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if trained or manifest.get(job.key, {}).get("mode") != "imported":
        completed = json.loads((args.output / job.key / "complete.json").read_text())
        manifest[job.key] = {"round1_key": job.legacy.key, "mode": "retrained",
                             "code": "unchanged round-1 run.train", "best_sha256": completed["best_sha256"]}
        runner.write_json(manifest_path, manifest)
        sync_files(args.output, args.drive, ["reference_manifest.json"])
    return trained


def train_job(job, args, bundle):
    if job.family == "reference":
        return train_reference(job, args, bundle)
    folder = args.output / job.key
    remote = args.drive / job.key if args.drive else None
    folder.mkdir(parents=True, exist_ok=True)
    if remote and remote.exists():
        for source in remote.iterdir():
            if source.is_file() and source.suffix in {".pth", ".json", ".npz"} and not (folder / source.name).exists():
                copy_verified(source, folder / source.name)
    if (folder / "complete.json").exists():
        completed = json.loads((folder / "complete.json").read_text())
        if file_hash(folder / "best.pth") != completed["best_sha256"]:
            raise ValueError(f"Completed run checkpoint hash mismatch: {job.key}")
        print("SKIP", job.key, flush=True)
        return False
    if job.dependency and not (args.output / job.dependency / "complete.json").exists():
        raise FileNotFoundError(f"Dependency not complete: {job.dependency}")
    P.set_seed(job.seed)
    data, weights, train_ids = job_loaders(job, bundle)
    weights = weights.to(device())
    configuration = {
        **asdict(job), "key": job.key, "protocol_sha256": fingerprint(), "hyper": hyper(job),
        "dependency": job.dependency,
        "dependency_sha256": file_hash(args.output / job.dependency / "best.pth") if job.dependency else None,
        "train_ids": train_ids, "class_weights": weights.cpu().tolist(), "training_size": len(train_ids),
        "augmentation": job.augment, "preprocessing": job.preprocessing,
    }
    config_path = folder / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError(f"Run configuration mismatch: {job.key}")
    runner.write_json(config_path, configuration)
    checkpoint = torch.load(folder / "last.pth", map_location="cpu", weights_only=False) if (folder / "last.pth").exists() else None
    model = build(job, args.output, checkpoint)
    settings = hyper(job)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=settings["lr"], weight_decay=1e-4)
    scheduler = scheduler_for(job, optimizer)
    start, best, wait, history = 0, -float("inf"), 0, []
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best, wait, history = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["wait"], checkpoint["history"]
        random.setstate(checkpoint["rng_python"])
        np.random.set_state(checkpoint["rng_numpy"])
        torch.set_rng_state(checkpoint["rng_torch"].cpu())
        if torch.cuda.is_available() and checkpoint["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint["rng_cuda"]])
    del checkpoint
    stage = checkpoint_stage(job)
    for epoch in range(start, settings["epochs"]):
        if wait >= settings["patience"]:
            break
        started = time.monotonic()
        model.train()
        losses = []
        for x, y in data["train"]:
            x, y = x.to(device()), y.to(device())
            optimizer.zero_grad(set_to_none=True)
            paths, extra = runner.forward_paths(model, x, job.legacy)
            loss = job_loss(job, paths["primary"], extra, y, weights, epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in {job.key}, epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        y, z = runner.predict(model, data["val"], job.legacy)
        qwk = P.compute_full_metrics(y, z["primary"].argmax(1), torch.tensor(z["primary"]).softmax(1).numpy())["qwk"]
        if not np.isfinite(qwk):
            raise FloatingPointError("Validation QWK is nonfinite.")
        if qwk > best:
            best, wait = qwk, 0
            runner.atomic_save(folder / "best.pth", {"model": runner.model_state(model, stage), "epoch": epoch, "qwk": best})
        else:
            wait += 1
        scheduler.step()
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_qwk": float(qwk),
                        "seconds": time.monotonic() - started})
        runner.atomic_save(folder / "last.pth", {
            "model": runner.model_state(model, stage), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best, "wait": wait, "history": history,
            "rng_python": random.getstate(), "rng_numpy": np.random.get_state(), "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        })
        runner.write_json(folder / "history.json", history)
        runner.synchronize(folder, remote)
        print(job.key, history[-1], "SAVED", flush=True)
    runner.write_json(folder / "complete.json", {"best_sha256": file_hash(folder / "best.pth"), "best_val_qwk": best,
                                               "epochs_completed": len(history), "test_evaluated": False})
    runner.synchronize(folder, remote)
    if not args.keep_last:
        for path in (folder / "last.pth", remote / "last.pth" if remote else None):
            if path is not None and path.exists():
                path.unlink()
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def selected(families):
    chosen = set(families) | {"reference"}
    return [job for job in jobs() if job.family in chosen]


def evaluate(args, bundle):
    todo = selected(args.families)
    missing = [job.key for job in todo if not (args.output / job.key / "complete.json").exists()]
    if missing:
        raise RuntimeError(f"Finish training of the requested families before test evaluation. Missing: {missing}")
    for job in todo:
        folder = args.output / job.key
        best_sha = json.loads((folder / "complete.json").read_text())["best_sha256"]
        marker = folder / "round2_evaluated.json"
        if marker.exists() and json.loads(marker.read_text()).get("best_sha256") == best_sha:
            continue
        if job.family == "reference":
            data = loaders(bundle, job.legacy)[0]
        else:
            data = job_loaders(job, bundle)[0]
        checkpoint = runner.load_best(folder)
        model = build(job, args.output, checkpoint).eval()
        del checkpoint
        report = {}
        for split in ("val", "test"):
            y, paths = runner.predict(model, data[split], job.legacy)
            ids = np.asarray(bundle[0][bundle[3][split]], dtype=str)
            np.savez_compressed(folder / f"round2_{split}_logits.npz", labels=y, ids=ids, **paths)
            report[split] = {name: P.compute_full_metrics(y, z.argmax(1), torch.tensor(z).softmax(1).numpy())
                             for name, z in paths.items()}
            imported = folder / f"{split}_logits.npz"
            if job.family == "reference" and imported.exists():
                with np.load(imported) as old:
                    report.setdefault("reproduction", {})[split] = {
                        name: {"max_abs_logit_difference": float(np.abs(old[name] - z).max()),
                               "argmax_agreement": float((old[name].argmax(1) == z.argmax(1)).mean())}
                        for name, z in paths.items() if name in old.files}
        runner.write_json(folder / "round2_metrics.json", report)
        runner.write_json(marker, {"best_sha256": best_sha})
        names = ["round2_val_logits.npz", "round2_test_logits.npz", "round2_metrics.json", "round2_evaluated.json"]
        sync_files(folder, args.drive / job.key if args.drive else None, names)
        print("EVALUATED", job.key, flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preflight", "inventory", "train", "evaluate"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drive", type=Path)
    parser.add_argument("--round1", type=Path, help=f"Drive folder of round-1 run '{ROUND1_RUN_ID}' (read-only).")
    parser.add_argument("--families", nargs="+", choices=[f for f in FAMILIES if f != "reference"],
                        default=[f for f in FAMILIES if f != "reference"])
    parser.add_argument("--limit", type=int, help="Maximum unfinished jobs this invocation; resume later.")
    parser.add_argument("--keep-last", action="store_true", help="Keep last.pth after completion.")
    args = parser.parse_args(argv)
    if args.stage in {"train", "evaluate"} and not torch.cuda.is_available():
        raise RuntimeError("Training/evaluation requires a Colab GPU; CPU is reserved for tests/preflight.")
    bundle = load_data(args.data, ROOT / "split_manifest.json")
    args.output.mkdir(parents=True, exist_ok=True)
    lock_protocol(args)
    manifest = reference_import(args)
    inventory = round1_inventory(args)
    print("ROUND-1 INVENTORY", inventory["counts"], flush=True)
    print("REFERENCES", {k: v["mode"] for k, v in manifest.items()}, flush=True)
    print("PREFLIGHT OK", len(selected(args.families)), "jobs incl. references; split 2928/367/367", flush=True)
    if args.stage == "train":
        count = 0
        for job in selected(args.families):
            if train_job(job, args, bundle):
                count += 1
            if args.limit and count >= args.limit:
                break
        done = sum((args.output / j.key / "complete.json").exists() for j in selected(args.families))
        print(f"COMPLETE {done}/{len(selected(args.families))}", flush=True)
    elif args.stage == "evaluate":
        evaluate(args, bundle)


if __name__ == "__main__":
    main()
