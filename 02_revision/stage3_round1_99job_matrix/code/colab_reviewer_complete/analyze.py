"""Validation-only calibration, paired intervals, figures and matched GPU benchmarks."""
import argparse
from copy import deepcopy
import gc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score
import torch
import torch.nn.functional as F

from data import loaders, load_data
from plan import Run, SEEDS, matrix
from run import DEVICE, ROOT, build_model, file_hash, load_best, synchronize, write_json


def probabilities(logits):
    return torch.as_tensor(logits, dtype=torch.float64).softmax(1).numpy()


def metrics(y, p):
    if p.shape != (len(y), 5) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1):
        raise ValueError("Invalid probabilities.")
    pred, confidence = p.argmax(1), p.max(1)
    correct = pred == y
    bins = np.minimum((confidence * 15).astype(int), 14)
    ece = sum(float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
              for i in range(15) if (mask := bins == i).any())
    cm = confusion_matrix(y, pred, labels=range(5))
    referral_true, referral_pred = y >= 2, p[:, 2:].sum(1) >= .5
    result = {
        "accuracy": float(correct.mean()), "macro_f1": float(f1_score(y, pred, labels=range(5), average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y, pred, labels=range(5), weights="quadratic")),
        "weighted_ovr_auc": float(roc_auc_score(y, p, multi_class="ovr", average="weighted")),
        "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "brier": float(((p - np.eye(5)[y]) ** 2).sum(1).mean()), "ece15": float(ece),
        "confusion_matrix": cm.tolist(),
        "per_class": {str(i): {"support": int(cm[i].sum()), "recall": float(cm[i, i] / cm[i].sum()),
                               "precision": float(cm[i, i] / cm[:, i].sum()) if cm[:, i].sum() else 0.0}
                      for i in range(5)},
        "referable": {
            "rule": "grades >=2; sum of probabilities for grades 2-4 >=0.5; no threshold fitting",
            "sensitivity": float(referral_pred[referral_true].mean()),
            "specificity": float((~referral_pred[~referral_true]).mean()),
        },
    }
    return result


def fit_temperature(logits, labels):
    z, y = torch.tensor(logits, dtype=torch.float64), torch.tensor(labels, dtype=torch.long)
    # Bounded validation NLL optimization; the bounds and optimizer are fixed before testing.
    parameter = torch.tensor(np.log((1 - .05) / (20 - 1)), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([parameter], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = .05 + 19.95 * parameter.sigmoid()
        loss = F.cross_entropy(z / temperature, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float((.05 + 19.95 * parameter.sigmoid()).detach())
    if not np.isfinite(temperature):
        raise FloatingPointError("Nonfinite fitted temperature.")
    return temperature


def resampled_qwk(y, prediction, indices):
    scores = []
    weights = (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2
    for start in range(0, len(indices), 1000):
        chunk = indices[start:start + 1000]
        codes = np.arange(len(chunk))[:, None] * 25 + y[chunk] * 5 + prediction[chunk]
        cm = np.bincount(codes.ravel(), minlength=len(chunk) * 25).reshape(-1, 5, 5)
        expected = cm.sum(2)[:, :, None] * cm.sum(1)[:, None, :] / len(y)
        denominator = (expected * weights).sum((1, 2))
        if (denominator == 0).any():
            raise FloatingPointError("Undefined bootstrap QWK; inspect class coverage.")
        scores.extend(1 - (cm * weights).sum((1, 2)) / denominator)
    return np.asarray(scores)


def bootstrap(y, left, right, replicates=10000):
    if len(left) != 3 or len(right) != 3:
        raise ValueError("Expected three paired seed arrays.")
    rng = np.random.default_rng(20261018)
    indices = rng.integers(0, len(y), (replicates, len(y)))
    values = np.mean([resampled_qwk(y, a, indices) - resampled_qwk(y, b, indices)
                      for a, b in zip(left, right)], axis=0)
    return {
        "mean_difference": float(np.mean([
            cohen_kappa_score(y, a, labels=range(5), weights="quadratic")
            - cohen_kappa_score(y, b, labels=range(5), weights="quadratic") for a, b in zip(left, right)
        ])),
        "percentile95": np.quantile(values, [.025, .975]).tolist(),
        "replicates": replicates, "seed": 20261018,
        "unit": "shared image indices across three fixed seed pairs; not pooled repeated images",
        "limitations": "conditional on trained models; no patient clustering or multiple-comparison adjustment",
    }


def reliability(y, p, path, title):
    confidence, prediction = p.max(1), p.argmax(1)
    bins = np.minimum((confidence * 15).astype(int), 14)
    x, a, counts = [], [], []
    for i in range(15):
        mask = bins == i
        if mask.any():
            x.append(confidence[mask].mean())
            a.append((prediction[mask] == y[mask]).mean())
            counts.append(int(mask.sum()))
    fig, axis = plt.subplots(figsize=(4, 4))
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.scatter(x, a, s=np.asarray(counts) * 2 + 10)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def analyze(output):
    destination = output / "analysis"
    destination.mkdir(exist_ok=True)
    summaries, calibrated, diversity = {}, {}, {}
    for run in matrix():
        path = output / run.key
        with np.load(path / "val_logits.npz", allow_pickle=False) as val, np.load(path / "test_logits.npz", allow_pickle=False) as test:
            y = test["labels"]
            paths = [k for k in test.files if k not in {"labels", "ids"}]
            summaries[run.key] = {name: metrics(y, probabilities(test[name])) for name in paths}
            if run.stage == "student":
                for name in ("primary", "student", "equal"):
                    if name not in paths:
                        continue
                    p = probabilities(test[name])
                    reliability(y, p, destination / f"{run.group}_{run.arm}_{name}_reliability.png", f"{run.arm}: {name}")
                    fig, axis = plt.subplots(figsize=(4, 4))
                    cm = confusion_matrix(y, p.argmax(1), labels=range(5))
                    axis.imshow(cm, cmap="Blues")
                    for (i, j), value in np.ndenumerate(cm):
                        axis.text(j, i, str(value), ha="center", va="center")
                    axis.set(xlabel="Predicted grade", ylabel="True grade", title=f"{run.arm}: {name}, seed {run.seed}")
                    fig.tight_layout()
                    fig.savefig(destination / f"{run.group}_{run.arm}_{name}_confusion.png", dpi=160)
                    plt.close(fig)
            if run.stage == "teacher2":
                left, right = "teacher1", "teacher2"
                a, b = probabilities(test[left]), probabilities(test[right])
                error_a, error_b = a.argmax(1) != y, b.argmax(1) != y
                diversity[run.key] = {
                    "disagreement": float((a.argmax(1) != b.argmax(1)).mean()),
                    "error_correlation": float(np.corrcoef(error_a, error_b)[0, 1]) if error_a.std() and error_b.std() else None,
                    "error_correlation_defined": bool(error_a.std() and error_b.std()),
                    "symmetric_kl_nats": float(.5 * (
                        (a * np.log(np.clip(a, 1e-12, 1) / np.clip(b, 1e-12, 1))).sum(1)
                        + (b * np.log(np.clip(b, 1e-12, 1) / np.clip(a, 1e-12, 1))).sum(1)).mean()),
                    "scope": "Descriptive diversity is not proof of a beneficial teacher contribution.",
                }
            elif run.stage == "student" and run.arm != "student_ce":
                left, right = "teacher", "student"
            else:
                continue
            tl, tr = fit_temperature(val[left], val["labels"]), fit_temperature(val[right], val["labels"])
            methods = {
                "raw_logit_average": probabilities((test[left] + test[right]) / 2),
                "probability_average": (probabilities(test[left]) + probabilities(test[right])) / 2,
                "validation_temperature_average": probabilities((test[left] / tl + test[right] / tr) / 2),
            }
            calibrated[run.key] = {
                "fit_split": "validation", "left_temperature": tl, "right_temperature": tr,
                "test_metrics": {name: metrics(y, p) for name, p in methods.items()},
            }
            for name, p in methods.items():
                reliability(y, p, destination / f"{run.group}_{run.stage}_{run.arm}_{name}.png", name)
    aggregates = {}
    for run in matrix():
        if run.seed != 42:
            continue
        key = f"{run.preprocessing}_f{int(run.fraction * 100):03d}/{run.stage}/{run.arm}"
        seed_results = [summaries[Run(run.preprocessing, run.fraction, seed, run.stage, run.arm).key] for seed in SEEDS]
        aggregates[key] = {
            path: {metric: {"mean": float(np.mean([r[path][metric] for r in seed_results])),
                            "sample_sd": float(np.std([r[path][metric] for r in seed_results], ddof=1))}
                   for metric in ("accuracy", "macro_f1", "qwk", "weighted_ovr_auc", "nll", "brier", "ece15")}
            for path in seed_results[0]
        }
    contrasts = {}
    pairs = [
        ("single_vs_ce", "joint_single_kd", "joint_ce"),
        ("multi_vs_single", "joint_multi_kd", "joint_single_kd"),
        ("ordinal", "joint_ordinal", "joint_multi_kd"),
        ("hinge", "joint_hinge", "joint_multi_kd"),
        ("gate_entropy", "joint_full", "joint_full_no_entropy"),
        ("teacher1_contribution", "joint_full", "teacher2_only"),
        ("augmentation", "augmentation", "joint_full"),
        ("teacher_buffers", "teacher_eval", "joint_full"),
        ("complete_vs_ce", "joint_full", "joint_ce"),
    ]
    for name, left, right in pairs:
        for path in ("student", "equal", "primary"):
            predictions = []
            ids = None
            labels = None
            for arm in (left, right):
                values = []
                for seed in SEEDS:
                    run = Run("raw", 1.0, seed, "student", arm)
                    with np.load(output / run.key / "test_logits.npz", allow_pickle=False) as z:
                        if ids is not None and (not np.array_equal(ids, z["ids"]) or not np.array_equal(labels, z["labels"])):
                            raise ValueError("Bootstrap prediction IDs or labels are not aligned.")
                        ids, labels = z["ids"].copy(), z["labels"].copy()
                        values.append(z[path].argmax(1))
                predictions.append(values)
            contrasts[f"{name}/{path}"] = bootstrap(labels, *predictions)
    additional = [
        ("equal_vs_independent_ce", ("raw", 1.0, "joint_full", "equal"), ("raw", 1.0, "student_ce", "primary")),
        ("student_vs_independent_ce", ("raw", 1.0, "joint_full", "student"), ("raw", 1.0, "student_ce", "primary")),
        ("equal_vs_teacher", ("raw", 1.0, "joint_full", "equal"), ("raw", 1.0, "joint_full", "teacher")),
        ("equal_vs_gated", ("raw", 1.0, "joint_full", "equal"), ("raw", 1.0, "joint_full", "primary")),
    ]
    for preprocessing in ("imagenet", "retinal"):
        for arm, path in (("joint_full", "equal"), ("student_ce", "primary")):
            additional.append((f"{preprocessing}/{arm}", (preprocessing, 1.0, arm, path), ("raw", 1.0, arm, path)))
    for fraction in (.1, .25, .5):
        additional.append((f"fraction_{fraction}/full_vs_ce", ("raw", fraction, "joint_full", "equal"),
                           ("raw", fraction, "student_ce", "primary")))
    for name, left, right in additional:
        predictions, reference_ids, labels = [], None, None
        for preprocessing, fraction, arm, path in (left, right):
            values = []
            for seed in SEEDS:
                run = Run(preprocessing, fraction, seed, "student", arm)
                with np.load(output / run.key / "test_logits.npz", allow_pickle=False) as z:
                    if reference_ids is not None and (not np.array_equal(reference_ids, z["ids"]) or not np.array_equal(labels, z["labels"])):
                        raise ValueError("Contrast image alignment failed.")
                    reference_ids, labels = z["ids"].copy(), z["labels"].copy()
                    values.append(z[path].argmax(1))
            predictions.append(values)
        contrasts[name] = bootstrap(labels, *predictions)
    write_json(destination / "per_seed.json", summaries)
    write_json(destination / "mean_sd.json", aggregates)
    write_json(destination / "calibration.json", calibrated)
    calibration_aggregates = {}
    for run in matrix():
        if run.seed != 42 or run.key not in calibrated:
            continue
        seed_results = [calibrated[Run(run.preprocessing, run.fraction, seed, run.stage, run.arm).key]["test_metrics"]
                        for seed in SEEDS]
        calibration_aggregates[run.key.replace("_s42", "")] = {
            name: {metric: {"mean": float(np.mean([r[name][metric] for r in seed_results])),
                            "sample_sd": float(np.std([r[name][metric] for r in seed_results], ddof=1))}
                   for metric in ("accuracy", "macro_f1", "qwk", "weighted_ovr_auc", "nll", "brier", "ece15")}
            for name in seed_results[0]
        }
    write_json(destination / "calibration_mean_sd.json", calibration_aggregates)
    write_json(destination / "teacher_diversity.json", diversity)
    write_json(destination / "paired_qwk.json", contrasts)
    write_json(destination / "status.json", {
        "training_jobs": len(matrix()), "complete": True,
        "interpretation": "New retraining results, not a replacement for an untouched external validation cohort.",
    })


def benchmark(output, bundle):
    if not torch.cuda.is_available():
        raise RuntimeError("Matched benchmark requires CUDA.")
    destination = output / "benchmark"
    destination.mkdir(exist_ok=True)
    records = []
    for seed in SEEDS:
        run = Run("raw", 1.0, seed, "student", "joint_full")
        checkpoint = load_best(output / run.key)
        source_model = build_model(run, output, checkpoint).eval().cpu()
        del checkpoint
        torch.cuda.empty_cache()
        for batch in (1, 16):
            data, _, _ = loaders(bundle, run)
            x = next(iter(data["val"]))[0][:batch].to(DEVICE)
            components = {
                "student": source_model.student_model, "teacher1": source_model.teacher_model.baseline_model,
                "teacher2": source_model.teacher_model.adv_model, "teacher_reference": source_model.teacher_model,
                "full_equal": source_model, "full_gated": source_model,
            }
            for name, component in components.items():
                isolated = deepcopy(component)
                if name == "full_equal":
                    isolated.gate = torch.nn.Identity()
                isolated = isolated.to(DEVICE).eval()
                def fn():
                    if name == "full_equal":
                        teacher, _ = isolated.teacher_model(x)
                        student, _ = isolated.student_model(x)
                        return (teacher + student) / 2
                    return isolated(x)
                with torch.inference_mode():
                    for _ in range(20):
                        result = fn()
                        del result
                    torch.cuda.synchronize()
                    resident = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    timings = []
                    for _ in range(100):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        result = fn()
                        end.record()
                        end.synchronize()
                        timings.append(start.elapsed_time(end))
                        del result
                parameters = sum(p.numel() for p in isolated.parameters())
                records.append({
                    "seed": seed, "path": name, "batch": batch, "precision": "float32", "warmup": 20, "repetitions": 100,
                    "gpu": torch.cuda.get_device_name(0), "parameters": parameters,
                    "ms_per_image_mean": float(np.mean(timings) / batch),
                    "ms_per_image_median": float(np.median(timings) / batch),
                    "ms_per_image_sd": float(np.std(timings, ddof=1) / batch),
                    "images_per_second": float(1000 * batch / np.mean(timings)),
                    "resident_allocated_bytes": resident, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "incremental_peak_bytes": torch.cuda.max_memory_allocated() - resident,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "memory_scope": "Only the measured path and input are resident on GPU; PyTorch allocated memory, not total driver/device usage.",
                    "timing_scope": "Forward only; excludes preprocessing, transfer and file I/O.",
                })
                del isolated
                gc.collect()
                torch.cuda.empty_cache()
            del x
        del source_model, components
        gc.collect()
        torch.cuda.empty_cache()
    write_json(destination / "matched_gpu.json", records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["analyze", "benchmark"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--drive", type=Path)
    args = parser.parse_args()
    if args.stage == "analyze":
        analyze(args.output)
        folder = args.output / "analysis"
    else:
        if args.data is None:
            parser.error("--data is required for benchmark")
        benchmark(args.output, load_data(args.data, ROOT / "split_manifest.json"))
        folder = args.output / "benchmark"
    if args.drive:
        synchronize(folder, args.drive / folder.name)
        # Figures are persistent reviewer evidence too.
        for path in folder.glob("*.png"):
            target = args.drive / folder.name / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(path, target)
            if file_hash(path) != file_hash(target):
                raise OSError(f"Figure persistence failed: {target}")
