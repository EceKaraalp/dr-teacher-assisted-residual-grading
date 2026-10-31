"""Recalculate metrics from archived DualExpert predictions without training."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "experiments_dual"
MANIFEST_PATH = ROOT / "results" / "split_manifest.json"
SUMMARY_PATH = INPUT_DIR / "results_summary.csv"
ALL_EXPERIMENTS = ("Exp2", "Exp4", "Exp5", "Exp7")
CLASS_NAMES = ("No DR", "Mild", "Moderate", "Severe", "Proliferative DR")
N_CLASSES = len(CLASS_NAMES)
REFERABLE_THRESHOLD = 2
ECE_BINS = 15
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261009
OUT_DIR = ROOT / "results" / "revision_existing_analysis_dual_expert_20261009"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != N_CLASSES:
        raise ValueError(f"Expected probabilities with shape (N, {N_CLASSES})")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Probabilities must be finite and non-negative")
    sums = probabilities.sum(axis=1, keepdims=True)
    if (sums <= 0).any():
        raise ValueError("Every probability row must have positive sum")
    return probabilities / sums


def ece_top_label(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == y_true
    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    result = 0.0
    for bin_index in range(ECE_BINS):
        in_bin = (confidence > edges[bin_index]) & (confidence <= edges[bin_index + 1])
        if in_bin.any():
            result += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(result)


def metric_values(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    y_pred = probabilities.argmax(axis=1)
    one_hot = np.eye(N_CLASSES, dtype=np.float64)[y_true]
    clipped = np.clip(probabilities, 1e-15, 1.0)
    try:
        macro_auc = float(
            roc_auc_score(y_true, probabilities, labels=np.arange(N_CLASSES), multi_class="ovr", average="macro")
        )
    except ValueError:
        macro_auc = None
    try:
        micro_auc = float(roc_auc_score(one_hot.ravel(), probabilities.ravel()))
    except ValueError:
        micro_auc = None

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(N_CLASSES))
    per_class = classification_report(
        y_true,
        y_pred,
        labels=np.arange(N_CLASSES),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    class_metrics = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        tp = int(cm[class_index, class_index])
        fn = int(cm[class_index, :].sum() - tp)
        fp = int(cm[:, class_index].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        class_metrics[class_name] = {
            "precision": float(per_class[class_name]["precision"]),
            "sensitivity_recall": float(per_class[class_name]["recall"]),
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "f1": float(per_class[class_name]["f1-score"]),
            "support": int(per_class[class_name]["support"]),
        }

    referable_true = y_true >= REFERABLE_THRESHOLD
    referable_pred = y_pred >= REFERABLE_THRESHOLD
    referable_prob = probabilities[:, REFERABLE_THRESHOLD:].sum(axis=1)
    tn, fp, fn, tp = confusion_matrix(referable_true, referable_pred, labels=[False, True]).ravel()
    return {
        "n": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc_ovr_macro": macro_auc,
        "roc_auc_micro": micro_auc,
        "negative_log_likelihood": float(-np.log(clipped[np.arange(y_true.size), y_true]).mean()),
        "multiclass_brier_score_mean_sum_squares": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_top_label_equal_width": ece_top_label(y_true, probabilities),
        "ece_bins": ECE_BINS,
        "per_class": class_metrics,
        "referable_dr": {
            "positive_definition": f"grade >= {REFERABLE_THRESHOLD} (grades 2, 3, 4)",
            "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "precision": float(tp / (tp + fp)) if tp + fp else None,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None,
            "roc_auc": float(roc_auc_score(referable_true, referable_prob)),
            "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
            "support_positive": int(referable_true.sum()),
            "support_negative": int((~referable_true).sum()),
        },
        "confusion_matrix": cm.astype(int).tolist(),
    }


def stratified_bootstrap(
    y_true: np.ndarray, probabilities: np.ndarray, seed: int
) -> dict[str, dict[str, float | None]]:
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y_true == c) for c in range(N_CLASSES)]
    samples: dict[str, list[float]] = {
        name: []
        for name in (
            "accuracy",
            "cohen_kappa",
            "qwk",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "balanced_accuracy",
            "matthews_correlation_coefficient",
            "roc_auc_ovr_macro",
            "roc_auc_micro",
            "negative_log_likelihood",
            "multiclass_brier_score_mean_sum_squares",
            "ece_top_label_equal_width",
        )
    }
    binary_samples: dict[str, list[float]] = {
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "f1": [],
        "roc_auc": [],
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = np.concatenate(
            [rng.choice(class_indices, size=class_indices.size, replace=True) for class_indices in by_class]
        )
        boot_y = y_true[indices]
        boot_probabilities = probabilities[indices]
        boot_pred = boot_probabilities.argmax(axis=1)
        boot_cm = confusion_matrix(boot_y, boot_pred, labels=np.arange(N_CLASSES))
        tp = np.diag(boot_cm).astype(float)
        support = boot_cm.sum(axis=1)
        predicted_count = boot_cm.sum(axis=0)
        per_class_recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
        per_class_precision = np.divide(tp, predicted_count, out=np.zeros_like(tp), where=predicted_count > 0)
        per_class_f1 = np.divide(
            2 * per_class_precision * per_class_recall,
            per_class_precision + per_class_recall,
            out=np.zeros_like(tp),
            where=(per_class_precision + per_class_recall) > 0,
        )
        one_hot = np.eye(N_CLASSES, dtype=np.float64)[boot_y]
        clipped = np.clip(boot_probabilities, 1e-15, 1.0)
        try:
            macro_auc = float(
                roc_auc_score(
                    boot_y,
                    boot_probabilities,
                    labels=np.arange(N_CLASSES),
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError:
            macro_auc = None
        try:
            micro_auc = float(roc_auc_score(one_hot.ravel(), boot_probabilities.ravel()))
        except ValueError:
            micro_auc = None
        tn, fp, fn, binary_tp = confusion_matrix(
            boot_y >= REFERABLE_THRESHOLD,
            boot_pred >= REFERABLE_THRESHOLD,
            labels=[False, True],
        ).ravel()
        referable_true = boot_y >= REFERABLE_THRESHOLD
        referable_prob = boot_probabilities[:, REFERABLE_THRESHOLD:].sum(axis=1)
        boot_values = {
            "accuracy": accuracy_score(boot_y, boot_pred),
            "cohen_kappa": cohen_kappa_score(boot_y, boot_pred),
            "qwk": cohen_kappa_score(boot_y, boot_pred, weights="quadratic"),
            "precision_macro": precision_score(boot_y, boot_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(boot_y, boot_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(boot_y, boot_pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(boot_y, boot_pred),
            "matthews_correlation_coefficient": matthews_corrcoef(boot_y, boot_pred),
            "roc_auc_ovr_macro": macro_auc,
            "roc_auc_micro": micro_auc,
            "negative_log_likelihood": -np.log(clipped[np.arange(boot_y.size), boot_y]).mean(),
            "multiclass_brier_score_mean_sum_squares": np.square(boot_probabilities - one_hot).sum(axis=1).mean(),
            "ece_top_label_equal_width": ece_top_label(boot_y, boot_probabilities),
        }
        for name, value in boot_values.items():
            if value is not None and np.isfinite(value):
                samples[name].append(float(value))
        binary_values = {
            "sensitivity": binary_tp / (binary_tp + fn) if binary_tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "precision": binary_tp / (binary_tp + fp) if binary_tp + fp else None,
            "f1": (
                2 * binary_tp / (2 * binary_tp + fp + fn)
                if 2 * binary_tp + fp + fn
                else None
            ),
            "roc_auc": roc_auc_score(referable_true, referable_prob),
        }
        for name, value in binary_values.items():
            if value is not None and np.isfinite(value):
                binary_samples[name].append(float(value))

    def intervals(values: dict[str, list[float]]) -> dict[str, dict[str, float | None]]:
        output = {}
        for name, metric_samples in values.items():
            if metric_samples:
                lower, upper = np.percentile(metric_samples, [2.5, 97.5])
                output[name] = {"lower_95": float(lower), "upper_95": float(upper)}
            else:
                output[name] = {"lower_95": None, "upper_95": None}
        return output

    return {
        "overall": intervals(samples),
        "referable_dr": intervals(binary_samples),
        "method": "class-stratified percentile bootstrap; images sampled within each observed class",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
    }


def save_figures(
    experiment: str, y_true: np.ndarray, probabilities: np.ndarray, train_log: pd.DataFrame, out_dir: Path
) -> None:
    y_pred = probabilities.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(N_CLASSES))
    row_totals = cm.sum(axis=1, keepdims=True)
    normalized_cm = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, matrix, title, fmt in (
        (axes[0], cm, "Archived test predictions: counts", "d"),
        (axes[1], normalized_cm, "Archived test predictions: row-normalized", ".2f"),
    ):
        image = ax.imshow(matrix, cmap="Blues", vmin=0)
        ax.set(title=title, xlabel="Predicted grade", ylabel="True grade")
        ax.set_xticks(range(N_CLASSES), CLASS_NAMES, rotation=35, ha="right")
        ax.set_yticks(range(N_CLASSES), CLASS_NAMES)
        for row in range(N_CLASSES):
            for column in range(N_CLASSES):
                value = matrix[row, column]
                ax.text(column, row, format(value, fmt), ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{experiment}: archived DualExpert output (not final TS model)")
    fig.savefig(out_dir / f"{experiment}_test_confusion.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for class_index, class_name in enumerate(CLASS_NAMES):
        binary_true = (y_true == class_index).astype(int)
        if np.unique(binary_true).size == 2:
            fpr, tpr, _ = roc_curve(binary_true, probabilities[:, class_index])
            auc = roc_auc_score(binary_true, probabilities[:, class_index])
            ax.plot(fpr, tpr, label=f"{class_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set(title=f"{experiment}: archived test one-vs-rest ROC", xlabel="False-positive rate", ylabel="Sensitivity")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(out_dir / f"{experiment}_test_roc.png", dpi=300)
    plt.close(fig)

    confidence = probabilities.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    edges = np.linspace(0, 1, ECE_BINS + 1)
    bin_confidence, bin_accuracy = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        selected = (confidence > left) & (confidence <= right)
        if selected.any():
            bin_confidence.append(float(confidence[selected].mean()))
            bin_accuracy.append(float(correct[selected].mean()))
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    ax.plot(bin_confidence, bin_accuracy, marker="o", label=f"Top-label, {ECE_BINS} equal-width bins")
    ax.set(xlim=(0, 1), ylim=(0, 1), title=f"{experiment}: archived test reliability", xlabel="Mean confidence", ylabel="Accuracy")
    ax.legend()
    fig.savefig(out_dir / f"{experiment}_test_reliability.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for class_index, class_name in enumerate(CLASS_NAMES):
        binary_true = (y_true == class_index).astype(int)
        if np.unique(binary_true).size == 2:
            precision, recall, _ = precision_recall_curve(binary_true, probabilities[:, class_index])
            ax.plot(recall, precision, label=class_name)
    ax.set(title=f"{experiment}: archived test one-vs-rest precision-recall", xlabel="Recall", ylabel="Precision")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / f"{experiment}_test_precision_recall.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(train_log["epoch"], train_log["train_loss"], label="Training")
    axes[0].plot(train_log["epoch"], train_log["val_loss"], label="Validation")
    axes[0].set(title=f"{experiment}: loss history", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(train_log["epoch"], train_log["val_qwk"], label="Validation QWK")
    axes[1].set(title=f"{experiment}: validation history", xlabel="Epoch", ylabel="QWK")
    axes[1].legend()
    fig.savefig(out_dir / f"{experiment}_training_validation_history.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    requested_experiments = tuple(
        item.strip() for item in os.environ.get("DUAL_EXPERIMENTS", ",".join(ALL_EXPERIMENTS)).split(",") if item.strip()
    )
    if not requested_experiments or any(item not in ALL_EXPERIMENTS for item in requested_experiments):
        raise ValueError(f"DUAL_EXPERIMENTS must contain values from {ALL_EXPERIMENTS}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    summary = pd.read_csv(SUMMARY_PATH).set_index("experiment")
    reference_ids = manifest["sample_ids"]["test"]
    labels_by_id = pd.read_csv(ROOT / "APTOS2019" / "train.csv").set_index("id_code")["diagnosis"]
    manifest_labels = labels_by_id.loc[reference_ids].to_numpy(dtype=np.int64)

    for experiment in requested_experiments:
        exp_dir = INPUT_DIR / experiment
        y_path = exp_dir / "test_y_true.npy"
        probability_path = exp_dir / "test_y_proba.npy"
        log_path = exp_dir / "training_log.csv"
        expected_outputs = [
            OUT_DIR / f"{experiment}_metrics.json",
            OUT_DIR / f"{experiment}_classification_report.json",
            OUT_DIR / f"{experiment}_per_class_metrics.csv",
            OUT_DIR / f"{experiment}_test_predictions.csv",
            OUT_DIR / f"{experiment}_test_labels.npy",
            OUT_DIR / f"{experiment}_test_probabilities.npy",
            OUT_DIR / f"{experiment}_test_confusion.png",
            OUT_DIR / f"{experiment}_test_roc.png",
            OUT_DIR / f"{experiment}_test_reliability.png",
            OUT_DIR / f"{experiment}_test_precision_recall.png",
            OUT_DIR / f"{experiment}_training_validation_history.png",
        ]
        existing_outputs = [path for path in expected_outputs if path.exists()]
        if existing_outputs:
            if len(existing_outputs) == len(expected_outputs):
                print(f"{experiment}: outputs already exist; preserving and skipping.")
                continue
            raise FileExistsError(
                f"{experiment}: partial output set exists; refusing to overwrite: "
                + ", ".join(path.name for path in existing_outputs)
            )
        y_true = np.load(y_path).astype(np.int64)
        probabilities = normalized_probabilities(np.load(probability_path))
        if y_true.ndim != 1 or y_true.size != probabilities.shape[0]:
            raise ValueError(f"{experiment}: label and probability array shapes do not align")
        if (y_true < 0).any() or (y_true >= N_CLASSES).any():
            raise ValueError(f"{experiment}: labels outside 0..{N_CLASSES-1}")
        log = pd.read_csv(log_path)
        metrics = metric_values(y_true, probabilities)
        intervals = stratified_bootstrap(y_true, probabilities, BOOTSTRAP_SEED)
        report = classification_report(
            y_true,
            probabilities.argmax(axis=1),
            labels=np.arange(N_CLASSES),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        per_class_rows = []
        for class_name in CLASS_NAMES:
            row = dict(metrics["per_class"][class_name])
            row["class"] = class_name
            per_class_rows.append(row)
        predictions = pd.DataFrame(
            {
                "archived_row_index": np.arange(y_true.size),
                "y_true": y_true,
                "y_pred": probabilities.argmax(axis=1),
                **{f"probability_grade_{class_index}": probabilities[:, class_index] for class_index in range(N_CLASSES)},
            }
        )
        predictions.to_csv(OUT_DIR / f"{experiment}_test_predictions.csv", index=False)
        np.save(OUT_DIR / f"{experiment}_test_labels.npy", y_true)
        np.save(OUT_DIR / f"{experiment}_test_probabilities.npy", probabilities.astype(np.float32))
        pd.DataFrame(per_class_rows).to_csv(OUT_DIR / f"{experiment}_per_class_metrics.csv", index=False)
        (OUT_DIR / f"{experiment}_classification_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (OUT_DIR / f"{experiment}_metrics.json").write_text(
            json.dumps({"metrics": metrics, "confidence_intervals": intervals}, indent=2),
            encoding="utf-8",
        )
        save_figures(experiment, y_true, probabilities, log, OUT_DIR)

    experiment_outputs = {}
    y_vectors = []
    completed_experiments = []
    for experiment in ALL_EXPERIMENTS:
        metrics_path = OUT_DIR / f"{experiment}_metrics.json"
        if not metrics_path.exists():
            continue
        exp_dir = INPUT_DIR / experiment
        y_path = exp_dir / "test_y_true.npy"
        probability_path = exp_dir / "test_y_proba.npy"
        log_path = exp_dir / "training_log.csv"
        y_true = np.load(y_path).astype(np.int64)
        raw_probabilities = np.load(probability_path)
        probabilities = normalized_probabilities(raw_probabilities)
        log = pd.read_csv(log_path)
        y_vectors.append(y_true)
        completed_experiments.append(experiment)
        experiment_outputs[experiment] = {
            "source_label_sha256": sha256(y_path),
            "source_probability_sha256": sha256(probability_path),
            "source_log_sha256": sha256(log_path),
            "n": int(y_true.size),
            "class_counts": np.bincount(y_true, minlength=N_CLASSES).tolist(),
            "probability_max_row_sum_error_before_normalization": float(
                np.abs(raw_probabilities.sum(axis=1) - 1).max()
            ),
            "labels_match_current_split_manifest_in_manifest_order": bool(
                np.array_equal(y_true, manifest_labels)
            ),
            "test_metrics_recomputed": json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"],
            "confidence_intervals": json.loads(metrics_path.read_text(encoding="utf-8"))["confidence_intervals"],
            "test_metrics_from_archived_summary": summary.loc[experiment].to_dict(),
            "training_log_rows": len(log),
            "training_log_final_epoch": int(log["epoch"].max()),
        }

    same_label_order = bool(y_vectors) and all(
        np.array_equal(y_vectors[0], vector) for vector in y_vectors[1:]
    )
    protocol = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "CPU-only recalculation of archived DualExpert predictions; no training or checkpoint inference.",
        "model_scope": "Historical DualExpert experiments Exp2/Exp4/Exp5/Exp7; not the manuscript's final TS_ConvNeXtTiny_Residual model.",
        "test_identity_status": "The archived labels have the same class counts as the current manifest but do not match its ordered label vector. Archived sample IDs are unavailable, so equivalence to the frozen manifest test images cannot be established.",
        "labels_same_order_across_archived_experiments": same_label_order,
        "paired_sample_identity_verified": False,
        "class_order": list(CLASS_NAMES),
        "referable_dr_definition": "Positive grades 2, 3, 4 (grade >= 2); fixed before analysis.",
        "probability_handling": "Input probabilities checked for finite, non-negative values and normalized row-wise to sum to one; maximum original row-sum error recorded per experiment.",
        "ece": f"Top-label ECE with {ECE_BINS} equal-width bins on [0,1].",
        "bootstrap": {
            "method": "Class-stratified image-row percentile bootstrap, resampling within each observed class.",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interpretation": "Intervals quantify finite-sample uncertainty conditional on each archived prediction set; they do not quantify between-seed training variability.",
        },
        "undefined_metric_policy": "AUC is null if undefined; precision/recall/F1 use zero_division=0, with per-class support provided.",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch_package": importlib.metadata.version("torch"),
            "cuda_used": False,
        },
        "frozen_manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
        "experiments": experiment_outputs,
        "limitations": [
            "Do not transfer these results to the final teacher-student model.",
            "Archived test arrays contain no sample IDs; exact image-level split identity is unverified.",
            "Do not conduct paired model significance tests without verified row-level sample identity.",
            "External validation, Grad-CAM, latency measurement, and controlled retraining ablations are not produced by this analysis.",
        ],
    }
    if len(completed_experiments) == len(ALL_EXPERIMENTS):
        metadata_path = OUT_DIR / "analysis_metadata.json"
        if metadata_path.exists():
            raise FileExistsError(f"Refusing to overwrite {metadata_path}")
        metadata_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    else:
        print(f"Pending archived experiments: {', '.join(sorted(set(ALL_EXPERIMENTS) - set(completed_experiments)))}")
    print(f"Analysis saved: {OUT_DIR}")
    print(f"Archived experiments completed: {', '.join(completed_experiments)}")
    print(f"All archived label vectors identical: {same_label_order}")
    print(f"Archived labels match frozen manifest order: {experiment_outputs['Exp2']['labels_match_current_split_manifest_in_manifest_order']}")
    for experiment in completed_experiments:
        metrics = experiment_outputs[experiment]["test_metrics_recomputed"]
        print(
            f"{experiment}: accuracy={metrics['accuracy']:.4f}, "
            f"QWK={metrics['qwk']:.4f}, macro-F1={metrics['f1_macro']:.4f}, "
            f"NLL={metrics['negative_log_likelihood']:.4f}, "
            f"ECE={metrics['ece_top_label_equal_width']:.4f}, "
            f"referable sensitivity={metrics['referable_dr']['sensitivity']:.4f}, "
            f"specificity={metrics['referable_dr']['specificity']:.4f}"
        )


if __name__ == "__main__":
    main()
