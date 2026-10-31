"""Summarize the archived equal teacher-student inference predictions, without training."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluate_idrid import bootstrap
from verify_archived_final import metrics, write_json


MAIN = "equal_ts_archived_teacher"
SEEDS = (42, 52, 62)


def wilson(successes: int, total: int) -> list[float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    z = 1.959963984540054
    p = successes / total
    divisor = 1 + z * z / total
    center = (p + z * z / (2 * total)) / divisor
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / divisor
    return [float(center - radius), float(center + radius)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    evidence = args.repo / "results" / "archived_final_verification_2026-10-16"
    arrays = {}
    for seed in SEEDS:
        with np.load(evidence / f"seed_{seed}_probabilities.npz") as data:
            arrays[seed] = {k: data[k].copy() for k in data.files}
    labels = arrays[42]["labels"]
    ids = arrays[42]["ids"]
    for a in arrays.values():
        if not np.array_equal(labels, a["labels"]) or not np.array_equal(ids, a["ids"]):
            raise ValueError("Seed predictions are not aligned")
    results = {str(seed): metrics(labels, arrays[seed][MAIN]) for seed in SEEDS}
    statistics = {
        key: {"mean": float(np.mean([r[key] for r in results.values()])),
              "sample_sd": float(np.std([r[key] for r in results.values()], ddof=1))}
        for key in ("accuracy", "f1_macro", "qwk", "roc_auc", "nll",
                    "brier_multiclass_sum", "ece_15_equal_width")
    }
    uncertainty = bootstrap(labels, [arrays[s][MAIN] for s in SEEDS])
    rng = np.random.default_rng(20261016)
    indices = rng.integers(0, len(labels), size=(10000, len(labels)))
    true = labels[indices]
    offsets = np.arange(len(indices))[:, None] * 25
    weights = (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2

    def qwk_samples(mode: str) -> np.ndarray:
        samples = []
        for seed in SEEDS:
            pred = arrays[seed][mode].argmax(1)[indices]
            codes = true * 5 + pred
            cm = np.bincount((codes + offsets).ravel(), minlength=len(indices) * 25).reshape(-1, 5, 5)
            expected = (cm.sum(2)[:, :, None] * cm.sum(1)[:, None, :] * weights).sum((1, 2)) / len(labels)
            if (expected <= 0).any():
                raise ValueError("Undefined paired-bootstrap QWK")
            samples.append(1 - (cm * weights).sum((1, 2)) / expected)
        return np.mean(samples, axis=0)

    main_boot = qwk_samples(MAIN)
    if not np.allclose(np.quantile(main_boot, [0.025, 0.975]),
                       uncertainty["mean_metric_95ci"]["qwk"], atol=1e-12):
        raise ValueError("Independent bootstrap implementations disagree")
    contrasts = {}
    for mode in ("archived_teacher", "archived_residual", "manuscript_equal_logits"):
        point = np.mean([metrics(labels, arrays[s][MAIN])["qwk"] -
                         metrics(labels, arrays[s][mode])["qwk"] for s in SEEDS])
        contrasts[mode] = {
            "mean_difference_equal_minus_comparator": float(point),
            "percentile95": np.quantile(main_boot - qwk_samples(mode), [0.025, 0.975]).tolist(),
        }
    cm = np.array(results["62"]["confusion_matrix"])
    draws = np.random.default_rng(20261016).multinomial(
        len(labels), cm.ravel() / len(labels), size=10000).reshape(-1, 5, 5)
    per_class = []
    grades = ("No DR", "Mild", "Moderate", "Severe", "Proliferative")
    for grade, name in enumerate(grades):
        tp = int(cm[grade, grade])
        fn = int(cm[grade].sum() - tp)
        fp = int(cm[:, grade].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        denominator = draws[:, grade, :].sum(1) + draws[:, :, grade].sum(1)
        f1 = np.divide(2 * draws[:, grade, grade], denominator,
                       out=np.zeros(len(draws)), where=denominator > 0)
        per_class.append({
            "grade": name, "n": tp + fn, "precision": tp / (tp + fp),
            "recall": tp / (tp + fn), "specificity": tn / (tn + fp),
            "f1": 2 * tp / (2 * tp + fp + fn),
            "precision_ci": wilson(tp, tp + fp), "recall_ci": wilson(tp, tp + fn),
            "specificity_ci": wilson(tn, tn + fp),
            "f1_ci": np.quantile(f1, [0.025, 0.975]).tolist(),
        })
    referable = []
    for seed in SEEDS:
        pred = arrays[seed][MAIN].argmax(1)
        tp = int(((labels >= 2) & (pred >= 2)).sum())
        fn = int(((labels >= 2) & (pred < 2)).sum())
        tn = int(((labels < 2) & (pred < 2)).sum())
        fp = int(((labels < 2) & (pred >= 2)).sum())
        referable.append({"seed": seed, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
                          "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
                          "sensitivity_ci": wilson(tp, tp + fn),
                          "specificity_ci": wilson(tn, tn + fp)})
    df = pd.read_csv(args.repo / "APTOS2019" / "train.csv")
    trainval, test = train_test_split(np.arange(len(df)), test_size=0.1,
                                    random_state=42, stratify=df["diagnosis"])
    if df.iloc[test]["id_code"].tolist() != ids.tolist():
        raise ValueError("Reconstructed APTOS test IDs do not match saved predictions")
    with np.load(args.repo / "APTOS2019" / "cache_224.npz") as cache:
        hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in cache["images"]]
    other = {hashes[i] for i in trainval}
    mask = np.array([hashes[i] not in other for i in test])
    duplicate = {
        "excluded": int((~mask).sum()), "remaining": int(mask.sum()),
        "per_seed": {str(s): metrics(labels[mask], arrays[s][MAIN][mask]) for s in SEEDS},
    }
    write_json(args.output / "summary.json", {
        "main_mode": MAIN, "seeds": results, "mean_sd": statistics,
        "bootstrap": uncertainty, "paired_qwk": contrasts,
        "seed62_classwise": per_class, "referable": referable,
        "duplicate_subset": duplicate,
        "source": "Existing directly evaluated per-image predictions; no fitting or training",
    })
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xlabel="Predicted grade", ylabel="True grade", xticks=range(5), yticks=range(5),
           title="Equal teacher-student inference, seed 62")
    fig.tight_layout()
    fig.savefig(args.output / "aptos_equal_seed62_confusion.png", dpi=200)
    plt.close(fig)
    print(json.dumps({"mean_sd": statistics, "paired_qwk": contrasts,
                      "seed62_classwise": per_class, "referable": referable}, indent=2))


if __name__ == "__main__":
    main()
