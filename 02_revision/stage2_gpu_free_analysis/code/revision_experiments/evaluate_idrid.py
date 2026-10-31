"""Locked APTOS checkpoints on the official IDRiD test set, with branch Grad-CAM."""

import argparse
import csv
import io
import json
import platform
import zipfile
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from torch.nn import functional as F

from verify_archived_final import build_model, metrics, sha256, write_json


def load_test(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    root = "B. Disease Grading/"
    label_path = root + "2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    image_root = root + "1. Original Images/b. Testing Set/"
    with zipfile.ZipFile(path) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read(label_path).decode("utf-8-sig"))))
        ids = [r["Image name"] for r in rows]
        labels = np.array([int(r["Retinopathy grade"]) for r in rows])
        names = {Path(n).stem: n for n in archive.namelist()
                 if n.startswith(image_root) and n.lower().endswith((".jpg", ".png", ".jpeg"))}
        if len(rows) != 103 or len(set(ids)) != 103 or set(ids) != set(names):
            raise ValueError("Official test images and labels must match exactly (103 IDs)")
        if set(labels) != set(range(5)):
            raise ValueError("Expected five DR severity labels 0..4")
        images = []
        hashes = {}
        for image_id in ids:
            raw = archive.read(names[image_id])
            import hashlib

            hashes[image_id] = hashlib.sha256(raw).hexdigest()
            with Image.open(io.BytesIO(raw)) as image:
                images.append(np.array(image.convert("RGB").resize(
                    (224, 224), Image.Resampling.BILINEAR)))
    return np.stack(images), labels, ids, {"label_member": label_path, "image_sha256": hashes}


def bootstrap(labels: np.ndarray, probabilities: list[np.ndarray]) -> dict:
    rng = np.random.default_rng(20261016)
    indices = rng.integers(0, len(labels), size=(10000, len(labels)))
    y = labels[indices]
    weights = (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2
    scores = []
    for probabilities_seed in probabilities:
        pred = probabilities_seed.argmax(1)[indices]
        cm = np.stack([((y == a) & (pred == b)).sum(1)
                       for a in range(5) for b in range(5)], axis=1).reshape(-1, 5, 5)
        support = cm.sum(2)
        predicted = cm.sum(1)
        denominator = ((support[:, :, None] * predicted[:, None, :]) *
                       weights).sum((1, 2)) / len(labels)
        if np.any(denominator == 0):
            raise ValueError("Undefined QWK in bootstrap; report explicitly")
        qwk = 1 - (cm * weights).sum((1, 2)) / denominator
        diag = np.diagonal(cm, axis1=1, axis2=2)
        f1_denominator = support + predicted
        f1 = np.divide(2 * diag, f1_denominator,
                       out=np.zeros_like(diag, dtype=float), where=f1_denominator > 0).mean(1)
        scores.append(np.stack([diag.sum(1) / len(labels), f1, qwk], axis=1))
    intervals = np.quantile(np.mean(scores, axis=0), [0.025, 0.975], axis=0)
    return {
        "replicates": 10000, "rng_seed": 20261016,
        "method": "Percentile image bootstrap; shared sampled IDs across three fixed seeds; "
                  "mean seed metric, not pooled repeated images; fixed five classes for macro-F1.",
        "limitations": "No patient-level grouping or model-training uncertainty; small class counts.",
        "mean_metric_95ci": {name: intervals[:, i].tolist()
                             for i, name in enumerate(("accuracy", "f1_macro", "qwk"))},
    }


def inference_logits(model, images: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "equal":
        teacher, _ = model.teacher_model(images)
        student, _ = model.student_model(images)
        return (teacher + student) / 2
    if mode == "residual":
        return model(images)[0]
    raise ValueError(f"Unknown inference mode: {mode}")


def gradcam(model, image: np.ndarray, output: Path, image_id: str, label: int,
            mode: str = "residual") -> dict:
    targets = {
        "Teacher1_CNN": model.teacher_model.baseline_model.cnn_backbone,
        "Teacher2_ResNet": model.teacher_model.adv_model.cnn_features,
        "Student_ConvNeXt": model.student_model.backbone.features,
    }
    tensor = torch.from_numpy(image[None]).permute(0, 3, 1, 2).float()
    with torch.no_grad():
        reference = inference_logits(model, tensor, mode)
    activations = {}
    handles = []

    def hook(name):
        def capture(module, inputs, value):
            activations[name] = value
        return capture

    for name, layer in targets.items():
        handles.append(layer.register_forward_hook(hook(name)))
    try:
        tensor.requires_grad_(True)
        # Preserve the original formulas, enabling only their detached teacher paths.
        # Scoped to this single-process attribution forward, never used for evaluation.
        with patch.object(torch, "no_grad", torch.enable_grad):
            logits = inference_logits(model, tensor, mode)
        error = float((logits.detach() - reference).abs().max())
        if not torch.allclose(logits.detach(), reference, atol=1e-4, rtol=1e-5):
            raise ValueError(f"Attribution forward changed logits for {image_id}: {error}")
        predicted = int(reference.argmax(1))
        gradients = torch.autograd.grad(logits[0, predicted], list(activations.values()))
        maps = {}
        status = {}
        for (name, activation), gradient in zip(activations.items(), gradients):
            if not torch.isfinite(gradient).all() or not torch.isfinite(activation).all():
                raise ValueError(f"Nonfinite Grad-CAM values: {image_id}, {name}")
            cam = (gradient.mean((2, 3), keepdim=True) * activation).sum(1, keepdim=True).relu()
            cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
            maximum = float(cam.max().detach())
            status[name] = {"positive_cam_max": maximum,
                            "gradient_abs_max": float(gradient.abs().max()),
                            "zero_positive_map": maximum == 0}
            maps[name] = (cam / maximum if maximum > 0 else cam).detach().numpy()
        np.savez_compressed(output / f"{image_id}_gradcam.npz", **maps)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(image)
        axes[0].set_title(f"{image_id}\nTrue={label}, predicted={predicted}")
        for axis, (name, cam) in zip(axes[1:], maps.items()):
            axis.imshow(image)
            if not status[name]["zero_positive_map"]:
                axis.imshow(cam, cmap="jet", vmin=0, vmax=1, alpha=0.45)
            axis.set_title(name + ("\nNo positive CAM" if status[name]["zero_positive_map"] else ""))
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(output / f"{image_id}_gradcam.png", dpi=180)
        plt.close(fig)
        return {"id": image_id, "true_grade": label, "target_predicted_grade": predicted,
                "forward_max_abs_logit_difference": error, "branches": status}
    finally:
        for handle in handles:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path(r"C:\Users\user\Downloads\B. Disease Grading.zip"))
    parser.add_argument("--archive", type=Path, default=Path(r"D:\m1\Documents"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--inference", choices=("equal", "residual"), default="residual")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new result directory: {args.output}")
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    paths = [args.archive / f"seed {seed}" / "best_model.pth" for seed in (42, 52, 62)]
    notebook = args.archive / "train_lowdata_hybrid_distill.ipynb"
    sources = [args.zip, *paths, notebook, Path(__file__),
               Path(__file__).with_name("verify_archived_final.py"),
               *sorted((args.archive / "TS_ConvNeXtTiny_Residual_Package").rglob("*.py"))]
    hashes = {str(path): sha256(path) for path in sources}
    images, labels, ids, provenance = load_test(args.zip)
    provenance.update({
        "source_sha256": hashes, "python": platform.python_version(), "torch": torch.__version__,
        "device": "cpu", "training": False, "split": "Official IDRiD testing set only",
        "n": len(ids), "class_counts": np.bincount(labels, minlength=5).tolist(),
        "preprocessing": "PIL RGB bilinear 224x224, float32 0..255; no normalization, no TTA",
        "selection": "All three archived APTOS checkpoints; no IDRiD fitting or seed selection",
        "inference_mode": args.inference,
        "inference_rule": ("(z_teacher + z_student) / 2; learned teacher fusion retained"
                           if args.inference == "equal" else "Archived gated residual forward"),
        "checkpoint_selection_caveat": "Historical APTOS checkpoint-selection limitations remain; "
                                      "IDRiD data were not used to select these checkpoints.",
    })
    write_json(args.output / "provenance.json", provenance)
    print("IDRiD official test verified: 103 images, labels 0..4", flush=True)
    model = build_model(args.archive)
    results = {}
    probabilities = []
    for seed, path in zip((42, 52, 62), paths):
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        batches = []
        with torch.inference_mode():
            for start in range(0, len(images), 8):
                x = torch.from_numpy(images[start:start + 8]).permute(0, 3, 1, 2).float()
                logits = inference_logits(model, x, args.inference)
                batches.append(logits.softmax(1).numpy())
                print(f"Seed {seed}: {min(start + 8, len(images))}/103", flush=True)
        probs = np.concatenate(batches).astype(np.float64)
        probabilities.append(probs)
        result = metrics(labels, probs)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, probs.argmax(1), labels=range(5), zero_division=0)
        result["per_class"] = [
            {"grade": i, "precision": float(precision[i]), "recall": float(recall[i]),
             "f1": float(f1[i]), "support": int(support[i])} for i in range(5)]
        result["referable_argmax_grade_ge_2"] = {
            "tp": int(((labels >= 2) & (probs.argmax(1) >= 2)).sum()),
            "fn": int(((labels >= 2) & (probs.argmax(1) < 2)).sum()),
            "tn": int(((labels < 2) & (probs.argmax(1) < 2)).sum()),
            "fp": int(((labels < 2) & (probs.argmax(1) >= 2)).sum()),
        }
        results[str(seed)] = result
        write_json(args.output / f"seed_{seed}_metrics.json", result)
        np.savez_compressed(args.output / f"seed_{seed}_probabilities.npz",
                            labels=labels, ids=np.array(ids), probabilities=probs)
        with (args.output / f"seed_{seed}_predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "true_grade", "predicted_grade", *[f"p_{i}" for i in range(5)]])
            for image_id, label, p in zip(ids, labels, probs):
                writer.writerow([image_id, int(label), int(p.argmax()), *p.tolist()])
        print(f"Seed {seed}: {result['accuracy']=}, {result['f1_macro']=}, {result['qwk']=}", flush=True)
    summary = {"seeds": results, "mean_sd": {
        key: {"mean": float(np.mean([r[key] for r in results.values()])),
              "sample_sd": float(np.std([r[key] for r in results.values()], ddof=1))}
        for key in ("accuracy", "f1_macro", "qwk", "roc_auc")},
        "bootstrap": bootstrap(labels, probabilities)}
    write_json(args.output / "summary.json", summary)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for axis, (seed, result) in zip(axes, results.items()):
        cm = np.array(result["confusion_matrix"])
        axis.imshow(cm, cmap="Blues")
        for i in range(5):
            for j in range(5):
                axis.text(j, i, str(cm[i, j]), ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black")
        axis.set(xticks=range(5), yticks=range(5), xlabel="Predicted grade",
                 ylabel="True grade", title=f"Seed {seed}; QWK={result['qwk']:.4f}")
    fig.tight_layout()
    fig.savefig(args.output / "idrid_confusion_matrices.png", dpi=180)
    plt.close(fig)
    # Fixed seed 42, not selected using external accuracy. Deterministic examples,
    # first ID in each true-class/correctness stratum, not hand-picked CAM quality.
    model.load_state_dict(torch.load(paths[0], map_location="cpu", weights_only=True), strict=True)
    model.eval()
    cam_output = args.output / "gradcam"
    cam_output.mkdir()
    prediction = probabilities[0].argmax(1)
    selected = []
    for grade in range(5):
        for correct in (True, False):
            candidates = [i for i in range(len(ids))
                          if labels[i] == grade and (prediction[i] == labels[i]) == correct]
            if candidates:
                selected.append(min(candidates, key=lambda i: ids[i]))
    records = []
    for index in selected:
        print(f"Grad-CAM: {ids[index]} true={labels[index]} predicted={prediction[index]}", flush=True)
        records.append(gradcam(model, images[index], cam_output, ids[index], int(labels[index]),
                               mode=args.inference))
    write_json(cam_output / "manifest.json", {
        "seed": 42, "target": f"Predicted class logit of full {args.inference} inference system",
        "selection": "First lexicographic ID in each true-grade x correct/incorrect stratum",
        "method": "Grad-CAM: spatial mean gradients, channel-weighted activations, ReLU, "
                  "bilinear resize; each branch map independently normalized.",
        "scope": ("Three CNN-branch conditional maps, differentiated through the learned teacher "
                  "gate and equal teacher-student logit average; no outer gate or conflict brake "
                  "in this inference graph. Not a joint map or ViT attribution."
                  if args.inference == "equal" else
                  "Three CNN-branch conditional maps, differentiated through teacher/outer gates "
                  "and active conflict-brake branch; not a joint map or ViT attribution. "
                  "Hard argmax/threshold decisions are locally fixed."),
        "limitations": "Qualitative only, no lesion annotations, no localization accuracy or "
                       "causal faithfulness claim; normalization prevents comparing branch magnitude.",
        "records": records,
    })
    if hashes != {str(path): sha256(path) for path in sources}:
        raise RuntimeError("Source archive, code or checkpoint changed during evaluation")
    write_json(args.output / "completion.json", {
        "status": "complete", "source_hashes_unchanged": True,
        "all_three_seeds": True, "gradcam_examples": len(records)})
    print(json.dumps(summary["mean_sd"], indent=2), flush=True)
    print(f"Completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
