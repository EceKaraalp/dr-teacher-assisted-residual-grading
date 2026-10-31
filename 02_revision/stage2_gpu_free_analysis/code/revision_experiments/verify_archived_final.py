"""Read-only checkpoint evaluation; never trains or modifies archived artifacts."""

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torchvision.models import convnext_tiny, resnet50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def no_download_convnext(*args, **kwargs):
    kwargs["weights"] = None
    return convnext_tiny(*args, **kwargs)


def no_download_resnet(*args, **kwargs):
    kwargs.pop("pretrained", None)
    kwargs["weights"] = None
    return resnet50(*args, **kwargs)


def no_download_timm(*args, **kwargs):
    kwargs["pretrained"] = False
    return TIMM_CREATE_MODEL(*args, **kwargs)


TIMM_CREATE_MODEL = timm.create_model


def build_model(archive: Path) -> nn.Module:
    package = archive / "TS_ConvNeXtTiny_Residual_Package"
    baseline_module = load_module("archive_baseline", package / "models" / "hybrid_cnn_vit_base.py")
    advanced_module = load_module("archive_advanced", package / "models" / "advanced_hybrid_models.py")
    notebook = json.loads((archive / "train_lowdata_hybrid_distill.ipynb").read_text(encoding="utf-8"))
    classes = {}
    wanted = {"ConvNeXtTinyStudent", "HeavyTeacherEnsemble", "ResidualTeacherStudentModel"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not any(f"class {name}(" in source for name in wanted):
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.ClassDef) and node.name in wanted:
                classes[node.name] = node
    if set(classes) != wanted:
        raise ValueError(f"Missing or ambiguous archived model definitions: {set(classes)}")
    namespace = {"torch": torch, "nn": nn, "F": F, "convnext_tiny": no_download_convnext}
    tree = ast.Module(body=list(classes.values()), type_ignores=[])
    exec(compile(tree, "archived_model_classes", "exec"), namespace)
    baseline = baseline_module.HybridCNNViTBase(
        num_classes=5,
        config={
            "fusion_method": "gate",
            "use_uncertainty_refinement": False,
            "use_ordinal_head": False,
            "use_prototype_memory": False,
        },
    )
    # Full state_dict is loaded strictly afterwards; pretraining downloads are unnecessary.
    with patch.object(advanced_module, "resnet50", no_download_resnet), patch.object(
        timm, "create_model", no_download_timm
    ):
        advanced = advanced_module.AdvancedHybridModel(
            num_classes=5, config={"fusion_method": "concat", "use_spectral_norm": True}
        )
    teacher = namespace["HeavyTeacherEnsemble"](baseline, advanced)
    student = namespace["ConvNeXtTinyStudent"](num_classes=5)
    return namespace["ResidualTeacherStudentModel"](
        teacher, student, residual_scale=0.70, use_conflict_brake=True,
        conflict_threshold=0.82, conflict_gmax=0.66,
    )


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    if probabilities.shape != (len(labels), 5) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability shape or nonfinite values")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probabilities do not sum to one")
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    bins = np.minimum((confidence * 15).astype(int), 14)
    ece = sum(
        float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
        for i in range(15) if (mask := bins == i).any()
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(labels, predictions, weights="quadratic")),
        "roc_auc": float(roc_auc_score(labels, probabilities, multi_class="ovr", average="weighted")),
        "nll": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
        "brier_multiclass_sum": float(((probabilities - np.eye(5)[labels]) ** 2).sum(axis=1).mean()),
        "ece_15_equal_width": ece,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=range(5)).tolist(),
    }


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def test_data(repo: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(repo / "APTOS2019" / "train.csv")
    if len(df) != 3662 or df["id_code"].duplicated().any():
        raise ValueError("Unexpected APTOS CSV size or duplicate IDs")
    with np.load(repo / "APTOS2019" / "cache_224.npz") as cache:
        images, labels = cache["images"], cache["labels"]
    if images.shape != (3662, 224, 224, 3) or images.dtype != np.uint8:
        raise ValueError("Unexpected cache shape/type")
    if not np.array_equal(labels, df["diagnosis"].to_numpy()):
        raise ValueError("Cache labels disagree with CSV order")
    _, indices = train_test_split(np.arange(len(df)), test_size=0.1, random_state=42, stratify=labels)
    ids = df.iloc[indices]["id_code"].tolist()
    manifest = json.loads((repo / "results" / "split_manifest.json").read_text(encoding="utf-8"))
    if set(ids) != set(manifest["sample_ids"]["test"]):
        raise ValueError("Reconstructed test IDs disagree with manifest")
    # Verify CSV/cache image correspondence for every evaluated image.
    from PIL import Image

    for index in indices:
        path = repo / "APTOS2019" / "train_images" / f"{df.iloc[index]['id_code']}.png"
        with Image.open(path) as image:
            resized = np.array(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
        if not np.array_equal(resized, images[index]):
            raise ValueError(f"Cache pixels do not match resized original: {path}")
    return images[indices], labels[indices].astype(int), ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(r"D:\m1\Documents"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    checkpoint_paths = [args.archive / f"seed {s}" / "best_model.pth" for s in (42, 52, 62)]
    source_paths = [
        args.archive / "train_lowdata_hybrid_distill.ipynb",
        args.repo / "APTOS2019" / "train.csv",
        args.repo / "APTOS2019" / "cache_224.npz",
        args.repo / "results" / "split_manifest.json",
        Path(__file__),
        *sorted((args.archive / "TS_ConvNeXtTiny_Residual_Package").rglob("*.py")),
    ]
    provenance = {
        "python": sys.version, "torch": torch.__version__, "timm": timm.__version__,
        "numpy": np.__version__, "platform": platform.platform(), "device": "cpu",
        "threads": args.threads, "training": False, "pretrained_downloads": False,
        "input": "RGB bilinear 224x224 float32 0..255, no normalization",
        "test_ids_provenance": "Reconstructed split seed 42, matched post-hoc manifest; not a training-time ID log",
        "source_sha256": {str(p): sha256(p) for p in source_paths},
        "checkpoint_sha256": {str(p): sha256(p) for p in checkpoint_paths},
        "original_server_hash_comparison": "Not available; these are hashes of local copies only",
        "runtime_note": "CPU validation only, not a manuscript GPU latency benchmark",
        "fusion_settings": {"residual_scale": 0.70, "conflict_threshold": 0.82, "conflict_gmax": 0.66},
    }
    write_json(args.output / "provenance.json", provenance)
    print("Loading and checking all 367 test images", flush=True)
    images, labels, ids = test_data(args.repo)
    print("Test pixels, labels and split verified", flush=True)
    model = build_model(args.archive)
    all_results = {}
    for seed, checkpoint_path in zip((42, 52, 62), checkpoint_paths):
        print(f"Seed {seed}: loading weights_only=True, strict=True", flush=True)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        model.eval()
        captured = {}

        def capture_base(module, inputs, output):
            captured["base"] = output

        def capture_advanced(module, inputs, output):
            captured["advanced"] = output[0]

        h1 = model.teacher_model.baseline_model.register_forward_hook(capture_base)
        h2 = model.teacher_model.adv_model.register_forward_hook(capture_advanced)
        outputs = {}
        try:
            with torch.inference_mode():
                for start in range(0, len(images), 16):
                    batch = torch.from_numpy(images[start:start + 16]).permute(0, 3, 1, 2).float()
                    final, extra = model(batch)
                    raw_teacher = (captured["base"] + captured["advanced"]) / 2
                    logits = {
                        "archived_residual": final,
                        "student_only": extra["z_student"],
                        "archived_teacher": extra["z_teacher"],
                        "teacher1_embedded": captured["base"],
                        "teacher2_embedded": captured["advanced"],
                        "equal_ts_archived_teacher": (extra["z_teacher"] + extra["z_student"]) / 2,
                        "manuscript_equal_logits": (raw_teacher + extra["z_student"]) / 2,
                    }
                    for mode, values in logits.items():
                        outputs.setdefault(mode, []).append(F.softmax(values, dim=1).cpu().numpy())
                    print(f"Seed {seed}: {min(start + 16, len(images))}/367", flush=True)
        finally:
            h1.remove()
            h2.remove()
        arrays = {mode: np.concatenate(parts).astype(np.float64) for mode, parts in outputs.items()}
        results = {mode: metrics(labels, values) for mode, values in arrays.items()}
        expected = json.loads((args.archive / f"seed {seed}" / "test_metrics.json").read_text(encoding="utf-8"))
        results["archived_reference_comparison"] = {
            "confusion_matrix_equal": results["archived_residual"]["confusion_matrix"] == expected["confusion_matrix"],
            "metric_deltas": {k: results["archived_residual"][k] - expected[k] for k in ("accuracy", "f1_macro", "qwk", "roc_auc")},
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "expected_parameter_count": expected["param_total"],
        }
        ref_pred = arrays["archived_residual"].argmax(axis=1)
        results["label_disagreement_with_archived_path"] = {
            mode: int((values.argmax(axis=1) != ref_pred).sum()) for mode, values in arrays.items()
        }
        np.savez_compressed(args.output / f"seed_{seed}_probabilities.npz", labels=labels, ids=np.array(ids), **arrays)
        with (args.output / f"seed_{seed}_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id_code", "diagnosis", *arrays])
            predictions = {mode: values.argmax(axis=1) for mode, values in arrays.items()}
            for i, image_id in enumerate(ids):
                writer.writerow([image_id, int(labels[i]), *[int(predictions[m][i]) for m in arrays]])
        write_json(args.output / f"seed_{seed}_metrics.json", results)
        all_results[str(seed)] = results
        print(f"Seed {seed} completed: {results['archived_reference_comparison']}", flush=True)
    write_json(args.output / "summary.json", all_results)
    print(f"Completed. Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
