"""Round-2 fixed-checkpoint external evaluation (IDRiD, optional Messidor-2 / DDR) and gated-output Grad-CAM."""
import argparse
import csv
import io
from pathlib import Path
import re
from unittest.mock import patch
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from analyze import bootstrap, metrics, probabilities
from data import Images, file_hash, load_data
from external import load_test
from plan import SEEDS
from plan2 import Job, jobs
import run as runner
from run2 import ROOT, build, copy_verified, device

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def read_labels(path, kind):
    """Return {image stem (lowercase): grade}; ungradable/out-of-range grades are excluded and counted."""
    text = Path(path).read_text(encoding="utf-8-sig")
    labels, excluded = {}, {"ungradable_or_invalid": 0}
    if kind == "ddr":
        for line in text.splitlines():
            if not line.strip():
                continue
            name, grade = line.split()[:2]
            grade = int(grade)
            if grade not in range(5):
                excluded["ungradable_or_invalid"] += 1
                continue
            labels[Path(name).stem.lower()] = grade
        return labels, excluded
    rows = list(csv.DictReader(io.StringIO(text)))
    columns = {c.lower(): c for c in rows[0]}
    image_column = next(columns[c] for c in ("image", "image_id", "id_code", "image name") if c in columns)
    if kind == "messidor2":
        grade_column, gradable_column = columns["adjudicated_dr_grade"], columns.get("adjudicated_gradable")
    else:
        grade_column, gradable_column = columns["grade"], columns.get("gradable")
    for row in rows:
        raw = str(row[grade_column]).strip()
        gradable = True if gradable_column is None else str(row[gradable_column]).strip() in {"1", "1.0", "True", "true"}
        if not gradable or not re.fullmatch(r"\d+(\.0)?", raw) or int(float(raw)) not in range(5):
            excluded["ungradable_or_invalid"] += 1
            continue
        labels[Path(row[image_column]).stem.lower()] = int(float(raw))
    return labels, excluded


def load_external(images_path, labels_path, kind):
    labels, excluded = read_labels(labels_path, kind)
    images_path = Path(images_path)
    if images_path.is_dir():
        found = {p.stem.lower(): p for p in images_path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
        reader = lambda item: item.read_bytes()
    else:
        archive = zipfile.ZipFile(images_path)
        found = {Path(n).stem.lower(): n for n in archive.namelist()
                 if n.lower().endswith(IMAGE_SUFFIXES) and not n.startswith("__MACOSX")}
        reader = archive.read
    missing = sorted(set(labels) - set(found))
    if missing:
        raise ValueError(f"{len(missing)} labelled images not found, e.g. {missing[:5]}")
    ids = sorted(labels)
    images = []
    for image_id in ids:
        with Image.open(io.BytesIO(reader(found[image_id]))) as image:
            images.append(np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)))
    y = np.array([labels[i] for i in ids], dtype=np.int64)
    return np.stack(images), y, ids, {"evaluated": len(ids), "class_counts": np.bincount(y, minlength=5).tolist(),
                                      **excluded, "source_sha256": file_hash(images_path) if images_path.is_file() else None}


def external_jobs():
    return [job for job in jobs() if job.stage != "teacher1"]


def evaluate_dataset(name, images, labels, ids, output):
    destination = output / "external2" / name
    destination.mkdir(parents=True, exist_ok=True)
    report, skipped = {}, []
    for job in external_jobs():
        folder = output / job.key
        if not (folder / "complete.json").exists():
            skipped.append(job.key)
            continue
        checkpoint = runner.load_best(folder)
        model = build(job, output, checkpoint).eval()
        del checkpoint
        loader = DataLoader(Images(images, labels, np.arange(len(labels)), job.preprocessing), batch_size=16, shuffle=False)
        y, paths = runner.predict(model, loader, job.legacy)
        np.savez_compressed(destination / (job.key.replace("/", "__") + ".npz"), labels=y, ids=np.asarray(ids), **paths)
        report[job.key] = {path: metrics(y, probabilities(z)) for path, z in paths.items()}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    runner.write_json(destination / "metrics.json", {"metrics": report, "skipped_incomplete": skipped})
    runner.write_json(destination / "contrasts.json", contrasts(destination, labels))
    return report


def contrasts(destination, labels):
    def load(job, path):
        file = destination / (job.key.replace("/", "__") + ".npz")
        if not file.exists():
            return None
        with np.load(file) as values:
            return values[path].argmax(1) if path in values.files else None

    def triple(family, stage, arm, path):
        values = [load(Job(family, seed, stage, arm), path) for seed in SEEDS]
        return None if any(v is None for v in values) else values

    pairs = {
        "reference_gated_vs_ce": (("reference", "student", "joint_full", "primary"), ("reference", "student", "student_ce", "primary")),
        "reference_equal_vs_ce": (("reference", "student", "joint_full", "equal"), ("reference", "student", "student_ce", "primary")),
        "t2indep_gated_vs_reference_gated": (("t2indep", "student", "joint_full", "primary"), ("reference", "student", "joint_full", "primary")),
        "nosn_gated_vs_reference_gated": (("nosn", "student", "joint_full", "primary"), ("reference", "student", "joint_full", "primary")),
        "aug_gated_vs_reference_gated": (("aug", "student", "joint_full", "primary"), ("reference", "student", "joint_full", "primary")),
        "aug_ce_vs_reference_ce": (("aug", "student", "student_ce", "primary"), ("reference", "student", "student_ce", "primary")),
        "fusion_historical_vs_reference_gated": (("fusion", "fusion", "historical", "primary"), ("reference", "student", "joint_full", "primary")),
        "fusion_matched_vs_reference_gated": (("fusion", "fusion", "matched", "primary"), ("reference", "student", "joint_full", "primary")),
    }
    result = {}
    for name, (left, right) in pairs.items():
        a, b = triple(*left), triple(*right)
        result[name] = bootstrap(labels, a, b) if a and b else {"status": "unavailable (missing jobs)"}
    return result


def gated_gradcam(model, image, label, image_id, destination):
    """Grad-CAM for the predicted class of the GATED primary output model(x)[0] (round-1 version used the equal logit)."""
    targets = {
        "Teacher1_CNN": model.teacher_model.baseline_model.cnn_backbone,
        "Teacher2_ResNet": model.teacher_model.adv_model.cnn_features,
        "Student_ConvNeXt": model.student_model.backbone.features,
    }
    tensor = torch.tensor(image[None], device=device()).permute(0, 3, 1, 2).float()
    model.eval()
    with torch.no_grad():
        reference = model(tensor)[0]
    activations, handles = {}, []

    def hook(name):
        def capture(module, inputs, value):
            activations[name] = value
        return capture

    for name, layer in targets.items():
        handles.append(layer.register_forward_hook(hook(name)))
    try:
        tensor.requires_grad_(True)
        with patch.object(torch, "no_grad", torch.enable_grad):
            logits = model(tensor)[0]
        torch.testing.assert_close(logits.detach(), reference, atol=1e-4, rtol=1e-5)
        predicted = int(reference.argmax(1))
        gradients = torch.autograd.grad(logits[0, predicted], list(activations.values()), allow_unused=True)
        maps, status = {}, {}
        for (name, activation), gradient in zip(activations.items(), gradients):
            if gradient is None:
                gradient = torch.zeros_like(activation)
            if not torch.isfinite(activation).all() or not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Nonfinite attribution: {image_id}/{name}")
            cam = (gradient.mean((2, 3), keepdim=True) * activation).sum(1, keepdim=True).relu()
            cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
            maximum = float(cam.max().detach())
            maps[name] = (cam / maximum if maximum else cam).detach().cpu().numpy()
            status[name] = {"positive_maximum": maximum, "no_positive_map": maximum == 0}
        np.savez_compressed(destination / f"{image_id}_gated_gradcam.npz", **maps)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(image)
        axes[0].set_title(f"{image_id}\nTrue {label}; gated predicted {predicted}")
        for axis, (name, cam) in zip(axes[1:], maps.items()):
            axis.imshow(image)
            if not status[name]["no_positive_map"]:
                axis.imshow(cam, cmap="jet", alpha=.45, vmin=0, vmax=1)
            axis.set_title(name + ("\nNo positive map" if status[name]["no_positive_map"] else ""))
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(destination / f"{image_id}_gated_gradcam.png", dpi=160)
        plt.close(fig)
        return {"id": image_id, "label": int(label), "gated_prediction": predicted, "branches": status,
                "forward_max_abs_difference": float((logits.detach() - reference).abs().max())}
    finally:
        for handle in handles:
            handle.remove()


def gradcam_cases(output, idrid, aptos):
    destination = output / "external2" / "gradcam_gated"
    destination.mkdir(parents=True, exist_ok=True)
    job = Job("reference", 42, "student", "joint_full")
    checkpoint = runner.load_best(output / job.key)
    model = build(job, output, checkpoint).eval()
    del checkpoint
    cases = []
    for source, (xs, ys, names) in (("IDRiD", idrid), ("APTOS", aptos)):
        for grade in range(5):
            index = np.flatnonzero(ys == grade)[0]
            cases.append(gated_gradcam(model, xs[index], ys[index], f"{source}_{names[index]}", destination))
    runner.write_json(destination / "gradcam_manifest.json", {
        "model": job.key, "target": "predicted class of the gated primary output model(x)[0]",
        "selection": "First test image of each true grade in fixed dataset order (same cases as round 1); seed 42 predefined.",
        "scope": "CNN-branch attribution only; qualitative; not validated lesion localization or ViT attribution.",
        "cases": cases,
    })
    del model


def persist(output, drive):
    if drive is None:
        return
    source = output / "external2"
    for path in source.rglob("*"):
        if path.is_file():
            target = drive / "external2" / path.relative_to(source)
            if not (target.exists() and file_hash(target) == file_hash(path)):
                copy_verified(path, target)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True, help="Official IDRiD 'B. Disease Grading.zip'.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drive", type=Path)
    parser.add_argument("--extra", nargs=4, action="append", default=[], metavar=("NAME", "IMAGES", "LABELS", "FORMAT"),
                        help="Optional dataset: images zip/folder, label file, format messidor2|ddr|generic.")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("External evaluation requires the Colab GPU.")
    provenance = {"preprocessing": "RGB bilinear 224 then each model's own training preprocessing; no TTA, "
                                   "no calibration, no threshold fitting, no target-domain training", "datasets": {}}
    images, labels, ids = load_test(args.zip)
    evaluate_dataset("IDRiD", images, labels, ids, args.output)
    provenance["datasets"]["IDRiD"] = {"evaluated": 103, "zip_sha256": file_hash(args.zip)}
    bundle = load_data(args.data, ROOT / "split_manifest.json")
    test = bundle[3]["test"]
    gradcam_cases(args.output, (images, labels, np.asarray(ids)), (bundle[1][test], bundle[2][test], bundle[0][test]))
    del bundle
    for name, images_path, labels_path, kind in args.extra:
        if kind not in {"messidor2", "ddr", "generic"}:
            raise ValueError(f"Unknown format {kind}")
        x, y, extra_ids, info = load_external(images_path, labels_path, kind)
        evaluate_dataset(name, x, y, extra_ids, args.output)
        provenance["datasets"][name] = {**info, "format": kind, "labels_sha256": file_hash(labels_path)}
        del x
    runner.write_json(args.output / "external2" / "provenance.json", provenance)
    persist(args.output, args.drive)
    print("EXTERNAL2 OK", list(provenance["datasets"]), flush=True)


if __name__ == "__main__":
    main()
