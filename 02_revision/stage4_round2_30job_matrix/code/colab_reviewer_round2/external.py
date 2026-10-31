"""Fixed-checkpoint IDRiD evaluation and descriptive CNN-branch Grad-CAM."""
import argparse
import csv
import io
from pathlib import Path
from unittest.mock import patch
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from analyze import metrics, probabilities
from data import file_hash, load_data
from plan import Run, SEEDS, matrix
from run import DEVICE, ROOT, build_model, load_best, predict, synchronize, write_json


def load_test(path):
    root = "B. Disease Grading/"
    label_path = root + "2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    image_root = root + "1. Original Images/b. Testing Set/"
    with zipfile.ZipFile(path) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read(label_path).decode("utf-8-sig"))))
        ids = [r["Image name"] for r in rows]
        labels = np.array([int(r["Retinopathy grade"]) for r in rows])
        names = {Path(n).stem: n for n in archive.namelist()
                 if n.startswith(image_root) and n.lower().endswith((".jpg", ".jpeg", ".png"))}
        if len(ids) != 103 or len(set(ids)) != 103 or set(ids) != set(names):
            raise ValueError("Expected all 103 official IDRiD test images and matching DR labels.")
        if np.bincount(labels, minlength=5).tolist() != [34, 5, 32, 19, 13]:
            raise ValueError("Unexpected official test class counts.")
        images = []
        for image_id in ids:
            with Image.open(io.BytesIO(archive.read(names[image_id]))) as image:
                images.append(np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)))
    return np.stack(images), labels, ids


def equal_logits(model, tensor):
    teacher, _ = model.teacher_model(tensor)
    student, _ = model.student_model(tensor)
    return (teacher + student) / 2


def gradcam(model, image, label, image_id, destination):
    targets = {
        "Teacher1_CNN": model.teacher_model.baseline_model.cnn_backbone,
        "Teacher2_ResNet": model.teacher_model.adv_model.cnn_features,
        "Student_ConvNeXt": model.student_model.backbone.features,
    }
    tensor = torch.tensor(image[None], device=DEVICE).permute(0, 3, 1, 2).float()
    model.eval()
    with torch.no_grad():
        reference = equal_logits(model, tensor)
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
            logits = equal_logits(model, tensor)
        torch.testing.assert_close(logits.detach(), reference, atol=1e-4, rtol=1e-5)
        predicted = int(reference.argmax(1))
        gradients = torch.autograd.grad(logits[0, predicted], list(activations.values()))
        maps, status = {}, {}
        for (name, activation), gradient in zip(activations.items(), gradients):
            if not torch.isfinite(activation).all() or not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Nonfinite attribution: {image_id}/{name}")
            cam = (gradient.mean((2, 3), keepdim=True) * activation).sum(1, keepdim=True).relu()
            cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
            maximum = float(cam.max().detach())
            maps[name] = (cam / maximum if maximum else cam).detach().cpu().numpy()
            status[name] = {"positive_maximum": maximum, "no_positive_map": maximum == 0}
        np.savez_compressed(destination / f"{image_id}_gradcam.npz", **maps)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(image)
        axes[0].set_title(f"{image_id}\nTrue {label}; predicted {predicted}")
        for axis, (name, cam) in zip(axes[1:], maps.items()):
            axis.imshow(image)
            if not status[name]["no_positive_map"]:
                axis.imshow(cam, cmap="jet", alpha=.45, vmin=0, vmax=1)
            axis.set_title(name + ("\nNo positive map" if status[name]["no_positive_map"] else ""))
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(destination / f"{image_id}_gradcam.png", dpi=160)
        plt.close(fig)
        return {"id": image_id, "label": int(label), "prediction": predicted, "branches": status,
                "forward_max_abs_difference": float((logits.detach() - reference).abs().max())}
    finally:
        for handle in handles:
            handle.remove()


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("External evaluation requires the Colab GPU.")
    if any(not (args.output / r.key / "complete.json").exists() for r in matrix()):
        raise RuntimeError("Complete the frozen training matrix before external testing.")
    images, labels, ids = load_test(args.zip)
    loader = DataLoader(TensorDataset(torch.from_numpy(images).permute(0, 3, 1, 2).float(),
                                     torch.tensor(labels)), batch_size=16, shuffle=False)
    destination = args.output / "external"
    destination.mkdir(exist_ok=True)
    report, attributions = {}, []
    for seed in SEEDS:
        for arm in ("joint_full", "student_ce"):
            run = Run("raw", 1, seed, "student", arm)
            checkpoint = load_best(args.output / run.key)
            model = build_model(run, args.output, checkpoint).eval()
            del checkpoint
            y, paths = predict(model, loader, run)
            np.savez_compressed(destination / f"{run.group}_{arm}.npz", labels=y, ids=np.asarray(ids), **paths)
            report[f"{seed}/{arm}"] = {name: metrics(y, probabilities(z)) for name, z in paths.items()}
            if seed == 42 and arm == "joint_full":
                bundle = load_data(args.data, ROOT / "split_manifest.json")
                for source, xs, ys, names in (
                    ("IDRiD", images, labels, np.asarray(ids)),
                    ("APTOS", bundle[1][bundle[3]["test"]], bundle[2][bundle[3]["test"]], bundle[0][bundle[3]["test"]]),
                ):
                    for grade in range(5):
                        index = np.flatnonzero(ys == grade)[0]
                        attributions.append(gradcam(model, xs[index], ys[index], f"{source}_{names[index]}", destination))
                del bundle
            del model
            torch.cuda.empty_cache()
    write_json(destination / "metrics.json", report)
    write_json(destination / "gradcam_manifest.json", {
        "selection": "First test image of each true grade in fixed dataset order; seed 42 predefined.",
        "scope": "CNN branch attribution only; qualitative, not validated lesion localization or ViT attribution.",
        "cases": attributions,
    })
    write_json(destination / "provenance.json", {
        "idrid_zip_sha256": file_hash(args.zip), "official_test_count": 103, "training_images_used": 0,
        "target_adaptation": False, "preprocessing": "RGB bilinear 224; float32 0..255; no TTA/calibration",
        "limitations": "Previously examined external dataset; not prospective/fresh multi-hospital validation. Both datasets from India.",
        "checkpoints": {str(seed): file_hash(args.output / Run("raw", 1, seed, "student", "joint_full").key / "best.pth")
                        for seed in SEEDS},
    })
    if args.drive:
        synchronize(destination, args.drive / "external")
        import shutil
        for path in destination.glob("*.png"):
            target = args.drive / "external" / path.name
            shutil.copy2(path, target)
            if file_hash(path) != file_hash(target):
                raise OSError(f"Figure persistence failed: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drive", type=Path)
    main(parser.parse_args())
