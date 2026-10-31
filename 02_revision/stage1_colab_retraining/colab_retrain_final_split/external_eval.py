"""Evaluation-only external dataset assessment; does not train or tune models.

Expected input: labels.csv with id_code, diagnosis columns and an image directory
containing files whose names begin with id_code. Diagnoses must be 0..4.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import dr_common as C
import dr_pipeline as P


class ExternalDRDataset(Dataset):
    def __init__(self, csv_path, image_dir, id_column="id_code", label_column="diagnosis"):
        self.frame = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        missing = {id_column, label_column} - set(self.frame.columns)
        if missing:
            raise ValueError(f"External labels CSV is missing required columns: {sorted(missing)}")
        self.ids = self.frame[id_column].astype(str).tolist()
        self.labels = self.frame[label_column].astype(int).to_numpy()
        if len(self.labels) == 0 or not set(np.unique(self.labels)).issubset(set(range(C.NUM_CLASSES))):
            raise ValueError("External diagnoses must be non-empty integer labels in [0, 4]")
        self.image_paths = []
        for sample_id in self.ids:
            matches = sorted(self.image_dir.glob(f"{sample_id}*"))
            if not matches:
                raise FileNotFoundError(f"No image found for ID {sample_id!r} in {self.image_dir}")
            self.image_paths.append(matches[0])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]) as image:
            resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            x = torch.from_numpy(np.asarray(resized, dtype=np.uint8).copy()).permute(2, 0, 1).float()
        y = torch.zeros(C.NUM_CLASSES, dtype=torch.float32)
        y[self.labels[index]] = 1.0
        return x, y


def evaluate(dataset, loader, base_ckpt, teacher_ckpts, student_ckpt, arm, seed, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_modes = {
        "teacher_gate__student_gated": ("gate", "gated"),
        "teacher_gate__student_equal": ("gate", "equal"),
        "teacher_equalprob__student_gated": ("equal_prob", "gated"),
        "teacher_equalprob__student_equal": ("equal_prob", "equal"),
        "teacher_equallogit__student_gated": ("equal_logits", "gated"),
        "teacher_equallogit__student_equal": ("equal_logits", "equal"),
        "teacher_t1__student_gated": ("t1", "gated"),
        "teacher_t1__student_equal": ("t1", "equal"),
        "teacher_t2__student_gated": ("t2", "gated"),
        "teacher_t2__student_equal": ("t2", "equal"),
        "teacher_gate__student_gated_aug": ("gate", "gated"),
    }
    if arm not in arm_modes:
        raise ValueError(f"Unknown student arm: {arm}")
    teacher_mode, mix = arm_modes[arm]
    teacher_runs = {}
    for mode in ("gate", "equal_prob", "equal_logits"):
        teacher_runs[mode] = P.collect_outputs(
            P.load_teacher(base_ckpt, teacher_ckpts[mode], fusion_mode=mode), loader, "teacher"
        )
    for mode in ("t1", "t2"):
        teacher_runs[mode] = P.collect_outputs(
            P.load_teacher(base_ckpt, teacher_ckpts["gate"], fusion_mode=mode), loader, "teacher"
        )
    tout = teacher_runs[teacher_mode]
    system = P.ResidualTeacherStudentModel(
        P.load_teacher(base_ckpt, teacher_ckpts[teacher_mode] if teacher_mode in teacher_ckpts
                       else teacher_ckpts["gate"], fusion_mode=teacher_mode),
        P.ConvNeXtTinyStudent(5, pretrained=False), mix=mix, teacher_mode=teacher_mode,
        **P.STUDENT_MODEL_KW,
    ).to(P.device)
    P.load_student_checkpoint(system, student_ckpt)
    sout = P.collect_outputs(system, loader, "student")
    y = dataset.labels
    logits = {
        "T1_baseline": teacher_runs["gate"]["z_t1"],
        "T2_SpectralNorm": tout["z_t2"],
        "teacher_learned_gate": teacher_runs["gate"]["z_teacher"],
        "teacher_equal_probability": teacher_runs["equal_prob"]["z_teacher"],
        "teacher_equal_logits": teacher_runs["equal_logits"]["z_teacher"],
        f"system_{arm}": sout["z_final"],
    }
    if mix == "gated":
        logits["system_inference_equal_weight"] = (sout["z_teacher"] + sout["z_student"]) / 2.0

    report = {
        "external_dataset": {
            "n": len(dataset),
            "student_arm": arm,
            "seed": seed,
            "checkpoint_selection": "validation QWK on APTOS2019 only",
            "class_counts": {str(int(c)): int(n) for c, n in zip(*np.unique(y, return_counts=True))},
            "label_mapping": "0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative",
            "image_preprocessing": "RGB, 224x224 PIL bilinear, raw pixel range 0..255",
            "training_or_tuning_on_external_data": False,
        },
        "models": {},
    }
    logits["teacher_learned_probability_gate"] = teacher_runs["gate"]["z_teacher"]
    for name, z in logits.items():
        prob = torch.softmax(torch.as_tensor(z, dtype=torch.float64), dim=1).numpy()
        pred = prob.argmax(1)
        report["models"][name] = C.detailed_metrics(y, pred, prob)
        pred_frame = {"sample_id": dataset.ids, "y_true": y, "y_pred": pred}
        pred_frame.update({f"p_class_{c}": prob[:, c] for c in range(C.NUM_CLASSES)})
        pd.DataFrame(pred_frame).to_csv(output_dir / f"predictions_{name}.csv", index=False)
    C.write_json(report, output_dir / "metrics.json")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="External labels CSV with id_code and diagnosis")
    parser.add_argument("--images", required=True, help="Directory containing the external image files")
    parser.add_argument("--out", default="results_retrain/external")
    parser.add_argument("--results-root", default="results_retrain")
    parser.add_argument("--arm", default="teacher_gate__student_gated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-column", default="id_code")
    parser.add_argument("--label-column", default="diagnosis")
    args = parser.parse_args()

    dataset = ExternalDRDataset(args.csv, args.images, args.id_column, args.label_column)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    out = Path(args.out)
    root = Path(args.results_root)
    base_ckpt = root / "teachers" / P.BASELINE_NAME / "best_model.pth"
    teacher_ckpts = {
        mode: root / "teachers" / f"fusion_{mode}" / P.TEACHER_NAME / "best_model.pth"
        for mode in ("gate", "equal_prob", "equal_logits")
    }
    teacher_ckpts["t1"] = teacher_ckpts["gate"]
    teacher_ckpts["t2"] = teacher_ckpts["gate"]
    student_ckpt = root / "students" / args.arm / f"seed_{args.seed}" / "best_model.pth"
    for path in (base_ckpt, *teacher_ckpts.values(), student_ckpt):
        if not path.exists():
            raise FileNotFoundError(f"Required trained checkpoint does not exist: {path}")
    evaluate(dataset, loader, base_ckpt, teacher_ckpts, student_ckpt, args.arm, args.seed, out)


if __name__ == "__main__":
    main()
