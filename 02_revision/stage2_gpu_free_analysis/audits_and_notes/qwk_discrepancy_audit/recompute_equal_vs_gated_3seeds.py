"""Read-only CPU recomputation of old-system outputs for seeds 42/52/62.

Loads the three server checkpoints (Documents/seed */best_model.pth), runs the frozen APTOS test split
and computes, from the SAME forward pass:
  gated        z_final = z_T + 0.70*g2*(z_S - mean z_S) with conflict brake   (reported 0.9186)
  equal        (z_T + z_S) / 2, z_T = learned-gate teacher output
  paper_eq     ((z1 + z2)/2 + z_S) / 2 with raw teacher logits (formula as written in the manuscript)
  teacher_gate z_T alone, student alone z_S
No file is modified; outputs go to this folder.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from torchvision.models import convnext_tiny

M1 = Path(r"D:\m1\Documents")
PACKAGE = M1 / "TS_ConvNeXtTiny_Residual_Package"
DATA = Path(r"<repo_root>\APTOS2019")
MANIFEST = Path(r"<repo_root>\colab_reviewer_round2\split_manifest.json")
OUT = Path(__file__).parent
sys.path.insert(0, str(PACKAGE))
from advanced_model_configs import get_advanced_model_config  # noqa: E402
from models.advanced_hybrid_models import AdvancedHybridModel  # noqa: E402
from models.hybrid_cnn_vit_base import HybridCNNViTBase  # noqa: E402

torch.set_num_threads(max(1, torch.get_num_threads()))
BASE_CONFIG = {"fusion_method": "gate", "use_uncertainty_refinement": False,
               "use_ordinal_head": False, "use_prototype_memory": False}


def entropy(p):
    p = p.clamp_min(1e-8)
    return -(p * torch.log(p)).sum(1, keepdim=True)


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.baseline_model = HybridCNNViTBase(num_classes=5, config=BASE_CONFIG)
        self.adv_model = AdvancedHybridModel(num_classes=5, config=get_advanced_model_config("Adv_Idea_5_SpectralNorm"))
        self.gate = nn.Sequential(nn.Linear(13, 64), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(64, 32),
                                  nn.ReLU(inplace=True), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        z1 = self.baseline_model(x)
        z2, _ = self.adv_model(x)
        p1, p2 = F.softmax(z1, 1), F.softmax(z2, 1)
        gap = (p1.max(1, keepdim=True).values - p2.max(1, keepdim=True).values).abs()
        g = self.gate(torch.cat([p1, p2, entropy(p1), entropy(p2), gap], 1))
        zt = torch.log(((1 - g) * p1 + g * p2).clamp_min(1e-8))
        return zt, z1, z2


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = convnext_tiny(weights=None)
        self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features, 5)

    def forward(self, x):
        return self.backbone(x), {}


class Full(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher_model = Teacher()
        self.student_model = Student()
        self.gate = nn.Sequential(nn.Linear(13, 32), nn.ReLU(inplace=True), nn.Linear(32, 1), nn.Sigmoid())

    @torch.no_grad()
    def forward(self, x):
        zt, z1, z2 = self.teacher_model(x)
        zs, _ = self.student_model(x)
        pt, ps = F.softmax(zt, 1), F.softmax(zs, 1)
        gap = (pt.max(1, keepdim=True).values - ps.max(1, keepdim=True).values).abs()
        g = self.gate(torch.cat([pt, ps, entropy(pt), entropy(ps), gap], 1))
        conf, tp = pt.max(1)
        mask = (conf > 0.82) & (tp != ps.argmax(1))
        cap = (0.66 * (1 - conf)).unsqueeze(1)
        g = torch.where(mask.unsqueeze(1), torch.minimum(g, cap), g)
        delta = zs - zs.mean(1, keepdim=True)
        return {"gated": zt + 0.70 * g * delta, "equal": (zt + zs) / 2, "paper_eq": ((z1 + z2) / 2 + zs) / 2,
                "teacher_gate": zt, "student_only": zs, "teacher1": z1, "teacher2": z2,
                "gate2": g.squeeze(1), "conflict": mask.float()}


def metrics(y, z):
    p = torch.tensor(z).softmax(1).numpy()
    pred = z.argmax(1)
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")),
            "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
            "roc_auc_ovr_macro": float(roc_auc_score(y, p, multi_class="ovr", average="macro")),
            "predicted_classes": int(len(np.unique(pred)))}


def main():
    frame = pd.read_csv(DATA / "train.csv")
    with np.load(DATA / "cache_224.npz") as cache:
        images, labels = cache["images"], cache["labels"].astype(int)
    ids = {str(v): i for i, v in enumerate(frame.id_code)}
    test = np.asarray([ids[str(v)] for v in json.loads(MANIFEST.read_text(encoding="utf-8"))["sample_ids"]["test"]])
    x_all, y = images[test], labels[test]
    result = {}
    for seed in (42, 52, 62):
        path = M1 / f"seed {seed}" / "best_model.pth"
        model = Full()
        print(seed, model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True), flush=True)
        model.eval()
        chunks = []
        for start in range(0, len(test), 16):
            x = torch.from_numpy(x_all[start:start + 16]).permute(0, 3, 1, 2).float()
            chunks.append({k: v.numpy() for k, v in model(x).items()})
            print(seed, start, flush=True) if start % 96 == 0 else None
        out = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        np.savez_compressed(OUT / f"recomputed_logits_seed{seed}.npz", labels=y, **out)
        recorded = json.loads((M1 / f"seed {seed}" / "test_metrics.json").read_text())
        result[seed] = {name: metrics(y, out[name]) for name in
                        ("gated", "equal", "paper_eq", "teacher_gate", "teacher1", "teacher2", "student_only")}
        result[seed]["recorded_test_metrics_json"] = {"accuracy": recorded["accuracy"], "qwk": recorded["qwk"]}
        result[seed]["same_labels_gated_vs_equal"] = float((out["gated"].argmax(1) == out["equal"].argmax(1)).mean())
        result[seed]["mean_gate2"] = float(out["gate2"].mean())
        result[seed]["conflict_rate"] = float(out["conflict"].mean())
        print(seed, json.dumps(result[seed]["gated"]), json.dumps(result[seed]["equal"]), flush=True)
        del model
    (OUT / "recomputed_equal_vs_gated.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
