"""Grad-CAM for the final system. Target layer: last ConvNeXt stage of the student; target = predicted class of z_final.
Usage: python gradcam.py (env: DATA_DIR, OUT_DIR, ARM=teacher_gate__student_gated, SEED=42, N_PER_CLASS=3)"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import dr_common as C
import dr_pipeline as P

DATA_DIR = os.environ.get("DATA_DIR", "APTOS2019")
OUT = Path(os.environ.get("OUT_DIR", "results_retrain"))
ARM, SEED, NPC = os.environ.get("ARM", "teacher_gate__student_gated"), int(os.environ.get("SEED", "42")), int(os.environ.get("N_PER_CLASS", "3"))
ck = OUT / "students" / ARM / f"seed_{SEED}" / "best_model.pth"
base_ck = OUT / "teachers" / P.BASELINE_NAME / "best_model.pth"
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
    "teacher_equallogit__student_equal_aug": ("equal_logits", "equal"),
}
if ARM not in arm_modes:
    raise ValueError(f"Unknown student arm: {ARM}")
teacher_mode, mix = arm_modes[ARM]
t2_ck = OUT / "teachers" / f"fusion_{teacher_mode if teacher_mode in {'gate', 'equal_prob', 'equal_logits'} else 'gate'}" / P.TEACHER_NAME / "best_model.pth"
model = P.ResidualTeacherStudentModel(
    P.load_teacher(base_ck, t2_ck, fusion_mode=teacher_mode), P.ConvNeXtTinyStudent(5, pretrained=False),
    mix=mix, teacher_mode=teacher_mode, **P.STUDENT_MODEL_KW,
).to(P.device)
P.load_student_checkpoint(model, ck)
model.eval()

acts, grads = {}, {}
layer = model.student_model.backbone.features[-1]

def save_activation(module, inputs, output):
    acts["a"] = output
    output.register_hook(lambda grad: grads.__setitem__("g", grad))


layer.register_forward_hook(save_activation)

d = C.get_data(DATA_DIR, batch_size=1, augment_train=False)
imgs, y = d["images"][d["indices"]["test"]], d["labels"]["test"]
names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
(OUT / "gradcam").mkdir(parents=True, exist_ok=True)
rng = np.random.RandomState(0)
picks = []
for c in range(5):
    idx = np.where(y == c)[0]
    picks += list(rng.choice(idx, min(NPC, len(idx)), replace=False))

fig, ax = plt.subplots(len(picks), 2, figsize=(5, 2.5 * len(picks)))
ax = np.atleast_2d(ax)
case_rows = []
for r, i in enumerate(picks):
    img = torch.as_tensor(imgs[i]).float()
    x = (img if img.shape[0] == 3 else img.permute(2, 0, 1)).unsqueeze(0).to(P.device).requires_grad_(True)
    model.zero_grad()
    z, _ = model(x)
    pred = int(z.argmax(1))
    prob = float(torch.softmax(z, dim=1)[0, pred].detach().cpu())
    z[0, pred].backward()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).detach().cpu().numpy()
    show = (x[0].detach().cpu().permute(1, 2, 0).numpy() / 255.0).clip(0, 1)
    ax[r, 0].imshow(show); ax[r, 0].set_title(f"true: {names[y[i]]}", fontsize=8); ax[r, 0].axis("off")
    ax[r, 1].imshow(show); ax[r, 1].imshow(cam, cmap="jet", alpha=0.45)
    ax[r, 1].set_title(f"pred: {names[pred]} ({prob:.2f})", fontsize=8); ax[r, 1].axis("off")
    case_rows.append({"sample_id": str(d["ids"]["test"][i]), "y_true": int(y[i]), "y_pred": pred,
                      "predicted_class_probability": prob, "correct": bool(pred == y[i]),
                      "cam_target": names[pred]})
plt.tight_layout()
plt.savefig(OUT / "gradcam" / f"gradcam_{ARM}_seed{SEED}.png", dpi=200)
import pandas as pd
pd.DataFrame(case_rows).to_csv(OUT / "gradcam" / f"gradcam_{ARM}_seed{SEED}_cases.csv", index=False)
print("saved", OUT / "gradcam")
