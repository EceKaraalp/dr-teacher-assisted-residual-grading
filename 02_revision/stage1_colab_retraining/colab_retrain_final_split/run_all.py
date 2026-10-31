"""Stage runner. Usage (from the notebook or a shell):

    python run_all.py --stage teachers            # Teacher 1 + separately trained T2 gate/equal fusion objectives
    python run_all.py --stage students            # ConvNeXt student, arms x seeds
    python run_all.py --stage analyze             # per-sample outputs, tables, paired bootstrap

Environment: DATA_DIR (folder with train.csv [+ cache_224.npz]), OUT_DIR (local results), DRIVE_DIR (mirror target),
DR_SMOKE=1 for a tiny CPU/GPU dry run.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, f1_score

import dr_common as C
import dr_pipeline as P

DATA_DIR = os.environ.get("DATA_DIR", "APTOS2019")
OUT = Path(os.environ.get("OUT_DIR", "results_retrain"))
DRIVE = os.environ.get("DRIVE_DIR", "")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,52,62").split(",")]
ARMS = os.environ.get(
    "ARMS",
    "teacher_equallogit__student_equal",
).split(",")
MAX_EP = int(os.environ.get("DR_MAX_EPOCHS", "0")) or None  # smoke only

BASE_DIR = OUT / "teachers" / P.BASELINE_NAME


def _teacher_dir(fusion_mode):
    return OUT / "teachers" / f"fusion_{fusion_mode}" / P.TEACHER_NAME


def _teacher_ckpt(fusion_mode):
    return _teacher_dir(fusion_mode) / "best_model.pth"


BASE_CKPT = BASE_DIR / "best_model.pth"
T2_CKPT = _teacher_ckpt("gate")
TEACHER_MODES = ("gate", "equal_prob", "equal_logits")


def _data(augment=False):
    return C.get_data(DATA_DIR, batch_size=int(os.environ.get("BATCH", "16")), augment_train=augment)


def _mirror():
    if DRIVE:
        C.sync_dirs(OUT, DRIVE)


def stage_teachers(seed=42):
    d = _data(False)  # original teachers: NO augmentation
    audit = _split_audit(d)
    C.write_json({"env": C.env_report(), "split": audit, "seed": seed,
                  "class_weights": d["class_weights"], "augmentation": "none",
                  "baseline": {"name": P.BASELINE_NAME, "config": P.BASELINE_CONFIG,
                               "training": P.BASELINE_HP, "optimizer": "AdamW, weight_decay=1e-4",
                               "scheduler": "CosineAnnealingWarmRestarts(T_0=10,T_mult=2)",
                               "loss": "class-weighted cross entropy", "early_stopping": "validation QWK"},
                  "anchored_teachers": {
                      mode: {"name": P.TEACHER_NAME, "config": "Adv_Idea_5_SpectralNorm",
                             "fusion_mode": mode, "training": P.ANCHORED_HP,
                             "optimizer": "AdamW, weight_decay=1e-4",
                             "scheduler": "CosineAnnealingWarmRestarts(T_0=10,T_mult=2)",
                             "early_stopping": "validation QWK"}
                      for mode in TEACHER_MODES
                  }},
                 OUT / "teachers" / "env_and_split.json")
    C.write_json({k: v for k, v in d["ids"].items()}, OUT / "split_ids.json")

    b = P.BaselineTrainer(BASE_DIR, d["class_weights"], seed=seed)
    if not b.is_done():
        b.fit(d["train_loader"], d["val_loader"], max_epochs=MAX_EP)
        b.model.load_state_dict(torch.load(b.best_path, map_location=P.device, weights_only=True))
        z, y = b._predict_logits(d["test_loader"])
        C.write_json({"model": P.BASELINE_NAME, "seed": seed, **C.metrics_from_logits(y, z),
                      "best_val_qwk": b.best_qwk,
                      "best_epoch": int(np.argmax(b.history["val_qwk"]) + 1)}, b.metrics_path)
    _mirror()

    for fusion_mode in TEACHER_MODES:
        t = P.AnchoredTeacherTrainer(
            _teacher_dir(fusion_mode), BASE_CKPT, d["class_weights"], seed=seed, fusion_mode=fusion_mode,
        )
        if not t.is_done():
            t.fit(d["train_loader"], d["val_loader"], max_epochs=MAX_EP)
            t.model.load_state_dict(torch.load(t.best_path, map_location=P.device, weights_only=True))
            z, y = t._predict_logits(d["test_loader"])
            if fusion_mode == "equal_logits":
                probabilities = torch.softmax(torch.as_tensor(z), dim=1).numpy()
            else:
                probabilities = np.exp(z)
            m = C.compute_full_metrics(y, probabilities.argmax(1), probabilities)
            C.write_json({"model": P.TEACHER_NAME, "fusion_mode": fusion_mode, "seed": seed, **m,
                          "best_val_qwk": t.best_qwk,
                          "best_epoch": int(np.argmax(t.history["val_qwk"]) + 1)}, t.metrics_path)
        _mirror()
    _mirror()
    print("teachers done")


def _student_dir(arm, seed):
    return OUT / "students" / arm / f"seed_{seed}"


def _arm_spec(arm):
    specs = {
        "teacher_gate__student_gated": ("gate", "gated", False),
        "teacher_gate__student_equal": ("gate", "equal", False),
        "teacher_equalprob__student_gated": ("equal_prob", "gated", False),
        "teacher_equalprob__student_equal": ("equal_prob", "equal", False),
        "teacher_equallogit__student_gated": ("equal_logits", "gated", False),
        "teacher_equallogit__student_equal": ("equal_logits", "equal", False),
        "teacher_t1__student_gated": ("t1", "gated", False),
        "teacher_t1__student_equal": ("t1", "equal", False),
        "teacher_t2__student_gated": ("t2", "gated", False),
        "teacher_t2__student_equal": ("t2", "equal", False),
        "teacher_gate__student_gated_aug": ("gate", "gated", True),
        "teacher_equallogit__student_equal_aug": ("equal_logits", "equal", True),
    }
    try:
        return specs[arm]
    except KeyError as exc:
        raise ValueError(f"Unknown arm {arm!r}; supported arms: {sorted(specs)}") from exc


def stage_students():
    missing_teacher = [str(_teacher_ckpt(mode)) for mode in TEACHER_MODES if not _teacher_ckpt(mode).exists()]
    if not BASE_CKPT.exists() or missing_teacher:
        raise FileNotFoundError(f"Run --stage teachers first. Missing: {missing_teacher or [str(BASE_CKPT)]}")
    for arm in ARMS:
        teacher_mode, mix, aug = _arm_spec(arm)
        d = _data(augment=aug)
        for seed in SEEDS:
            run_dir = _student_dir(arm, seed)
            C.write_json({
                "arm": arm, "seed": seed, "teacher_fusion": teacher_mode, "student_fusion": mix,
                "augmentation": {
                    "enabled": aug, "horizontal_flip_p": 0.5 if aug else 0.0,
                    "vertical_flip_p": 0.5 if aug else 0.0, "rotation_degrees": [-20, 20] if aug else [0, 0],
                    "brightness_range": [0.9, 1.1] if aug else [1, 1],
                    "contrast_range": [0.9, 1.1] if aug else [1, 1],
                    "applied_to": "student training split only",
                },
                "split": _split_audit(d),
                "model": "ConvNeXt_Tiny_Residual",
                "student_model_kwargs": P.STUDENT_MODEL_KW,
                "student_loss": P.STUDENT_LOSS_CFG,
                "training": {**P.STUDENT_HP, "optimizer": "AdamW, weight_decay=1e-4",
                             "scheduler": "CosineAnnealingWarmRestarts(T_0=10,T_mult=2)",
                             "batch_size": int(os.environ.get("BATCH", "16")),
                             "checkpoint_selection": "validation QWK"},
                "pretraining": {"student_convnext": "ImageNet1K_V1",
                                "teacher2_resnet50": "ImageNet1K_V1",
                                "teacher2_vit": "pretrained timm ViT-B/16",
                                "teacher1": "initialized from model defaults, trained on APTOS"},
                "environment": C.env_report(),
            }, run_dir / "run_config.json")
            tr = P.StudentTrainer(
                run_dir, f"TS_ConvNeXtTiny_{arm}", BASE_CKPT,
                _teacher_ckpt(teacher_mode) if teacher_mode in TEACHER_MODES else T2_CKPT,
                d["class_weights"], seed=seed, mix=mix, teacher_mode=teacher_mode,
            )
            if tr.is_done():
                print(f"skip {arm} seed {seed} (done)")
                continue
            tr.fit(d["train_loader"], d["val_loader"], max_epochs=MAX_EP)
            P.load_student_checkpoint(tr.model, tr.best_path)
            z, y = tr._predict_logits(d["test_loader"])
            m = C.metrics_from_logits(y, z)
            tot, trn, frz = P.count_params(tr.model)
            C.write_json({"arm": arm, "teacher_fusion": teacher_mode, "student_fusion": mix,
                          "augmentation": aug, "seed": seed, **m, "best_val_qwk": tr.best_qwk, "epochs_run": len(tr.history["val_qwk"]),
                          "best_epoch": int(np.argmax(tr.history["val_qwk"]) + 1),
                          "param_total": tot, "param_trainable": trn, "param_frozen": frz}, tr.metrics_path)
            _mirror()
    print("students done")


# ------------------------------------------------------------------ analysis
def qwk(y, p):
    return float(cohen_kappa_score(y, p, weights="quadratic"))


def _split_audit(d):
    counts = {sp: len(ids) for sp, ids in d["ids"].items()}
    class_counts = {
        sp: {str(int(c)): int(n) for c, n in zip(*np.unique(d["labels"][sp], return_counts=True))}
        for sp in ("train", "val", "test")
    }
    audit = {"counts": counts, "class_counts": class_counts, "smoke_mode": C.SMOKE}
    if not C.SMOKE:
        if counts != {"train": 2929, "val": 366, "test": 367}:
            raise RuntimeError(f"Unexpected split sizes: {counts}")
        ref_path = Path(__file__).parent / "reference" / "split_manifest_original.json"
        with open(ref_path, encoding="utf-8") as f:
            reference = json.load(f)
        actual = {split: set(map(str, d["ids"][split])) for split in ("train", "val", "test")}
        expected = {split: set(map(str, reference["sample_ids"][split])) for split in ("train", "val", "test")}
        moved_from_val = expected["val"] - actual["val"]
        moved_to_train = actual["train"] - expected["train"]
        if actual["test"] != expected["test"]:
            raise RuntimeError("Manuscript-count split unexpectedly changed frozen test sample IDs")
        if (len(moved_from_val) != 1 or moved_from_val != moved_to_train
                or actual["val"] - expected["val"] or expected["train"] - actual["train"]):
            raise RuntimeError("Expected exactly one sample to move from frozen validation to training")
        audit["matches_reference_test_ids"] = True
        audit["one_validation_sample_moved_to_train"] = next(iter(moved_from_val))
        audit["split_seed"] = reference["split_seed"]
        audit["split_protocol"] = (
            f"{reference['split_protocol']}; then moved the first sample from the most frequent "
            "validation class into training to match manuscript counts; test IDs unchanged"
        )
    return audit


def paired_bootstrap(y, pa, pb, n=2000, seed=0):
    """Stratified paired percentile bootstrap; preserves all ordinal classes per resample."""
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    if not (len(y) == len(pa) == len(pb)) or len(y) == 0:
        raise ValueError("Paired bootstrap arrays must have the same non-zero length")
    rng = np.random.RandomState(seed)
    by_class = [np.flatnonzero(y == c) for c in range(C.NUM_CLASSES)]
    if any(len(i) == 0 for i in by_class):
        raise ValueError("Paired bootstrap requires every ordinal class in the evaluation split")
    diffs = {"qwk": [], "accuracy": [], "f1_macro": []}
    for _ in range(n):
        i = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in by_class])
        for metric, fn in (
            ("qwk", lambda a, b: qwk(a, b)),
            ("accuracy", lambda a, b: float((a == b).mean())),
            ("f1_macro", lambda a, b: float(f1_score(a, b, labels=list(range(C.NUM_CLASSES)),
                                                       average="macro", zero_division=0))),
        ):
            diffs[metric].append(fn(y[i], pa[i]) - fn(y[i], pb[i]))
    result = {"n_boot": n, "bootstrap": "paired, stratified by true class, percentile 95% CI"}
    for metric, fn in (
        ("qwk", lambda a, b: qwk(a, b)),
        ("accuracy", lambda a, b: float((a == b).mean())),
        ("f1_macro", lambda a, b: float(f1_score(a, b, labels=list(range(C.NUM_CLASSES)),
                                                   average="macro", zero_division=0))),
    ):
        vals = np.asarray(diffs[metric])
        result[metric] = {
            "delta_a_minus_b": fn(y, pa) - fn(y, pb),
            "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
        }
    if np.array_equal(pa, pb):
        if result["qwk"]["delta_a_minus_b"] != 0.0:
            raise AssertionError("Identical predictions produced different QWK values")
        result["identical_predictions"] = True
        result["n_prediction_differences"] = 0
    else:
        result["identical_predictions"] = False
        result["n_prediction_differences"] = int(np.count_nonzero(pa != pb))
    return result


def _save_predictions(path, sample_ids, y, logits):
    logits = np.asarray(logits)
    p = torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=1).numpy()
    pred = p.argmax(1)
    data = {"sample_id": sample_ids, "y_true": y, "y_pred": pred}
    data.update({f"p_class_{c}": p[:, c] for c in range(C.NUM_CLASSES)})
    pd.DataFrame(data).to_csv(path, index=False)


def _save_detailed(details, name, split, outputs, logits_key):
    z = outputs[logits_key]
    prob = torch.softmax(torch.as_tensor(z, dtype=torch.float64), dim=1).numpy()
    pred = prob.argmax(1)
    details.setdefault(name, {})[split] = C.detailed_metrics(outputs["y"], pred, prob)


def _benchmark(model, batch_size):
    try:
        return {"status": "ok", **P.time_model(
            model, input_shape=(batch_size, 3, 224, 224), warmup=10, iters=50,
        )}
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "unavailable_cuda_oom", "batch_size": batch_size, "error": str(exc)}


def stage_analyze():
    d = _data(False)
    split_audit = _split_audit(d)
    missing_teachers = [str(_teacher_ckpt(mode)) for mode in TEACHER_MODES if not _teacher_ckpt(mode).exists()]
    if not BASE_CKPT.exists() or missing_teachers:
        raise FileNotFoundError(
            f"Teacher checkpoints are missing; complete --stage teachers first: "
            f"{missing_teachers or [str(BASE_CKPT)]}"
        )
    missing = [
        f"{arm}/seed_{seed}"
        for arm in ARMS for seed in SEEDS
        if not (_student_dir(arm, seed) / "best_model.pth").exists()
    ]
    if missing and not C.SMOKE:
        raise FileNotFoundError(f"Student runs are incomplete; missing best checkpoints: {missing}")

    out = OUT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    y_test = d["labels"]["test"]
    rows, report, detailed = [], {"split_audit": split_audit}, {}
    student_outputs = {}

    # duplicate audit on the exact split used
    groups = C.duplicate_groups(d["images"])
    pos = {"train": set(d["indices"]["train"].tolist()), "val": set(d["indices"]["val"].tolist()),
           "test": set(d["indices"]["test"].tolist())}
    development = pos["train"] | pos["val"]
    leak = sorted({i for g in groups for i in g if i in pos["test"] and any(j in development for j in g)})
    report["duplicate_groups"] = len(groups)
    report["test_images_with_development_duplicate"] = len(leak)
    report["duplicate_groups_with_conflicting_labels"] = sum(
        len({int(d["all_labels"][i]) for i in g}) > 1 for g in groups
    )
    duplicate_rows = []
    for group_id, group in enumerate(groups):
        if len({int(d["all_labels"][i]) for i in group}) > 1 or any(i in pos["test"] and any(j in development for j in group) for i in group):
            for i in group:
                split = next((name for name, indices in pos.items() if i in indices), "unused")
                duplicate_rows.append({
                    "group_id": group_id, "sample_id": str(d["all_ids"][i]),
                    "diagnosis": int(d["all_labels"][i]), "split": split,
                    "conflicting_labels_in_group": len({int(d["all_labels"][j]) for j in group}) > 1,
                    "test_development_overlap": i in pos["test"] and any(j in development for j in group),
                })
    pd.DataFrame(duplicate_rows).to_csv(out / "duplicate_audit.csv", index=False)
    report["exact_duplicate_groups_crossing_train_test"] = [
        [{"sample_id": str(d["all_ids"][i]), "split": next(name for name, indices in pos.items() if i in indices),
          "diagnosis": int(d["all_labels"][i])} for i in g]
        for g in groups if any(i in development for i in g) and any(i in pos["test"] for i in g)
    ]
    test_pos = {int(i): k for k, i in enumerate(d["indices"]["test"])}
    keep_mask = np.ones(len(y_test), bool)
    for i in leak:
        keep_mask[test_pos[int(i)]] = False

    # ---- teachers
    base = P.HybridCNNViTBase(num_classes=5, config=P.BASELINE_CONFIG).to(P.device)
    base.load_state_dict(torch.load(BASE_CKPT, map_location=P.device, weights_only=True))
    teacher_outputs = {}
    for teacher_mode in TEACHER_MODES:
        teacher = P.load_teacher(BASE_CKPT, _teacher_ckpt(teacher_mode), fusion_mode=teacher_mode)
        teacher_outputs[teacher_mode] = {
            sp: P.collect_outputs(teacher, d[f"{sp}_loader"], "teacher") for sp in ("val", "test")
        }
    for teacher_mode in ("t1", "t2"):
        teacher = P.load_teacher(BASE_CKPT, T2_CKPT, fusion_mode=teacher_mode)
        teacher_outputs[teacher_mode] = {
            sp: P.collect_outputs(teacher, d[f"{sp}_loader"], "teacher") for sp in ("val", "test")
        }
    tout = teacher_outputs["gate"]
    for sp in ("val", "test"):
        arrays = {}
        for teacher_mode, splits in teacher_outputs.items():
            for key, values in splits[sp].items():
                arrays[f"{teacher_mode}__{key}"] = values
        np.savez(out / f"teachers_{sp}.npz", sample_ids=np.asarray(d["ids"][sp]), **arrays)
    teacher_arms = {
        "T1 baseline": ("gate", "z_t1"),
        "T2 SpectralNorm": ("gate", "z_t2"),
        "Equal probability mixture T1+T2 (equal-prob trained)": ("equal_prob", "z_teacher"),
        "Equal logit mixture T1+T2 (equal-logit trained)": ("equal_logits", "z_teacher"),
        "Learned probability gate T1+T2": ("gate", "z_teacher"),
    }
    for sp in ("val", "test"):
        for name, (teacher_mode, key) in teacher_arms.items():
            outputs = teacher_outputs[teacher_mode][sp]
            m = C.metrics_from_logits(outputs["y"], outputs[key])
            rows.append({"level": "teacher", "model": name, "split": sp, "seed": "", **m})
            _save_detailed(detailed, f"teacher:{name}", sp, outputs, key)
            _save_predictions(out / f"predictions_teacher_{name.replace(' ', '_')}_{sp}.csv",
                              d["ids"][sp], outputs["y"], outputs[key])
    report["teacher_gate_test"] = {
        "mean": float(tout["test"]["g_teacher"].mean()),
        "median": float(np.median(tout["test"]["g_teacher"])),
        "std": float(tout["test"]["g_teacher"].std()),
        "min": float(tout["test"]["g_teacher"].min()),
        "max": float(tout["test"]["g_teacher"].max()),
        "fraction_below_0.2": float((tout["test"]["g_teacher"] < 0.2).mean()),
        "fraction_above_0.8": float((tout["test"]["g_teacher"] > 0.8).mean()),
    }
    report["teacher_pairwise_test"] = {}
    equal_logit = teacher_outputs["equal_logits"]["test"]["z_teacher"]
    equal_prob = teacher_outputs["equal_prob"]["test"]["z_teacher"]
    pred_logit, pred_prob = equal_logit.argmax(1), equal_prob.argmax(1)
    report["teacher_pairwise_test"]["equal_logits_vs_equal_probability"] = paired_bootstrap(
        y_test, pred_logit, pred_prob
    )
    gate_trained_pred = teacher_outputs["gate"]["test"]["z_teacher"].argmax(1)
    report["teacher_pairwise_test"]["gate_trained_vs_equal_probability_trained"] = paired_bootstrap(
        y_test, gate_trained_pred, pred_prob
    )
    report["teacher_pairwise_test"]["gate_trained_vs_equal_logits_trained"] = paired_bootstrap(
        y_test, gate_trained_pred, pred_logit
    )
    for name_a, key_a, name_b, key_b in (
        ("learned_gate", "z_teacher", "equal_probability_posthoc_on_gate_teacher", "z_equal_teacher"),
        ("learned_gate", "z_teacher", "T2_only", "z_t2"),
        ("equal_probability", "z_equal_teacher", "T2_only", "z_t2"),
        ("learned_gate", "z_teacher", "T1_only", "z_t1"),
    ):
        pa = tout["test"][key_a].argmax(1)
        pb = tout["test"][key_b].argmax(1)
        report["teacher_pairwise_test"][f"{name_a}_vs_{name_b}"] = paired_bootstrap(y_test, pa, pb)

    # ---- students
    for arm in ARMS:
        for seed in SEEDS:
            ck = _student_dir(arm, seed) / "best_model.pth"
            if not ck.exists():
                continue
            teacher_mode, mix, _ = _arm_spec(arm)
            m_ = P.ResidualTeacherStudentModel(
                P.load_teacher(BASE_CKPT, _teacher_ckpt(teacher_mode) if teacher_mode in TEACHER_MODES
                               else T2_CKPT, fusion_mode=teacher_mode),
                P.ConvNeXtTinyStudent(5, pretrained=False),
                mix=mix, teacher_mode=teacher_mode, **P.STUDENT_MODEL_KW,
            ).to(P.device)
            P.load_student_checkpoint(m_, ck)
            so = {sp: P.collect_outputs(m_, d[f"{sp}_loader"], "student") for sp in ("val", "test")}
            student_outputs[(arm, seed)] = so
            for sp in ("val", "test"):
                np.savez(out / f"student_{arm}_seed{seed}_{sp}.npz",
                         sample_ids=np.asarray(d["ids"][sp]), **so[sp])
                rows.append({"level": "system", "model": f"ConvNeXt student [{arm}]", "split": sp, "seed": seed,
                             **C.metrics_from_logits(so[sp]["y"], so[sp]["z_final"])})
                _save_detailed(detailed, f"student:{arm}:seed_{seed}", sp, so[sp], "z_final")
                _save_predictions(out / f"predictions_student_{arm}_seed{seed}_{sp}.csv",
                                  d["ids"][sp], so[sp]["y"], so[sp]["z_final"])
                rows.append({"level": "student_branch", "model": f"ConvNeXt only [{arm}]", "split": sp,
                             "seed": seed, **C.metrics_from_logits(so[sp]["y"], so[sp]["z_student"])})
                _save_detailed(detailed, f"student_branch:{arm}:seed_{seed}", sp, so[sp], "z_student")
                _save_predictions(out / f"predictions_student_branch_{arm}_seed{seed}_{sp}.csv",
                                  d["ids"][sp], so[sp]["y"], so[sp]["z_student"])
            # test set without the images that have an exact duplicate in train
            p = so["test"]["z_final"].argmax(1)
            report.setdefault("qwk_test_excluding_train_duplicates", {})[f"{arm}_seed{seed}"] = {
                "n": int(keep_mask.sum()), "qwk": qwk(y_test[keep_mask], p[keep_mask])}

    # Inference-only comparison on the very same trained gated student.
    report["inference_gate_vs_equal_swap"] = {}
    for teacher_mode in ("gate", "equal_prob", "equal_logits", "t1", "t2"):
        gated_arm = next((arm for arm in ARMS if _arm_spec(arm)[:2] == (teacher_mode, "gated")), None)
        if gated_arm is None:
            continue
        for seed in SEEDS:
            pair = student_outputs.get((gated_arm, seed))
            if pair is None:
                continue
            for split in ("val", "test"):
                z_equal = (pair[split]["z_teacher"] + pair[split]["z_student"]) / 2.0
                pred_gated = pair[split]["z_final"].argmax(1)
                pred_equal = z_equal.argmax(1)
                gate = pair[split]["g_student"].ravel()
                key = f"{teacher_mode}_seed_{seed}_{split}"
                report["inference_gate_vs_equal_swap"][key] = {
                    "n": len(pred_gated),
                    "qwk_gated": qwk(pair[split]["y"], pred_gated),
                    "qwk_equal_weight": qwk(pair[split]["y"], pred_equal),
                    "roc_auc_gated": C.metrics_from_logits(pair[split]["y"], pair[split]["z_final"])["roc_auc"],
                    "roc_auc_equal_weight": C.metrics_from_logits(pair[split]["y"], z_equal)["roc_auc"],
                    "gate_mean": float(gate.mean()),
                    "gate_median": float(np.median(gate)),
                    "fraction_gate_below_0.2": float((gate < 0.2).mean()),
                    "fraction_gate_below_0.3": float((gate < 0.3).mean()),
                    "fraction_gate_above_0.7": float((gate > 0.7).mean()),
                    "mean_absolute_logit_difference": float(np.abs(pair[split]["z_final"] - z_equal).mean()),
                    "paired_comparison": paired_bootstrap(pair[split]["y"], pred_gated, pred_equal)
                        if split == "test" else {
                            "identical_predictions": bool(np.array_equal(pred_gated, pred_equal)),
                            "n_prediction_differences": int(np.count_nonzero(pred_gated != pred_equal)),
                            "qwk_gated": qwk(pair[split]["y"], pred_gated),
                            "qwk_equal_weight": qwk(pair[split]["y"], pred_equal),
                        },
                }
    equal_student_only = {}
    for teacher_mode in ("gate", "equal_prob", "equal_logits", "t1", "t2"):
        arm = next((a for a in ARMS if _arm_spec(a)[:2] == (teacher_mode, "equal")), None)
        if arm is not None:
            equal_student_only[teacher_mode] = {
                seed: student_outputs[(arm, seed)]["test"]
                for seed in SEEDS if (arm, seed) in student_outputs
            }
    for teacher_mode, gated_arm in (
        ("gate", "teacher_gate__student_gated"),
        ("equal_prob", "teacher_equalprob__student_gated"),
        ("equal_logits", "teacher_equallogit__student_gated"),
        ("t1", "teacher_t1__student_gated"),
        ("t2", "teacher_t2__student_gated"),
    ):
        for seed in SEEDS:
            gated = student_outputs.get((gated_arm, seed))
            equal = equal_student_only.get(teacher_mode, {}).get(seed)
            if gated is None or equal is None:
                continue
            zg = gated["test"]["z_final"]
            ze = (equal["z_teacher"] + equal["z_student"]) / 2.0
            report["inference_gate_vs_equal_swap"][f"{teacher_mode}_trained_equal_arm_seed_{seed}_test"] = {
                "n": len(y_test),
                "qwk_gated": qwk(y_test, zg.argmax(1)),
                "qwk_equal_weight": qwk(y_test, ze.argmax(1)),
                "n_prediction_differences": int(np.count_nonzero(zg.argmax(1) != ze.argmax(1))),
                "note": "Post-hoc inference swap on separately trained checkpoints; descriptive, not a trained-arm comparison.",
            }

    # Primary comparison: independently retrained gated and equal-weight students, matched by seed.
    report["retrained_gate_vs_equal"] = {}
    for teacher_mode in ("gate", "equal_prob", "equal_logits", "t1", "t2"):
        gated_arm = next((arm for arm in ARMS if _arm_spec(arm)[:2] == (teacher_mode, "gated")), None)
        equal_arm = next((arm for arm in ARMS if _arm_spec(arm)[:2] == (teacher_mode, "equal")), None)
        if gated_arm is None or equal_arm is None:
            continue
        for seed in SEEDS:
            gated = student_outputs.get((gated_arm, seed))
            equal = student_outputs.get((equal_arm, seed))
            if gated is None or equal is None:
                continue
            for split in ("val", "test"):
                if not np.array_equal(gated[split]["y"], equal[split]["y"]):
                    raise AssertionError(f"Label order differs between gate/equal, {teacher_mode}, seed {seed}, split {split}")
                pg, pe = gated[split]["z_final"].argmax(1), equal[split]["z_final"].argmax(1)
                cmp = {
                    "n": len(pg),
                    "qwk_gated": qwk(gated[split]["y"], pg),
                    "qwk_equal_weight": qwk(equal[split]["y"], pe),
                    "n_prediction_differences": int(np.count_nonzero(pg != pe)),
                    "identical_predictions": bool(np.array_equal(pg, pe)),
                }
                if split == "test":
                    cmp["paired_bootstrap"] = paired_bootstrap(gated[split]["y"], pg, pe)
                report["retrained_gate_vs_equal"][f"{teacher_mode}_seed_{seed}_{split}"] = cmp

    report["teacher_signal_ablation"] = {}
    reference_arm = "teacher_gate__student_gated"
    for arm in ARMS:
        if arm == reference_arm or _arm_spec(arm)[1] != "gated":
            continue
        for seed in SEEDS:
            ref_outputs = student_outputs.get((reference_arm, seed))
            other_outputs = student_outputs.get((arm, seed))
            if ref_outputs is None or other_outputs is None:
                continue
            for split in ("val", "test"):
                y_ref, y_other = ref_outputs[split]["y"], other_outputs[split]["y"]
                if not np.array_equal(y_ref, y_other):
                    raise AssertionError(f"Label order differs in teacher ablation {arm}, seed {seed}, split {split}")
                p_ref = ref_outputs[split]["z_final"].argmax(1)
                p_other = other_outputs[split]["z_final"].argmax(1)
                cmp = {
                    "reference_teacher_gate_qwk": qwk(y_ref, p_ref),
                    "ablation_qwk": qwk(y_other, p_other),
                    "n_prediction_differences": int(np.count_nonzero(p_ref != p_other)),
                    "identical_predictions": bool(np.array_equal(p_ref, p_other)),
                }
                if split == "test":
                    cmp["paired_bootstrap"] = paired_bootstrap(y_ref, p_ref, p_other)
                report["teacher_signal_ablation"][f"{arm}_seed_{seed}_{split}"] = cmp

    # Accuracy/ordinal-score gain versus the student branch alone, paired per image.
    report["system_vs_student_only"] = {}
    for arm in ARMS:
        for seed in SEEDS:
            outputs = student_outputs.get((arm, seed))
            if outputs is None:
                continue
            for split in ("val", "test"):
                y = outputs[split]["y"]
                pred_full = outputs[split]["z_final"].argmax(1)
                pred_student = outputs[split]["z_student"].argmax(1)
                arm_teacher_mode, arm_mix, _ = _arm_spec(arm)
                cmp = {
                    "teacher_mode": arm_teacher_mode,
                    "student_fusion": arm_mix,
                    "qwk_full_system": qwk(y, pred_full),
                    "qwk_student_only": qwk(y, pred_student),
                    "n_prediction_differences": int(np.count_nonzero(pred_full != pred_student)),
                }
                if split == "test":
                    cmp["paired_bootstrap"] = paired_bootstrap(y, pred_full, pred_student)
                report["system_vs_student_only"][f"{arm}_seed_{seed}_{split}"] = cmp

    # Compare augmentation within each matched model configuration, paired by seed.
    report["augmentation_vs_no_augmentation"] = {}
    augmentation_pairs = {
        "gate_gated": ("teacher_gate__student_gated", "teacher_gate__student_gated_aug"),
        "equal_logits_equal": ("teacher_equallogit__student_equal",
                               "teacher_equallogit__student_equal_aug"),
    }
    for comparison, (plain_arm, augmented_arm) in augmentation_pairs.items():
        for seed in SEEDS:
            plain = student_outputs.get((plain_arm, seed))
            augmented = student_outputs.get((augmented_arm, seed))
            if plain is None or augmented is None:
                continue
            for split in ("val", "test"):
                y = plain[split]["y"]
                if not np.array_equal(y, augmented[split]["y"]):
                    raise AssertionError(
                        f"Label order differs for {comparison} augmentation comparison, seed {seed}, split {split}"
                    )
                pp = plain[split]["z_final"].argmax(1)
                pa = augmented[split]["z_final"].argmax(1)
                cmp = {
                    "plain_arm": plain_arm,
                    "augmented_arm": augmented_arm,
                    "qwk_no_augmentation": qwk(y, pp),
                    "qwk_augmentation": qwk(y, pa),
                    "n_prediction_differences": int(np.count_nonzero(pp != pa)),
                    "identical_predictions": bool(np.array_equal(pp, pa)),
                }
                if split == "test":
                    cmp["paired_bootstrap"] = paired_bootstrap(y, pa, pp)
                report["augmentation_vs_no_augmentation"][f"{comparison}_seed_{seed}_{split}"] = cmp

    df = pd.DataFrame(rows)
    df.to_csv(out / "all_metrics.csv", index=False)
    # arm x seed summary (test)
    s = df[(df.level == "system") & (df.split == "test")]
    if len(s):
        summ = s.groupby("model")[["accuracy", "f1_macro", "qwk", "roc_auc"]].agg(["mean", "std", "count"])
        summ.to_csv(out / "student_summary_mean_sd.csv")
        print(summ.round(4).to_string())

    # ---- latency / size on the trained seed-42 system, including teacher inference.
    latency_arm = ARMS[0]
    latency_teacher_mode, latency_mix, _ = _arm_spec(latency_arm)
    latency_model = student_outputs.get((latency_arm, 42))
    latency_ck = _student_dir(latency_arm, 42) / "best_model.pth"
    if latency_model is None and latency_ck.exists():
        sys_model = P.ResidualTeacherStudentModel(
            P.load_teacher(BASE_CKPT, _teacher_ckpt(latency_teacher_mode),
                           fusion_mode=latency_teacher_mode),
            P.ConvNeXtTinyStudent(5, pretrained=False),
            mix=latency_mix, teacher_mode=latency_teacher_mode, **P.STUDENT_MODEL_KW,
        ).to(P.device)
        P.load_student_checkpoint(sys_model, latency_ck)
    elif latency_model is not None:
        sys_model = P.ResidualTeacherStudentModel(
            P.load_teacher(BASE_CKPT, _teacher_ckpt(latency_teacher_mode),
                           fusion_mode=latency_teacher_mode),
            P.ConvNeXtTinyStudent(5, pretrained=False),
            mix=latency_mix, teacher_mode=latency_teacher_mode, **P.STUDENT_MODEL_KW,
        ).to(P.device)
        P.load_student_checkpoint(sys_model, latency_ck)
    else:
        raise FileNotFoundError(f"A trained seed-42 model is required for latency reporting: {latency_arm}")
    stu, full_teacher = sys_model.student_model, sys_model.teacher_model
    lat = {
        "device": str(P.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "batch_1": {"student_only": _benchmark(stu, 1), "teacher_ensemble": _benchmark(full_teacher, 1),
                    "full_system": _benchmark(sys_model, 1)},
        "batch_16": {"student_only": _benchmark(stu, 16), "teacher_ensemble": _benchmark(full_teacher, 16),
                     "full_system": _benchmark(sys_model, 16)},
        "params": {"student": P.count_params(stu), "teacher_ensemble": P.count_params(full_teacher),
                   "full_system": P.count_params(sys_model), "t1": P.count_params(base)},
        "measurement_note": "Forward-pass latency only; preprocessing, file I/O, and network transfer excluded.",
    }
    report["latency_and_params"] = lat

    report["experiment_matrix"] = {
        arm: {
            "teacher_fusion": _arm_spec(arm)[0],
            "student_fusion": _arm_spec(arm)[1],
            "augmentation": _arm_spec(arm)[2],
            "seeds": SEEDS,
        }
        for arm in ARMS
    }
    report["model_selection_and_test_use"] = {
        "checkpoint_selection": "Per-run best validation QWK; no test metric enters checkpoint selection.",
        "preplanned_test_evaluation": "All predeclared arms use the same frozen test IDs for paired comparisons.",
        "no_posthoc_winner_claim": "Do not select or tune the manuscript's final method using these test comparisons.",
        "historical_caveat": "Previously reported model-family selection used test comparisons; this run does not erase that historical exposure.",
    }
    report["experiment_scope"] = {
        "purpose": "Manuscript-aligned final configuration on a 2929/366/367 split",
        "student_arms": ARMS,
        "seeds": SEEDS,
        "split_note": "Preserves the prior frozen test IDs and transfers one validation sample to training; this is not claimed to recover the historical sample IDs.",
    }
    report["metrics_definition"] = {
        "selection": "best validation quadratic weighted kappa (QWK); test set is not used for checkpoint selection",
        "primary_test_metrics": ["QWK", "accuracy", "macro precision", "macro recall", "macro F1",
                                 "weighted F1", "macro specificity", "weighted one-vs-rest ROC-AUC",
                                 "mean absolute grade error", "within-one-grade accuracy", "multiclass Brier score",
                                 "negative log-likelihood", "15-bin expected calibration error"],
        "classwise": "precision, recall/sensitivity, specificity, one-vs-rest ROC-AUC, F1, support, confusion matrix",
        "paired_bootstrap": "stratified by true class, paired image resampling, 2000 replicates, percentile 95% CI",
        "duplicate_sensitivity_analysis": "secondary test QWK excludes test images having exact resized-pixel duplicates in train or validation",
        "teacher_equal_probability": "mean of teacher softmax probabilities",
        "teacher_equal_logits": "arithmetic mean of raw Teacher 1 and Teacher 2 logits",
        "student_equal_weight": "arithmetic mean of teacher and student logits, matching the original inference ablation",
    }
    C.write_json(report, out / "report.json")
    C.write_json(detailed, out / "detailed_metrics.json")
    C.write_json(C.env_report(), out / "env.json")
    _mirror()
    print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["teachers", "students", "analyze"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sync_stop = C.start_background_sync(OUT, DRIVE, interval_s=600) if DRIVE else None
    try:
        if a.stage == "teachers":
            stage_teachers(a.seed)
        elif a.stage == "students":
            stage_students()
        else:
            stage_analyze()
    finally:
        if sync_stop is not None:
            sync_stop.set()
            _mirror()
