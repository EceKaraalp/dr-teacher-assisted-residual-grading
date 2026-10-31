"""Cross-experiment summary. Usage: python summarize_all.py <results_parent_dir> <output_dir>

Reads every experiment folder (<parent>/<experiment>/students/<arm>/seed_*/test_metrics.json and the
teacher test_metrics.json files) and writes long/mean-sd CSVs plus a comparison to the manuscript's final model.
"""
import json
import sys
from pathlib import Path

import pandas as pd

PAPER = {"accuracy": 0.8547, "f1_macro": 0.703, "qwk": 0.9186, "qwk_sd": 0.0010}
FINAL_ARM = "teacher_equallogit__student_equal"
KEYS = ["accuracy", "f1_macro", "qwk", "roc_auc", "expected_calibration_error_15_bins"]


def main(parent, out):
    parent, out = Path(parent), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, teachers = [], []
    for exp in sorted(p for p in parent.iterdir() if p.is_dir() and (p / "students").exists()):
        for f in sorted(exp.glob("students/*/seed_*/test_metrics.json")):
            m = json.loads(f.read_text(encoding="utf-8"))
            rows.append({"experiment": exp.name, "arm": m.get("arm", f.parts[-3]), "seed": m.get("seed"),
                         **{k: m.get(k) for k in KEYS}, "best_val_qwk": m.get("best_val_qwk"),
                         "best_epoch": m.get("best_epoch")})
        for f in sorted(exp.glob("teachers/**/test_metrics.json")):
            m = json.loads(f.read_text(encoding="utf-8"))
            teachers.append({"experiment": exp.name, "teacher": m.get("fusion_mode") or m.get("model"),
                             "seed": m.get("seed"), **{k: m.get(k) for k in KEYS},
                             "best_val_qwk": m.get("best_val_qwk")})
    if not rows:
        raise SystemExit(f"No student results under {parent}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "all_student_runs.csv", index=False)
    pd.DataFrame(teachers).to_csv(out / "all_teacher_runs.csv", index=False)
    summ = df.groupby(["experiment", "arm"])[KEYS].agg(["mean", "std", "count"])
    summ.to_csv(out / "summary_mean_sd_per_experiment_arm.csv")

    final = df[df.arm == FINAL_ARM].groupby("experiment")[KEYS].agg(["mean", "std"])
    final[("paper", "qwk")] = PAPER["qwk"]
    final[("delta_qwk_vs_paper", "")] = final[("qwk", "mean")] - PAPER["qwk"]
    final[("delta_acc_vs_paper", "")] = final[("accuracy", "mean")] - PAPER["accuracy"]
    final[("delta_f1_vs_paper", "")] = final[("f1_macro", "mean")] - PAPER["f1_macro"]
    final.to_csv(out / "final_model_vs_paper.csv")
    print("=== Final model (equal-logit teacher + equal student) by experiment; paper QWK 0.9186+-0.0010 ===")
    print(final.round(4).to_string())
    print("\n=== Arm means over all experiments that have the arm ===")
    print(df.groupby(["experiment", "arm"])["qwk"].agg(["mean", "std", "count"]).round(4).to_string())
    print("\nWrote CSVs to", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
