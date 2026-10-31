"""Round-2 analysis: per-seed metrics, mean/sample-SD, validation-only temperature calibration and paired bootstraps."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze import bootstrap, fit_temperature, metrics, probabilities
from plan import SEEDS, Run
from plan2 import Job, jobs
import run as runner
from run2 import sync_files

SUMMARY_METRICS = ("accuracy", "macro_f1", "qwk", "weighted_ovr_auc", "nll", "brier", "ece15")


def load(output, job, split):
    path = output / job.key / f"round2_{split}_logits.npz"
    if not path.exists():
        return None
    with np.load(path) as values:
        return {name: values[name] for name in values.files}


def per_seed(output):
    report = {}
    for job in jobs():
        val, test = load(output, job, "val"), load(output, job, "test")
        if val is None or test is None:
            continue
        entry = {}
        for path in (k for k in test if k not in {"labels", "ids"}):
            temperature = fit_temperature(val[path], val["labels"])
            entry[path] = {"test": metrics(test["labels"], probabilities(test[path])),
                           "val": metrics(val["labels"], probabilities(val[path])),
                           "temperature": temperature,
                           "test_calibrated": metrics(test["labels"], probabilities(test[path] / temperature))}
        report[job.key] = entry
    return report


def mean_sd(report):
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for job in jobs():
        if job.key not in report:
            continue
        group = f"{job.family}/{job.stage}" + (f"/{job.arm}" if job.arm else "")
        for path, values in report[job.key].items():
            for metric in SUMMARY_METRICS:
                groups[group][path][metric].append(values["test"][metric])
            for metric in ("nll", "brier", "ece15"):
                groups[group][path][f"calibrated_{metric}"].append(values["test_calibrated"][metric])
    result = {}
    for group, paths in groups.items():
        result[group] = {}
        for path, values in paths.items():
            result[group][path] = {metric: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                                            "n_seeds": len(v), "values": [float(x) for x in v]}
                                   for metric, v in values.items()}
    return result


def predictions(output, family, stage, arm, path):
    values = []
    for seed in SEEDS:
        data = load(output, Job(family, seed, stage, arm), "test")
        if data is None or path not in data:
            return None, None
        values.append((data["labels"], data["ids"], data[path].argmax(1)))
    labels, ids = values[0][0], values[0][1]
    if any(not np.array_equal(v[0], labels) or not np.array_equal(v[1], ids) for v in values):
        raise ValueError(f"Test labels/ids differ across seeds for {family}/{stage}/{arm}")
    return labels, [v[2] for v in values]


CONTRASTS = {
    # Reviewer 1.4/1.9 and 2.4: gated primary and equal inference against the matched independent CE student.
    "reference/gated_vs_ce": (("reference", "student", "joint_full", "primary"), ("reference", "student", "student_ce", "primary")),
    "reference/equal_vs_ce": (("reference", "student", "joint_full", "equal"), ("reference", "student", "student_ce", "primary")),
    "reference/gated_vs_equal": (("reference", "student", "joint_full", "primary"), ("reference", "student", "joint_full", "equal")),
    "reference/teacher_gate_vs_ce": (("reference", "student", "joint_full", "teacher"), ("reference", "student", "student_ce", "primary")),
    # Reviewer 1.6 / 2.3: Teacher 1 contribution.
    "teacher_level/gate_ensemble_vs_joint_t2_alone": (("reference", "teacher2", "", "primary"), ("reference", "teacher2", "", "teacher2")),
    "teacher_level/independent_t2_vs_gate_ensemble": (("t2indep", "teacher2_indep", "", "primary"), ("reference", "teacher2", "", "primary")),
    "teacher_level/independent_t2_vs_joint_t2_alone": (("t2indep", "teacher2_indep", "", "primary"), ("reference", "teacher2", "", "teacher2")),
    "t2indep/gated_vs_reference_gated": (("t2indep", "student", "joint_full", "primary"), ("reference", "student", "joint_full", "primary")),
    "t2indep/equal_vs_reference_equal": (("t2indep", "student", "joint_full", "equal"), ("reference", "student", "joint_full", "equal")),
    "t2indep/gated_vs_ce": (("t2indep", "student", "joint_full", "primary"), ("reference", "student", "student_ce", "primary")),
    # Reviewer 1.8: spectral normalization.
    "nosn/teacher_ensemble_vs_reference": (("nosn", "teacher2", "", "primary"), ("reference", "teacher2", "", "primary")),
    "nosn/t2_alone_vs_reference": (("nosn", "teacher2", "", "teacher2"), ("reference", "teacher2", "", "teacher2")),
    "nosn/gated_vs_reference_gated": (("nosn", "student", "joint_full", "primary"), ("reference", "student", "joint_full", "primary")),
    "nosn/equal_vs_reference_equal": (("nosn", "student", "joint_full", "equal"), ("reference", "student", "joint_full", "equal")),
    # Reviewer 2.2: augmentation in all stages.
    "aug/teacher1_vs_reference": (("aug", "teacher1", "", "primary"), ("reference", "teacher1", "", "primary")),
    "aug/teacher_ensemble_vs_reference": (("aug", "teacher2", "", "primary"), ("reference", "teacher2", "", "primary")),
    "aug/gated_vs_reference_gated": (("aug", "student", "joint_full", "primary"), ("reference", "student", "joint_full", "primary")),
    "aug/equal_vs_reference_equal": (("aug", "student", "joint_full", "equal"), ("reference", "student", "joint_full", "equal")),
    "aug/ce_vs_reference_ce": (("aug", "student", "student_ce", "primary"), ("reference", "student", "student_ce", "primary")),
    "aug/gated_vs_aug_ce": (("aug", "student", "joint_full", "primary"), ("aug", "student", "student_ce", "primary")),
    # Reviewer 1.5: Table-1 Fusion CNN-ViT retrained on the frozen split.
    "fusion/historical_vs_reference_gated": (("fusion", "fusion", "historical", "primary"), ("reference", "student", "joint_full", "primary")),
    "fusion/matched_vs_reference_gated": (("fusion", "fusion", "matched", "primary"), ("reference", "student", "joint_full", "primary")),
    "fusion/matched_vs_reference_ce": (("fusion", "fusion", "matched", "primary"), ("reference", "student", "student_ce", "primary")),
    "fusion/historical_vs_matched": (("fusion", "fusion", "historical", "primary"), ("fusion", "fusion", "matched", "primary")),
}


def paired(output, replicates=10000):
    result = {}
    for name, (left, right) in CONTRASTS.items():
        y_left, a = predictions(output, *left)
        y_right, b = predictions(output, *right)
        if a is None or b is None:
            result[name] = {"status": "unavailable (missing jobs or path)"}
            continue
        if not np.array_equal(y_left, y_right):
            raise ValueError(f"Contrast {name} compares different test labels.")
        result[name] = {"left": "/".join(left), "right": "/".join(right), **bootstrap(y_left, a, b, replicates=replicates)}
    return result


def round1_label_budget(round1, replicates=10000):
    """Gated and equal paths versus the independent CE student at each nested label budget (round-1 test logits)."""
    result = {}
    if round1 is None:
        return {"status": "round-1 folder not given"}
    for fraction in (0.1, 0.25, 0.5, 1.0):
        loaded = {}
        for arm in ("joint_full", "student_ce"):
            values = []
            for seed in SEEDS:
                path = round1 / Run("raw", fraction, seed, "student", arm).key / "test_logits.npz"
                if not path.exists():
                    values = None
                    break
                with np.load(path) as data:
                    values.append({k: data[k] for k in data.files})
            loaded[arm] = values
        label = f"fraction_{fraction}"
        if not loaded["joint_full"] or not loaded["student_ce"]:
            result[label] = {"status": "unavailable (round-1 test_logits.npz missing)"}
            continue
        y = loaded["joint_full"][0]["labels"]
        for group in loaded.values():
            for data in group:
                if not np.array_equal(data["labels"], y):
                    raise ValueError("Round-1 label mismatch.")
        ce = [d["primary"].argmax(1) for d in loaded["student_ce"]]
        result[label] = {
            "gated_vs_ce": bootstrap(y, [d["primary"].argmax(1) for d in loaded["joint_full"]], ce, replicates=replicates),
            "equal_vs_ce": bootstrap(y, [d["equal"].argmax(1) for d in loaded["joint_full"]], ce, replicates=replicates),
            "gated_qwk_per_seed": [metrics(y, probabilities(d["primary"]))["qwk"] for d in loaded["joint_full"]],
            "equal_qwk_per_seed": [metrics(y, probabilities(d["equal"]))["qwk"] for d in loaded["joint_full"]],
            "ce_qwk_per_seed": [metrics(y, probabilities(d["primary"]))["qwk"] for d in loaded["student_ce"]],
        }
    return result


def summary(summary_values, contrasts, budgets, references):
    lines = ["# Round-2 reviewer experiments (APTOS frozen test, 3 seeds)", "",
             "Mean +/- sample SD across seeds. Paired bootstrap: 10,000 shared-image resamples, conditional on trained models.", "",
             "## Reference provenance", ""]
    lines += [f"- {key}: {value.get('mode')}" for key, value in sorted(references.items())]
    lines += ["", "## Test metrics", "", "| Group | Path | Accuracy | Macro-F1 | QWK | NLL | ECE | Calibrated ECE |",
              "|---|---|---|---|---|---|---|---|"]
    for group, paths in summary_values.items():
        for path, values in paths.items():
            cell = lambda m: f"{values[m]['mean']:.4f} +/- {values[m]['sd']:.4f}" if values[m]["sd"] is not None else f"{values[m]['mean']:.4f}"
            lines.append(f"| {group} | {path} | {cell('accuracy')} | {cell('macro_f1')} | {cell('qwk')} | {cell('nll')} | "
                         f"{cell('ece15')} | {cell('calibrated_ece15')} |")
    lines += ["", "## Paired QWK differences (left - right)", "", "| Contrast | Mean difference | 95% interval |", "|---|---|---|"]
    for name, value in contrasts.items():
        if "mean_difference" in value:
            lines.append(f"| {name} | {value['mean_difference']:+.4f} | [{value['percentile95'][0]:+.4f}, {value['percentile95'][1]:+.4f}] |")
        else:
            lines.append(f"| {name} | {value['status']} | |")
    lines += ["", "## Round-1 label budgets (gated / equal minus CE)", ""]
    if isinstance(budgets, dict) and "status" in budgets:
        lines.append(budgets["status"])
    for label, value in budgets.items():
        if isinstance(value, dict) and "gated_vs_ce" in value:
            g, e = value["gated_vs_ce"], value["equal_vs_ce"]
            lines.append(f"- {label}: gated {g['mean_difference']:+.4f} [{g['percentile95'][0]:+.4f}, {g['percentile95'][1]:+.4f}]; "
                         f"equal {e['mean_difference']:+.4f} [{e['percentile95'][0]:+.4f}, {e['percentile95'][1]:+.4f}]")
        elif isinstance(value, dict) and "status" in value:
            lines.append(f"- {label}: {value['status']}")
    return "\n".join(lines) + "\n"


def analyze(output, round1=None, drive=None, replicates=10000):
    destination = output / "analysis2"
    report = per_seed(output)
    values = mean_sd(report)
    contrasts = paired(output, replicates)
    budgets = round1_label_budget(round1, replicates)
    manifest_path = output / "reference_manifest.json"
    references = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    runner.write_json(destination / "per_seed.json", report)
    runner.write_json(destination / "mean_sd.json", values)
    runner.write_json(destination / "paired_qwk.json", contrasts)
    runner.write_json(destination / "label_budget_round1.json", budgets)
    expected = [job.key for job in jobs()]
    runner.write_json(destination / "status.json", {"evaluated_jobs": len(report), "expected_jobs": len(expected),
                                                    "missing": [k for k in expected if k not in report],
                                                    "complete": len(report) == len(expected)})
    (destination / "summary.md").write_text(summary(values, contrasts, budgets, references), encoding="utf-8")
    sync_files(destination, drive / "analysis2" if drive else None,
               ["per_seed.json", "mean_sd.json", "paired_qwk.json", "label_budget_round1.json", "status.json", "summary.md"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drive", type=Path)
    parser.add_argument("--round1", type=Path)
    args, _ = parser.parse_known_args(argv)
    analyze(args.output, args.round1, args.drive)
    print((args.output / "analysis2" / "summary.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
