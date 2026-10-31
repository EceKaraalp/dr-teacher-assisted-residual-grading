"""Frozen reviewer experiment matrix; no model choice is made from test scores."""
from dataclasses import asdict, dataclass
import hashlib
import json


SEEDS = (42, 52, 62)
ARMS = (
    "joint_ce", "joint_single_kd", "joint_multi_kd", "joint_ordinal",
    "joint_hinge", "joint_full_no_entropy", "joint_full",
    "student_ce", "teacher2_only", "augmentation", "teacher_eval",
)


@dataclass(frozen=True)
class Run:
    preprocessing: str
    fraction: float
    seed: int
    stage: str
    arm: str = ""

    @property
    def group(self):
        return f"{self.preprocessing}_f{int(self.fraction * 100):03d}_s{self.seed}"

    @property
    def key(self):
        return f"{self.group}/{self.stage}" + (f"/{self.arm}" if self.arm else "")

    @property
    def dependency(self):
        if self.stage == "teacher2":
            return f"{self.group}/teacher1"
        if self.stage == "student" and self.arm != "student_ce":
            return f"{self.group}/teacher2"
        return None


def matrix():
    runs = []
    settings = [("raw", 1.0), ("imagenet", 1.0), ("retinal", 1.0)]
    settings += [("raw", fraction) for fraction in (0.1, 0.25, 0.5)]
    for preprocessing, fraction in settings:
        arms = ARMS if (preprocessing, fraction) == ("raw", 1.0) else ("joint_full", "student_ce")
        for seed in SEEDS:
            runs += [Run(preprocessing, fraction, seed, "teacher1"),
                     Run(preprocessing, fraction, seed, "teacher2")]
            runs += [Run(preprocessing, fraction, seed, "student", arm) for arm in arms]
    return runs


def loss_config(arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    full = arm in {"joint_full", "joint_full_no_entropy", "teacher2_only", "augmentation", "teacher_eval"}
    kd = arm not in {"joint_ce", "student_ce"}
    return {
        "kd": kd, "ordinal": full or arm == "joint_ordinal",
        "hinge": full or arm == "joint_hinge",
        "temperatures": [2.0] if arm == "joint_single_kd" else [2.0, 4.0],
        "temperature_weights": [1.0] if arm == "joint_single_kd" else [0.6, 0.4],
        "kd_coefficient": 0.30, "ordinal_coefficient": 0.12,
        "entropy_coefficient": 0.01 if full and arm != "joint_full_no_entropy" else 0.0,
    }


def protocol():
    return {
        "version": 1, "seeds": SEEDS, "split": "frozen historical 2928/367/367; test exposed historically",
        "selection": "validation QWK; never choose arm, temperature, preprocessing or fraction from test scores",
        "student_training": "original joint gated output, detached teacher; equal inference is reported separately",
        "independent_baseline": "student_ce optimizes standalone student weighted CE",
        "teacher_buffers": "historical updating behavior except teacher_eval control",
        "preprocessing": {
            "raw": "RGB bilinear 224, float32 0..255",
            "imagenet": "RGB bilinear 224, /255, ImageNet mean/std",
            "retinal": "CLAHE on LAB L channel at cached 224 resolution; then ImageNet normalization",
        },
        "fractions": "nested class-stratified subsets; ALL teacher/student labels restricted to subset",
        "augmentation": "student training only; H/V flip .5, rotation +/-20 deg, brightness/contrast .9..1.1",
        "student": {"epochs": 70, "patience": 10, "lr": 2e-4, "batch": 16},
        "teacher1": {"epochs": 110, "patience": 15, "lr": 1e-3},
        "teacher2": {"epochs": 120, "patience": 18, "lr": 1e-4},
        "optimizer": "AdamW; weight decay 1e-4; clip norm 1; float32",
        "scheduler": "CosineAnnealingWarmRestarts T_0=10 T_mult=2",
        "single_temperature": "T=2 predefined; not selected by test",
        "test_policy": "separate explicit evaluation stage after all training is complete",
        "runs": [asdict(run) for run in matrix()],
    }


def fingerprint():
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
    print(json.dumps({"training_jobs": len(matrix()), "protocol_sha256": fingerprint(), **protocol()}, indent=2))
