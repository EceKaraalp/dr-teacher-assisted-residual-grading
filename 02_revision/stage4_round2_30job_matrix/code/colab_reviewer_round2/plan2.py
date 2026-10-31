"""Frozen round-2 reviewer matrix. Round-1 modules are reused unchanged; nothing here is chosen from test scores."""
from dataclasses import asdict, dataclass
import hashlib
import json

from plan import SEEDS, Run, protocol as round1_protocol


ROUND1_RUN_ID = "reviewer_complete_v1"
FAMILIES = ("reference", "t2indep", "nosn", "aug", "fusion")


@dataclass(frozen=True)
class Job:
    family: str
    seed: int
    stage: str
    arm: str = ""

    @property
    def legacy(self):
        """Round-1 Run used for loss/forward semantics (and the exact round-1 key for references)."""
        if self.stage == "teacher2_indep":
            return Run("raw", 1.0, self.seed, "teacher1")
        if self.stage == "fusion":
            return Run("retinal" if self.arm == "historical" else "raw", 1.0, self.seed, "student", "student_ce")
        return Run("raw", 1.0, self.seed, self.stage, self.arm)

    @property
    def key(self):
        if self.family == "reference":
            return f"reference/{self.legacy.key}"
        return f"{self.family}/raw_f100_s{self.seed}/{self.stage}" + (f"/{self.arm}" if self.arm else "")

    @property
    def dependency(self):
        group = f"raw_f100_s{self.seed}"
        if self.family == "reference":
            return f"reference/{self.legacy.dependency}" if self.legacy.dependency else None
        if self.family == "t2indep" and self.stage == "student":
            return f"t2indep/{group}/teacher2_indep"
        if self.family == "nosn":
            return f"reference/{group}/teacher1" if self.stage == "teacher2" else f"nosn/{group}/teacher2"
        if self.family == "aug":
            if self.stage == "teacher2":
                return f"aug/{group}/teacher1"
            if self.arm == "joint_full":
                return f"aug/{group}/teacher2"
        return None

    @property
    def augment(self):
        return self.family == "aug"

    @property
    def preprocessing(self):
        return self.legacy.preprocessing


def jobs():
    result = []
    for seed in SEEDS:
        result += [Job("reference", seed, "teacher1"), Job("reference", seed, "teacher2"),
                   Job("reference", seed, "student", "joint_full"), Job("reference", seed, "student", "student_ce")]
    for seed in SEEDS:
        result += [Job("t2indep", seed, "teacher2_indep"), Job("t2indep", seed, "student", "joint_full")]
    for seed in SEEDS:
        result += [Job("nosn", seed, "teacher2"), Job("nosn", seed, "student", "joint_full")]
    for seed in SEEDS:
        result += [Job("aug", seed, "teacher1"), Job("aug", seed, "teacher2"),
                   Job("aug", seed, "student", "joint_full"), Job("aug", seed, "student", "student_ce")]
    for seed in SEEDS:
        result += [Job("fusion", seed, "fusion", "historical"), Job("fusion", seed, "fusion", "matched")]
    return result


def hyper(job):
    base = round1_protocol()
    if job.stage == "fusion" and job.arm == "historical":
        return {"epochs": 100, "patience": 20, "lr": 1e-4, "batch": 8, "scheduler": "cosine_annealing_tmax100"}
    if job.stage == "fusion":
        return {**base["student"], "scheduler": "warm_restarts"}
    stage = "teacher2" if job.stage == "teacher2_indep" else job.stage
    return {**base[stage], "batch": 16, "scheduler": "warm_restarts"}


def protocol():
    return {
        "version": 1, "round1_run_id": ROUND1_RUN_ID, "seeds": SEEDS,
        "split": "frozen historical 2928/367/367 (split_manifest.json); test exposed historically",
        "selection": "validation QWK of the primary training output; no test-based choice",
        "reference": "round-1 raw_f100 teacher1/teacher2/joint_full/student_ce imported read-only with SHA-256 check; "
                     "retrained with unchanged round-1 code only if unavailable (recorded in reference_manifest.json)",
        "t2indep": "Teacher 2 (ResNet50+ViT-B/16, spectral-norm head) trained ALONE with class-weighted CE "
                   "(teacher-2 epochs/patience/lr); student joint_full on its raw logits (teacher_mode t2)",
        "nosn": "Teacher 2 without spectral normalization; same gated stage on the matched round-1 Teacher 1; "
                "classifier keys and initialization RNG stream differ from the SN model",
        "aug": "augmentation in ALL stages (teacher1, teacher2, student joint_full, student_ce): "
               "H/V flip .5, rotation +/-20 deg, brightness/contrast .9..1.1",
        "fusion_historical": "Table-1 Fusion CNN-ViT: torchvision ResNet50 (2048) + ViT-B/16 (768) features, MLP 1024-512-5, "
                             "dropout .3; CLAHE at cached 224 + ImageNet normalization; rotation 15 (nearest, fill 0); "
                             "weighted random sampler; unweighted CE; AdamW 1e-4 wd 1e-4; CosineAnnealingLR T_max 100; "
                             "100 epochs; patience 20; batch 8; clip 1. Historical run used its own split and full-resolution CLAHE.",
        "fusion_matched": "same Fusion CNN-ViT model with the student_ce protocol (raw 0..255, class-weighted CE, "
                          "70 epochs, patience 10, lr 2e-4, batch 16, warm restarts)",
        "student_teacher_buffers": "historical behavior: teacher parameters frozen, BN/SN buffers update during student training",
        "optimizer": "AdamW; weight decay 1e-4; clip norm 1; float32",
        "storage": "last.pth deleted after a job completes (resume only); best.pth kept",
        "test_policy": "evaluation stage only after all training jobs of the requested families are complete",
        "jobs": [asdict(job) | {"key": job.key, "dependency": job.dependency, "hyper": hyper(job)} for job in jobs()],
    }


def fingerprint():
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
    print(json.dumps({"jobs": len(jobs()), "protocol_sha256": fingerprint()}, indent=2))
