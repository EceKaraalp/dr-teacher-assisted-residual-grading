import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import preprocess
from plan import Run, matrix
from plan2 import Job, hyper, jobs
import analyze2
import external2
import run as runner
import run2


def bundle(n_per_class=10, size=8):
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(5), n_per_class * 3)
    images = rng.integers(0, 256, (len(labels), size, size, 3), dtype=np.uint8)
    ids = np.array([f"img{i}" for i in range(len(labels))])
    order = rng.permutation(len(labels))
    third = len(labels) // 3
    splits = {"train": np.sort(order[:third]), "val": np.sort(order[third:2 * third]), "test": np.sort(order[2 * third:])}
    return ids, images, labels, splits


class Round2Test(unittest.TestCase):
    def test_matrix(self):
        all_jobs = jobs()
        self.assertEqual(len(all_jobs), 42)
        self.assertEqual(len({j.key for j in all_jobs}), 42)
        seen = set()
        for job in all_jobs:
            if job.dependency:
                self.assertIn(job.dependency, seen, job.key)
            seen.add(job.key)
        round1 = {r.key for r in matrix()}
        references = [j for j in all_jobs if j.family == "reference"]
        self.assertEqual(len(references), 12)
        for job in references:
            self.assertEqual(job.key, "reference/" + job.legacy.key)
            self.assertIn(job.legacy.key, round1)
        self.assertEqual(sum(j.family != "reference" for j in all_jobs), 30)
        self.assertEqual(Job("nosn", 52, "teacher2").dependency, "reference/raw_f100_s52/teacher1")
        self.assertIsNone(Job("aug", 42, "student", "student_ce").dependency)
        self.assertTrue(Job("aug", 42, "teacher1").augment)
        self.assertFalse(Job("nosn", 42, "teacher2").augment)
        self.assertEqual(Job("fusion", 42, "fusion", "historical").preprocessing, "retinal")
        self.assertEqual(Job("fusion", 42, "fusion", "matched").preprocessing, "raw")

    def test_hyper(self):
        self.assertEqual(hyper(Job("t2indep", 42, "teacher2_indep"))["lr"], 1e-4)
        self.assertEqual(hyper(Job("t2indep", 42, "teacher2_indep"))["epochs"], 120)
        self.assertEqual(hyper(Job("aug", 42, "teacher1"))["lr"], 1e-3)
        self.assertEqual(hyper(Job("fusion", 42, "fusion", "historical"))["batch"], 8)
        self.assertEqual(hyper(Job("fusion", 42, "fusion", "matched"))["epochs"], 70)

    def test_losses(self):
        torch.manual_seed(0)
        z = torch.randn(6, 5)
        y = torch.tensor([0, 1, 2, 3, 4, 0])
        weights = torch.tensor([1., 2., 3., 4., 5.])
        expected = F.cross_entropy(z, y, weight=weights, reduction="none").mean()
        self.assertTrue(torch.allclose(run2.job_loss(Job("t2indep", 42, "teacher2_indep"), z, {}, y, weights, 0), expected))
        self.assertTrue(torch.allclose(run2.job_loss(Job("fusion", 42, "fusion", "historical"), z, {}, y, weights, 0),
                                       F.cross_entropy(z, y)))
        self.assertTrue(torch.allclose(run2.job_loss(Job("fusion", 42, "fusion", "matched"), z, {}, y, weights, 0), expected))
        extra = {"base_logits": torch.randn(6, 5), "gate": torch.full((6, 1), .4)}
        self.assertTrue(torch.allclose(
            run2.job_loss(Job("aug", 42, "teacher2"), z, extra, y, weights, 7),
            runner.compute_loss(z, extra, y, weights, Run("raw", 1.0, 42, "teacher2"), 7)))
        extra = {"z_teacher": torch.randn(6, 5), "gate": torch.full((6, 1), .4)}
        self.assertTrue(torch.allclose(
            run2.job_loss(Job("nosn", 42, "student", "joint_full"), z, extra, y, weights, 3),
            runner.compute_loss(z, extra, y, weights, Run("raw", 1.0, 42, "student", "joint_full"), 3)))

    def test_loaders(self):
        data = bundle()
        loaders, weights, train_ids = run2.job_loaders(Job("aug", 42, "student", "student_ce"), data)
        self.assertTrue(loaders["train"].dataset.augment)
        self.assertFalse(loaders["val"].dataset.augment)
        self.assertEqual(len(train_ids), len(data[3]["train"]))
        loaders, _, _ = run2.job_loaders(Job("nosn", 42, "student", "joint_full"), data)
        self.assertFalse(loaders["train"].dataset.augment)
        self.assertEqual(loaders["train"].dataset.mode, "raw")
        loaders, weights, _ = run2.job_loaders(Job("fusion", 42, "fusion", "historical"), data)
        self.assertEqual(loaders["train"].batch_size, 8)
        self.assertIsInstance(loaders["train"].sampler, torch.utils.data.WeightedRandomSampler)
        x, _ = loaders["val"].dataset[0]
        torch.testing.assert_close(x, preprocess(data[1][data[3]["val"][0]], "retinal"))
        x, _ = loaders["train"].dataset[0]
        self.assertEqual(tuple(x.shape), (3, 8, 8))

    def test_single_teacher_matches_t2_mode(self):
        class Adv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 5)
            def forward(self, x):
                return self.linear(x), {}
        teacher = run2.SingleTeacher(Adv())
        z, extra = teacher(torch.randn(3, 4))
        self.assertEqual(tuple(extra["gate"].shape), (3, 1))
        torch.testing.assert_close(z, extra["adv_logits"])

    def test_reference_import_verifies_and_never_writes_round1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            round1 = root / "round1"
            good = Job("reference", 42, "teacher1")
            bad = Job("reference", 52, "teacher1")
            for job, tamper in ((good, False), (bad, True)):
                folder = round1 / job.legacy.key
                folder.mkdir(parents=True)
                torch.save({"model": {"w": torch.ones(2)}}, folder / "best.pth")
                digest = runner.file_hash(folder / "best.pth")
                (folder / "complete.json").write_text(json.dumps({"best_sha256": "0" * 64 if tamper else digest}))
                np.savez_compressed(folder / "test_logits.npz", labels=np.arange(3))
            before = {p: runner.file_hash(p) for p in round1.rglob("*") if p.is_file()}
            args = SimpleNamespace(output=root / "local", drive=root / "drive", round1=round1)
            args.output.mkdir()
            manifest = run2.reference_import(args)
            self.assertEqual(manifest[good.key]["mode"], "imported")
            self.assertTrue((args.output / good.key / "test_logits.npz").is_file())
            self.assertEqual(manifest[bad.key]["mode"], "unavailable_retrain_required")
            self.assertFalse((args.output / bad.key / "best.pth").exists())
            self.assertEqual(manifest[Job("reference", 62, "student", "student_ce").key]["mode"], "unavailable_retrain_required")
            self.assertTrue((args.drive / "reference_manifest.json").is_file())
            self.assertEqual(before, {p: runner.file_hash(p) for p in round1.rglob("*") if p.is_file()})

    def test_interrupted_resume_matches_uninterrupted_and_prunes_last(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Dropout(.2), torch.nn.Linear(8, 5))
            def forward(self, x):
                return self.net(x)
        x, y = torch.eye(5).repeat(4, 1), torch.arange(5).repeat(4)
        def data_loader(job, data):
            dataset = TensorDataset(x, y)
            return {"train": DataLoader(dataset, batch_size=5, shuffle=True),
                    "val": DataLoader(dataset, batch_size=5)}, torch.ones(5), list(map(str, range(20)))
        def build(job, output, checkpoint=None):
            runner.P.set_seed(job.seed)
            model = Tiny()
            if checkpoint:
                runner.restore_state(model, checkpoint["model"])
            return model
        def interrupted_sync(folder, remote):
            if (folder / "last.pth").exists() and json.loads((folder / "history.json").read_text())[-1]["epoch"] == 2:
                raise InterruptedError("Simulated runtime reset after persisted epoch.")
        job = Job("fusion", 42, "fusion", "matched")
        settings = {"epochs": 4, "patience": 10, "lr": .01, "batch": 5, "scheduler": "warm_restarts"}
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, "DEVICE", torch.device("cpu")), \
                patch.object(run2, "job_loaders", data_loader), patch.object(run2, "build", build), \
                patch.object(run2, "hyper", return_value=settings):
            root = Path(temporary)
            args = SimpleNamespace(output=root / "complete", drive=None, keep_last=True)
            run2.train_job(job, args, None)
            expected = torch.load(args.output / job.key / "last.pth", weights_only=False)
            args = SimpleNamespace(output=root / "resumed", drive=None, keep_last=False)
            with patch.object(runner, "synchronize", interrupted_sync), self.assertRaises(InterruptedError):
                run2.train_job(job, args, None)
            self.assertTrue((args.output / job.key / "last.pth").exists())
            run2.train_job(job, args, None)
            self.assertFalse((args.output / job.key / "last.pth").exists())
            best_expected = torch.load(root / "complete" / job.key / "best.pth", weights_only=False)
            best_actual = torch.load(args.output / job.key / "best.pth", weights_only=False)
            for key in best_expected["model"]:
                torch.testing.assert_close(best_expected["model"][key], best_actual["model"][key], rtol=0, atol=0)
            self.assertEqual(json.loads((root / "complete" / job.key / "history.json").read_text())[-1]["val_qwk"],
                             json.loads((args.output / job.key / "history.json").read_text())[-1]["val_qwk"])
            self.assertEqual(expected["epoch"], 3)
            self.assertFalse(run2.train_job(job, args, None))

    def test_external_label_formats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            buffer = io.BytesIO()
            Image.new("RGB", (40, 30), (10, 20, 30)).save(buffer, format="PNG")
            with zipfile.ZipFile(root / "images.zip", "w") as archive:
                for name in ("IMAGES/a.png", "IMAGES/b.PNG", "IMAGES/c.jpg", "__MACOSX/IMAGES/._a.png"):
                    archive.writestr(name, buffer.getvalue())
            (root / "messidor.csv").write_text(
                "image_id,adjudicated_dr_grade,adjudicated_dme,adjudicated_gradable\n"
                "a.png,0,0,1\nb.png,3,1,1\nc.jpg,,0,0\n", encoding="utf-8")
            x, y, ids, info = external2.load_external(root / "images.zip", root / "messidor.csv", "messidor2")
            self.assertEqual(ids, ["a", "b"])
            self.assertEqual(y.tolist(), [0, 3])
            self.assertEqual(x.shape, (2, 224, 224, 3))
            self.assertEqual(info["ungradable_or_invalid"], 1)
            (root / "ddr.txt").write_text("a.jpg 4\nb.jpg 5\nc.jpg 1\n", encoding="utf-8")
            _, y, ids, info = external2.load_external(root / "images.zip", root / "ddr.txt", "ddr")
            self.assertEqual((ids, y.tolist(), info["ungradable_or_invalid"]), (["a", "c"], [4, 1], 1))
            folder = root / "folder"
            folder.mkdir()
            (folder / "a.png").write_bytes(buffer.getvalue())
            (root / "generic.csv").write_text("image,grade\na,2\n", encoding="utf-8")
            _, y, ids, _ = external2.load_external(folder, root / "generic.csv", "generic")
            self.assertEqual((ids, y.tolist()), (["a"], [2]))
            (root / "missing.csv").write_text("image,grade\nzzz,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                external2.load_external(folder, root / "missing.csv", "generic")

    def test_analysis_contract(self):
        rng = np.random.default_rng(1)
        y = np.arange(5).repeat(6)
        ids = np.array([f"t{i}" for i in range(len(y))])
        with tempfile.TemporaryDirectory() as temporary:
            output, round1 = Path(temporary) / "out", Path(temporary) / "round1"
            for job in jobs():
                folder = output / job.key
                folder.mkdir(parents=True)
                paths = {"primary": rng.normal(size=(len(y), 5))}
                if job.stage == "teacher2":
                    paths.update({k: rng.normal(size=(len(y), 5)) for k in ("teacher1", "teacher2")})
                if job.stage == "student" and job.arm == "joint_full":
                    paths.update({k: rng.normal(size=(len(y), 5)) for k in ("teacher", "student", "equal")})
                for split in ("val", "test"):
                    np.savez_compressed(folder / f"round2_{split}_logits.npz", labels=y, ids=ids, **paths)
            for run in matrix():
                if run.stage == "student" and run.arm in {"joint_full", "student_ce"} and run.preprocessing == "raw":
                    folder = round1 / run.key
                    folder.mkdir(parents=True)
                    paths = {"primary": rng.normal(size=(len(y), 5))}
                    if run.arm == "joint_full":
                        paths["equal"] = rng.normal(size=(len(y), 5))
                    np.savez_compressed(folder / "test_logits.npz", labels=y, ids=ids, **paths)
            analyze2.analyze(output, round1, None, replicates=50)
            status = json.loads((output / "analysis2" / "status.json").read_text())
            self.assertTrue(status["complete"])
            contrasts = json.loads((output / "analysis2" / "paired_qwk.json").read_text())
            self.assertEqual(set(contrasts), set(analyze2.CONTRASTS))
            self.assertTrue(all("mean_difference" in v for v in contrasts.values()))
            budgets = json.loads((output / "analysis2" / "label_budget_round1.json").read_text())
            self.assertEqual(set(budgets), {"fraction_0.1", "fraction_0.25", "fraction_0.5", "fraction_1.0"})
            self.assertIn("gated_vs_ce", budgets["fraction_0.1"])
            values = json.loads((output / "analysis2" / "mean_sd.json").read_text())
            self.assertEqual(values["reference/student/joint_full"]["primary"]["qwk"]["n_seeds"], 3)
            self.assertIn("| reference/gated_vs_ce |", (output / "analysis2" / "summary.md").read_text())

    def test_external_contrasts(self):
        rng = np.random.default_rng(2)
        y = np.arange(5).repeat(4)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            for job in external2.external_jobs():
                paths = {"primary": rng.normal(size=(len(y), 5))}
                if job.stage == "student" and job.arm == "joint_full":
                    paths["equal"] = rng.normal(size=(len(y), 5))
                np.savez_compressed(destination / (job.key.replace("/", "__") + ".npz"), labels=y, **paths)
            with patch.object(external2, "bootstrap", side_effect=lambda a, b, c: {"mean_difference": 0.0}):
                result = external2.contrasts(destination, y)
            self.assertTrue(all("mean_difference" in v for v in result.values()))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
