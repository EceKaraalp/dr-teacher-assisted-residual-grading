import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from data import nested_subset, preprocess
from plan import Run, loss_config, matrix
from run import compute_loss, model_state, restore_state, synchronize
from analyze import fit_temperature, metrics, resampled_qwk
from sklearn.metrics import cohen_kappa_score
import run as runner
import analyze as analysis
from torch.utils.data import DataLoader, TensorDataset


class ExperimentsTest(unittest.TestCase):
    def test_matrix_complete_and_dependencies(self):
        runs = matrix()
        self.assertEqual(len(runs), 99)
        self.assertEqual(len({r.key for r in runs}), 99)
        visited = set()
        for run in runs:
            if run.dependency:
                self.assertIn(run.dependency, visited)
            visited.add(run.key)
        for fraction in (.1, .25, .5, 1.0):
            for seed in (42, 52, 62):
                for stage, arm in (("teacher1", ""), ("teacher2", ""), ("student", "joint_full"), ("student", "student_ce")):
                    self.assertIn(Run("raw", fraction, seed, stage, arm).key, visited)

    def test_nested_label_budget(self):
        y = np.repeat(np.arange(5), 100)
        train = np.arange(500)
        subsets = [nested_subset(train, y, f, 42) for f in (.1, .25, .5, 1)]
        for left, right in zip(subsets, subsets[1:]):
            self.assertTrue(set(left) <= set(right))
        self.assertEqual(len(subsets[0]), 50)
        self.assertTrue(np.array_equal(subsets[0], nested_subset(train, y, .1, 42)))

    def test_preprocessing(self):
        image = np.full((224, 224, 3), 255, dtype=np.uint8)
        self.assertEqual(float(preprocess(image, "raw").max()), 255)
        expected = (1 - .485) / .229
        self.assertAlmostEqual(float(preprocess(image, "imagenet")[0, 0, 0]), expected, places=5)
        retinal = preprocess(image, "retinal")
        self.assertEqual(tuple(retinal.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(retinal).all())
        with self.assertRaises(ValueError):
            preprocess(image, "unknown")

    def test_ce_and_detached_reference(self):
        z = torch.randn(6, 5, requires_grad=True)
        t = torch.randn(6, 5, requires_grad=True)
        y, weights = torch.tensor([0, 1, 2, 3, 4, 0]), torch.ones(5)
        extra = {"z_teacher": t, "gate": torch.full((6, 1), .5, requires_grad=True)}
        ce = compute_loss(z, extra, y, weights, Run("raw", 1, 42, "student", "joint_ce"), 0)
        self.assertTrue(torch.allclose(ce, F.cross_entropy(z, y)))
        full = compute_loss(z, extra, y, weights, Run("raw", 1, 42, "student", "joint_full"), 0)
        full.backward()
        self.assertIsNone(t.grad)
        self.assertIsNotNone(z.grad)
        self.assertEqual(loss_config("joint_single_kd")["temperatures"], [2.0])
        self.assertEqual(loss_config("joint_full_no_entropy")["entropy_coefficient"], 0)

    def test_hinge_sign(self):
        z = torch.zeros(5, 5)
        y, weights = torch.arange(5), torch.ones(5)
        extra = {"z_teacher": z.clone(), "gate": torch.full((5, 1), .5)}
        run = Run("raw", 1, 42, "student", "joint_hinge")
        baseline = compute_loss(z, extra, y, weights, Run("raw", 1, 42, "student", "joint_multi_kd"), 0)
        self.assertAlmostEqual(float(compute_loss(z, extra, y, weights, run, 0) - baseline), .8 * .01, places=6)

    def test_teacher_buffers_checkpointed(self):
        model = torch.nn.Module()
        model.teacher_model = torch.nn.BatchNorm1d(5)
        model.student = torch.nn.Linear(5, 5)
        state = model_state(model, "student")
        self.assertNotIn("teacher_model.weight", state)
        self.assertIn("teacher_model.running_mean", state)
        state["teacher_model.running_mean"] = torch.ones(5)
        restore_state(model, state)
        self.assertTrue(torch.equal(model.teacher_model.running_mean, torch.ones(5)))

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local, remote = root / "local", root / "remote"
            local.mkdir()
            (local / "history.json").write_text(json.dumps({"epoch": 1}))
            synchronize(local, remote)
            self.assertEqual((remote / "history.json").read_bytes(), (local / "history.json").read_bytes())

    def test_temperature_validation_nll(self):
        rng = np.random.default_rng(42)
        z = rng.normal(size=(100, 5)) * 5
        y = rng.integers(0, 5, 100)
        temperature = fit_temperature(z, y)
        before = F.cross_entropy(torch.tensor(z), torch.tensor(y))
        after = F.cross_entropy(torch.tensor(z / temperature), torch.tensor(y))
        self.assertLessEqual(float(after), float(before) + 1e-8)
        self.assertTrue(.05 <= temperature <= 20)

    def test_metric_shapes(self):
        y = np.arange(5)
        p = np.eye(5) * .9 + .02
        result = metrics(y, p)
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["qwk"], 1)

    def test_vector_bootstrap_matches_metric_definition(self):
        rng = np.random.default_rng(42)
        y, pred = np.arange(5).repeat(20), rng.integers(0, 5, 100)
        indices = rng.integers(0, len(y), (50, len(y)))
        expected = [cohen_kappa_score(y[idx], pred[idx], labels=range(5), weights="quadratic") for idx in indices]
        np.testing.assert_allclose(resampled_qwk(y, pred, indices), expected, atol=1e-14)

    def test_interrupted_resume_matches_uninterrupted(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Dropout(.2), torch.nn.Linear(8, 5))
            def forward(self, x):
                return self.net(x)
        x, y = torch.eye(5).repeat(4, 1), torch.arange(5).repeat(4)
        def data_loader(bundle, run):
            dataset = TensorDataset(x, y)
            return {"train": DataLoader(dataset, batch_size=5, shuffle=True),
                    "val": DataLoader(dataset, batch_size=5)}, torch.ones(5), list(map(str, range(20)))
        def build(run, output, checkpoint):
            runner.P.set_seed(run.seed)
            model = Tiny()
            if checkpoint:
                runner.restore_state(model, checkpoint["model"])
            return model
        def interrupted_sync(folder, remote):
            if (folder / "last.pth").exists():
                raise InterruptedError("Simulated runtime reset after persisted epoch.")
        bundle = (np.array(list(map(str, range(20)))), None, y.numpy())
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, "DEVICE", torch.device("cpu")), \
                patch.object(runner, "loaders", data_loader), patch.object(runner, "build_model", build), \
                patch.object(runner, "protocol", return_value={"teacher1": {"epochs": 2, "patience": 3, "lr": .001}}):
            root = Path(temporary)
            run = Run("raw", 1, 42, "teacher1")
            args = SimpleNamespace(output=root / "complete", drive=None)
            runner.train(run, args, bundle)
            expected = torch.load(args.output / run.key / "last.pth", weights_only=False)
            args.output = root / "resumed"
            with patch.object(runner, "synchronize", interrupted_sync), self.assertRaises(InterruptedError):
                runner.train(run, args, bundle)
            runner.train(run, args, bundle)
            actual = torch.load(args.output / run.key / "last.pth", weights_only=False)
            for key in expected["model"]:
                torch.testing.assert_close(expected["model"][key], actual["model"][key], rtol=0, atol=0)
            self.assertEqual(expected["best"], actual["best"])
            self.assertTrue((args.output / run.key / "complete.json").exists())

    def test_analysis_full_matrix_output_contract(self):
        rng = np.random.default_rng(42)
        y = np.arange(5).repeat(5)
        ids = np.array([f"test_{i}" for i in range(len(y))])
        original_bootstrap = analysis.bootstrap
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for run in matrix():
                folder = output / run.key
                folder.mkdir(parents=True)
                payload = {"primary": rng.normal(size=(len(y), 5))}
                if run.stage == "teacher2":
                    payload.update({name: rng.normal(size=(len(y), 5)) for name in ("teacher1", "teacher2")})
                if run.stage == "student" and run.arm != "student_ce":
                    payload.update({name: rng.normal(size=(len(y), 5)) for name in ("teacher", "student", "equal")})
                for split in ("val", "test"):
                    np.savez_compressed(folder / f"{split}_logits.npz", labels=y, ids=ids, **payload)
            # Numerical paths run fully; figure rendering is covered by the real-model smoke.
            with patch.object(analysis, "reliability"), patch.object(analysis.plt, "subplots", return_value=(MagicMock(), MagicMock())), \
                    patch.object(analysis.plt, "close"), patch.object(
                        analysis, "bootstrap", side_effect=lambda y, a, b: original_bootstrap(y, a, b, replicates=100)):
                analysis.analyze(output)
            report = json.loads((output / "analysis" / "per_seed.json").read_text())
            self.assertEqual(len(report), 99)
            contrasts = json.loads((output / "analysis" / "paired_qwk.json").read_text())
            self.assertIn("teacher1_contribution/equal", contrasts)
            self.assertIn("fraction_0.1/full_vs_ce", contrasts)
            self.assertIn("equal_vs_independent_ce", contrasts)
            self.assertTrue(json.loads((output / "analysis" / "status.json").read_text())["complete"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
