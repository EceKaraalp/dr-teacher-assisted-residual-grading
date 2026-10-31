"""Real-architecture forward/backward checks through run2.build for every new round-2 job type; no results."""
import argparse
import gc
from pathlib import Path
import tempfile

import numpy as np
import torch

import dr_pipeline as P
import run as runner
import run2
from plan2 import Job


def parameters(module):
    return sum(p.numel() for p in module.parameters())


def check_step(job, model, x, y, weights):
    model.train()
    paths, extra = runner.forward_paths(model, x, job.legacy)
    loss = run2.job_loss(job, paths["primary"], extra, y, weights, 0)
    loss.backward()
    if not torch.isfinite(loss) or not any(p.grad is not None for p in model.parameters() if p.requires_grad):
        raise AssertionError(f"{job.key}: nonfinite loss or no gradients")
    model.zero_grad(set_to_none=True)
    return float(loss.detach())


def mark_complete(root, job, model):
    folder = Path(root) / job.key
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"model": runner.model_state(model, run2.checkpoint_stage(job)), "epoch": 0, "qwk": 0.0}, folder / "best.pth")
    runner.write_json(folder / "complete.json", {"best_sha256": run2.file_hash(folder / "best.pth")})


def smoke(pretrained=False):
    torch.set_num_threads(2)
    run2.PRETRAINED = pretrained
    device = run2.device()
    if pretrained and device.type != "cuda":
        raise RuntimeError("Pretrained smoke requires the Colab GPU.")
    x = torch.rand(2, 3, 224, 224, device=device) * 255
    y, weights = torch.tensor([0, 3], device=device), torch.ones(5, device=device)
    report = {}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        sequence = [Job("aug", 42, "teacher1"), Job("t2indep", 42, "teacher2_indep"), Job("nosn", 42, "teacher2"),
                    Job("t2indep", 42, "student", "joint_full"), Job("nosn", 42, "student", "joint_full"),
                    Job("aug", 42, "student", "student_ce"), Job("fusion", 42, "fusion", "historical")]
        reference_t1 = Job("reference", 42, "teacher1")
        for job in sequence:
            model = run2.build(job, root)
            report[job.key] = {"loss": check_step(job, model, x, y, weights), "parameters": parameters(model)}
            if job.stage == "teacher2" and any(p.grad is not None for p in model.baseline_model.parameters()):
                raise AssertionError("Teacher 1 received gradients.")
            if job.family == "nosn" and job.stage == "teacher2":
                if any("weight_orig" in k for k in model.adv_model.state_dict()):
                    raise AssertionError("No-SN teacher still has spectral normalization.")
                report["nosn_t2_parameters"] = parameters(model.adv_model)
            if job.stage == "teacher2_indep":
                if not any("weight_orig" in k for k in model.state_dict()):
                    raise AssertionError("Independent Teacher 2 must keep spectral normalization.")
                report["sn_t2_parameters"] = parameters(model)
            if job.stage == "student" and job.arm == "joint_full":
                if any(p.grad is not None for p in model.teacher_model.parameters()):
                    raise AssertionError("Teacher received gradients in the student stage.")
                model.eval()
                with torch.inference_mode():
                    before = model(x)[0]
                    runner.restore_state(model, {k: v.clone() for k, v in runner.model_state(model, "student").items()})
                    after = model(x)[0]
                torch.testing.assert_close(before, after, rtol=0, atol=0)
            mark_complete(root, job, model)
            if job == sequence[0]:
                mark_complete(root, reference_t1, model)
            del model
            gc.collect()
        from external2 import gated_gradcam
        teacher = runner.teacher_modules(42, pretrained=pretrained)
        model = P.ResidualTeacherStudentModel(teacher, P.ConvNeXtTinyStudent(5, pretrained=pretrained), mix="gated",
                                              teacher_mode="gate", **P.STUDENT_MODEL_KW).to(device)
        image = (np.random.default_rng(0).random((224, 224, 3)) * 255).astype("uint8")
        attribution = gated_gradcam(model, image, 0, "smoke", root)
        if len(attribution["branches"]) != 3 or not (root / "smoke_gated_gradcam.png").is_file():
            raise AssertionError("Gated Grad-CAM failed.")
        report["gradcam_forward_difference"] = attribution["forward_max_abs_difference"]
        del model, teacher
    if report["sn_t2_parameters"] != report["nosn_t2_parameters"]:
        raise AssertionError("SN and no-SN Teacher 2 must have the same number of trainable weights.")
    print({"smoke2": "PASS", "device": str(device), "pretrained": pretrained, **report}, flush=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true")
    smoke(parser.parse_args().pretrained)
