"""Actual architecture forward/backward check; does not produce training results."""
import argparse
import gc
from pathlib import Path
import tempfile

import torch

import dr_pipeline as P
from plan import Run
from run import compute_loss, forward_paths, model_state, restore_state, teacher_modules


def smoke(pretrained=False):
    torch.set_num_threads(2)
    device = P.device
    if pretrained and device.type != "cuda":
        raise RuntimeError("Colab pretrained smoke requires a GPU.")
    teacher = teacher_modules(42, pretrained=pretrained).to(device)
    student = P.ConvNeXtTinyStudent(5, pretrained=pretrained).to(device)
    expected = (4851623, 109320773, 3009, 27823973)
    actual = tuple(sum(p.numel() for p in m.parameters()) for m in (
        teacher.baseline_model, teacher.adv_model, teacher.gate, student))
    if actual != expected:
        raise AssertionError(f"Architecture drift: {actual} != {expected}")
    x = torch.rand(2, 3, 224, 224, device=device) * 255
    y, weights = torch.tensor([0, 1], device=device), torch.ones(5, device=device)
    for parameter in teacher.baseline_model.parameters():
        parameter.requires_grad_(True)
    base_logits = teacher.baseline_model(x)
    base_loss = compute_loss(base_logits, {}, y, weights, Run("raw", 1, 42, "teacher1"), 0)
    base_loss.backward()
    if not torch.isfinite(base_loss) or not any(p.grad is not None for p in teacher.baseline_model.parameters()):
        raise AssertionError("Teacher-1 training failed.")
    teacher.baseline_model.zero_grad(set_to_none=True)
    for parameter in teacher.baseline_model.parameters():
        parameter.requires_grad_(False)
    teacher.train()
    z, extra = teacher(x)
    loss = compute_loss(z, extra, y, weights, Run("raw", 1, 42, "teacher2"), 0)
    loss.backward()
    if not torch.isfinite(loss) or not any(p.grad is not None for p in teacher.adv_model.parameters()):
        raise AssertionError("Teacher training did not produce finite loss and advanced-model gradients.")
    if any(p.grad is not None for p in teacher.baseline_model.parameters()):
        raise AssertionError("Baseline teacher unexpectedly received gradients.")
    teacher.zero_grad(set_to_none=True)
    model = P.ResidualTeacherStudentModel(teacher, student, mix="gated", teacher_mode="gate", **P.STUDENT_MODEL_KW).to(device)
    if any(parameter.device != x.device for parameter in model.parameters()):
        raise AssertionError("Combined model parameters and input must share the same device.")
    model.train()
    paths, extra = forward_paths(model, x, Run("raw", 1, 42, "student", "joint_full"))
    loss = compute_loss(paths["primary"], extra, y, weights, Run("raw", 1, 42, "student", "joint_full"), 0)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None for p in teacher.parameters()):
        raise AssertionError("Student loss is nonfinite or teacher gradient stop failed.")
    if not any(p.grad is not None for p in student.parameters()):
        raise AssertionError("Student received no gradients.")
    model.eval()
    with torch.inference_mode():
        original, _ = model(x)
        state = {k: v.clone() for k, v in model_state(model, "student").items()}
        restore_state(model, state)
        restored, _ = model(x)
    torch.testing.assert_close(original, restored, rtol=0, atol=0)
    from external import gradcam
    from analyze import reliability, probabilities
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary)
        image = x[0].detach().cpu().permute(1, 2, 0).numpy().astype("uint8")
        attribution = gradcam(model, image, 0, "smoke", destination)
        reliability(y.cpu().numpy(), probabilities(original.detach().cpu().numpy()), destination / "reliability.png", "Smoke")
        if not (destination / "smoke_gradcam.png").is_file() or not (destination / "reliability.png").is_file():
            raise AssertionError("Figure rendering failed.")
        if len(attribution["branches"]) != 3:
            raise AssertionError("Missing Grad-CAM branches.")
    print({"smoke": "PASS", "device": str(device), "pretrained": pretrained,
           "parameters": actual, "total_with_outer_gate": sum(p.numel() for p in model.parameters()),
           "student_loss": float(loss.detach())}, flush=True)
    del teacher, student, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    smoke(args.pretrained)
