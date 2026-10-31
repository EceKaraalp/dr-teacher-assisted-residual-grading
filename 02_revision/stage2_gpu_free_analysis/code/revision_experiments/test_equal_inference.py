import unittest

import torch
from torch import nn

from evaluate_idrid import inference_logits
from summarize_equal_inference import wilson


class ConstantBranch(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor([logits], dtype=torch.float32))

    def forward(self, images):
        return self.logits.expand(len(images), -1), {}


class InferenceFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher_model = ConstantBranch([1, 2, 3, 4, 5])
        self.student_model = ConstantBranch([5, 4, 3, 2, 1])
        self.outer_forward_calls = 0

    def forward(self, images):
        self.outer_forward_calls += 1
        return torch.full((len(images), 5), 19.0), {}


class EqualInferenceTests(unittest.TestCase):
    def test_equal_average_bypasses_outer_forward(self):
        model = InferenceFixture()
        logits = inference_logits(model, torch.zeros(2, 3, 224, 224), "equal")
        torch.testing.assert_close(logits, torch.full((2, 5), 3.0))
        self.assertEqual(model.outer_forward_calls, 0)

    def test_residual_mode_preserves_original_forward(self):
        model = InferenceFixture()
        logits = inference_logits(model, torch.zeros(1, 3, 224, 224), "residual")
        torch.testing.assert_close(logits, torch.full((1, 5), 19.0))
        self.assertEqual(model.outer_forward_calls, 1)

    def test_unknown_mode_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown inference mode"):
            inference_logits(InferenceFixture(), torch.zeros(1, 3, 224, 224), "invalid")

    def test_wilson_clinical_interval(self):
        low, high = wilson(6, 19)
        self.assertAlmostEqual(low, 0.1536437908779228)
        self.assertAlmostEqual(high, 0.5398959232467253)
        with self.assertRaises(ValueError):
            wilson(0, 0)


if __name__ == "__main__":
    unittest.main()
