from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from logit_fusion import (LogitFusionHead, fusion_features,
                          load_fusion_checkpoint, normalise_probability)


class LogitFusionTests(unittest.TestCase):
    def test_zero_initialisation_is_exact_equal_probability_average(self):
        generator = torch.Generator().manual_seed(7)
        b3 = torch.rand(2, 4, 13, 17, generator=generator)
        se = torch.rand(2, 4, 13, 17, generator=generator)
        model = LogitFusionHead(classes=4, hidden=8)
        actual = model(b3, se).softmax(1)
        expected = (normalise_probability(b3) + normalise_probability(se)) * 0.5
        self.assertTrue(torch.allclose(actual, expected, atol=2e-7, rtol=2e-6))

    def test_features_have_logits_and_two_finite_entropies(self):
        b3 = torch.zeros(1, 4, 5, 6)
        se = torch.zeros(1, 4, 5, 6)
        b3[:, 0], se[:, 3] = 1, 1
        features = fusion_features(b3, se)
        self.assertEqual(tuple(features.shape), (1, 10, 5, 6))
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.allclose(features[:, -2:], torch.zeros_like(features[:, -2:]),
                                       atol=1e-5))

    def test_checkpoint_roundtrip(self):
        model = LogitFusionHead(classes=4, hidden=6)
        with tempfile.TemporaryDirectory(prefix="komap_fusion_") as temporary:
            path = Path(temporary) / "fusion.pth"
            torch.save({"format": "komap_logit_fusion_v1", "model": model.state_dict(),
                        "classes": 4, "hidden": 6,
                        "inference_signature": {"tile": 768, "overlap": 0.5,
                                                "flip_tta": True, "d4": True,
                                                "scale_tta": [0.875, 1.0, 1.125]}}, path)
            loaded, value = load_fusion_checkpoint(path)
            self.assertEqual(loaded.hidden, 6)
            self.assertEqual(value["inference_signature"]["tile"], 768)
            x = torch.from_numpy(np.full((1, 4, 4, 3), 0.25, np.float32))
            self.assertTrue(torch.equal(model(x, x), loaded(x, x)))


if __name__ == "__main__":
    unittest.main()
