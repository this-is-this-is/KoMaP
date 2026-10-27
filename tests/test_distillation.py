from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from komap import COLORS
from sampling import ClassAwareCropper, grid_crop
from train_v2 import CropDataset, distillation_kl


class DistillationTests(unittest.TestCase):
    def test_teacher_map_uses_exact_same_crop(self):
        mask = np.zeros((30, 34), np.uint8)
        mask[4:20, 8:25] = 1
        mask[13:27, 17:31] = 2
        image = np.repeat((mask * 70)[..., None], 3, 2).astype(np.uint8)
        teacher = np.eye(4, dtype=np.float32)[mask]
        cropper = ClassAwareCropper(20, class_ratio=0,
                                    rng=np.random.default_rng(4))
        _, cropped_mask, cropped_teacher = cropper(image, mask, teacher=teacher)
        self.assertTrue(np.array_equal(cropped_teacher.argmax(2), cropped_mask))

        _, grid_mask, grid_teacher = grid_crop(image, mask, 40, 0, teacher=teacher)
        self.assertTrue(np.array_equal(grid_teacher.argmax(2), grid_mask))

    def test_matching_soft_distribution_has_zero_kl(self):
        teacher = torch.tensor([0.7, 0.1, 0.1, 0.1])[None, :, None, None].expand(1, 4, 3, 3)
        matching = teacher.log()
        wrong = torch.roll(matching, 1, dims=1)
        valid = torch.ones(1, 3, 3)
        boundary = torch.zeros_like(valid)
        boundary[:, 1] = 1
        exact, exact_boundary = distillation_kl(
            matching, teacher, valid, boundary, temperature=2, confidence=0)
        mismatch, mismatch_boundary = distillation_kl(
            wrong, teacher, valid, boundary, temperature=2, confidence=0)
        self.assertLess(abs(float(exact)), 1e-6)
        self.assertLess(abs(float(exact_boundary)), 1e-6)
        self.assertGreater(float(mismatch), 0.1)
        self.assertGreater(float(mismatch_boundary), 0.1)

    def test_dataset_loads_cached_probability_contract(self):
        with tempfile.TemporaryDirectory(prefix="komap_kd_") as temporary:
            root = Path(temporary)
            images = root / "train" / "train" / "images"
            masks = root / "train" / "train" / "masks"
            cache = root / "teacher"
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            cache.mkdir()
            name = "A001_step1_sample1"
            image_path = images / f"{name}_image.png"
            mask_path = masks / f"{name}_mask.png"
            image = np.full((32, 36, 3), 150, np.uint8)
            mask = np.zeros((32, 36), np.uint8)
            mask[7:21, 9:25] = 1
            mask[16:27, 20:31] = 2
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mask_path), cv2.cvtColor(COLORS[mask], cv2.COLOR_RGB2BGR))
            probability = np.eye(4, dtype=np.float32)[mask].transpose(2, 0, 1)
            np.savez_compressed(cache / f"{name}_prob.npz", prob=probability.astype(np.float16),
                                image=str(image_path))

            dataset = CropDataset(root / "train" / "train", tile=16, class_ratio=0,
                                  samples_per_image=1, geo_aug="none", color_aug="none",
                                  teacher_prob_dir=cache)
            item = dataset[0]
            self.assertEqual(tuple(item["teacher"].shape), (4, 16, 16))
            self.assertTrue(torch.equal(item["teacher"].argmax(0), item["mask"]))
            self.assertTrue(torch.all(item["teacher_valid"] == 1))
            self.assertEqual(tuple(item["teacher_boundary"].shape), (16, 16))


if __name__ == "__main__":
    unittest.main()
