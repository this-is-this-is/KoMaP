import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from audit_oof_labels import find_regions, read_oof
from group_cv import (discover_labelled, grouped_folds, prepare_manifests,
                      read_manifest, specimen_group)
from komap import COLORS
from predict import inference_views


def labelled_scan(root: Path, split: str, name: str, label=0):
    images = root / split / "images"
    masks = root / split / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    image_name = f"{name}_image.png"
    mask_name = f"{name}_mask.png"
    image = np.full((32, 40, 3), 80, np.uint8)
    mask = np.broadcast_to(COLORS[label], (32, 40, 3)).copy()
    cv2.imwrite(str(images / image_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(masks / mask_name), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
    return images / image_name


class GroupCrossValidationTests(unittest.TestCase):
    def test_specimen_prefix_is_strict(self):
        self.assertEqual(specimen_group("A125_step1_image.png"), "A125")
        with self.assertRaises(ValueError):
            specimen_group("sample_A125_image.png")

    def test_each_group_is_held_out_once_without_leakage(self):
        records = []
        for group in range(12):
            for scan in range(1 + (group % 3 == 0)):
                records.append({"name": f"A{group:03d}_{scan}_image.png",
                                "group": f"A{group:03d}", "image": "unused",
                                "source_split": "train/train"})
        folds = grouped_folds(records, n_splits=5, seed=7)
        held_out = []
        for fold in folds:
            self.assertFalse(set(fold["train_groups"]) & set(fold["valid_groups"]))
            held_out.extend(fold["valid_indices"])
        self.assertEqual(sorted(held_out), list(range(len(records))))

    def test_manifests_merge_old_train_and_valid(self):
        with tempfile.TemporaryDirectory(prefix="komap_group_cv_") as temporary:
            root = Path(temporary)
            # A002 deliberately appears in both old splits. Grouping, not the
            # old directory, decides which new fold owns both scans.
            labelled_scan(root, "train/train", "A001_first")
            labelled_scan(root, "train/train", "A002_first")
            labelled_scan(root, "valid/valid", "A002_second")
            labelled_scan(root, "valid/valid", "A003_first")
            labelled_scan(root, "valid/valid", "A004_first")
            labelled_scan(root, "valid/valid", "A005_first")

            records = discover_labelled(root)
            self.assertEqual(len(records), 6)
            summary = prepare_manifests(root, root / "manifests", n_splits=5, seed=0)
            all_paths = read_manifest(root / "manifests" / "all_labelled.json")
            self.assertEqual(len(all_paths), 6)
            self.assertTrue(all(Path(path).is_file() for path in all_paths))
            owner = []
            for fold in summary["folds"]:
                names = {records[index]["name"] for index in fold["valid_indices"]}
                if any(name.startswith("A002_") for name in names):
                    owner.append((fold["fold"], names))
            self.assertEqual(len(owner), 1)
            self.assertEqual(sum(name.startswith("A002_") for name in owner[0][1]), 2)

    def test_multiscale_view_keeps_native_field_of_view(self):
        # cs2_plus: the scan itself uses x4, while its model tile was trained
        # at x2. Every TTA multiplier changes both quantities together.
        views = inference_views(768, 2.0, 480, [(800, 4)], [0.875, 1.0, 1.125])
        self.assertEqual(views, [(1344, 3.5), (1536, 4.0), (1728, 4.5)])

    def test_oof_audit_finds_confident_long_range_confusion(self):
        with tempfile.TemporaryDirectory(prefix="komap_oof_audit_") as temporary:
            root = Path(temporary)
            image_path = labelled_scan(root, "train/train", "A001_first", label=1)
            probabilities = np.zeros((2, 4, 32, 40), np.float32)
            probabilities[:, 0] = 0.9
            probabilities[:, 1] = 0.1
            probabilities[:, 0, 10:18, 12:22] = 0.0
            probabilities[:, 1, 10:18, 12:22] = 0.05
            probabilities[:, 2, 10:18, 12:22] = 0.95
            rows = find_regions(probabilities, image_path, [(1, 2, 5.0)],
                                confidence=0.75, margin=0.15,
                                agreement=1.0, min_area=16)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["area"], 80)
            self.assertEqual(rows[0]["true_name"], "Al3Ni")
            self.assertEqual(rows[0]["predicted_name"], "EutecticSi")

            saved = root / "A001_first_prob.npz"
            np.savez_compressed(saved, prob=probabilities[0].astype(np.float16),
                                image=str(image_path))
            stack, source = read_oof([saved])
            self.assertEqual(stack.shape, (1, 4, 32, 40))
            self.assertEqual(source, image_path)


if __name__ == "__main__":
    unittest.main()
