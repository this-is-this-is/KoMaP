import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from external_pretraining import CrossMicroscopePairs, nasa_mask, supervised_contrastive_loss
from models import load_encoder_checkpoint, load_transfer_checkpoint
from prepare_external_microscopy import safe_unzip


class TinyTransferModel(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 1)
        self.decoder = nn.Conv2d(4, 4, 1)
        self.stem = nn.Conv2d(3, 2, 1)
        self.fuse = nn.Conv2d(6, 4, 1)
        self.segmentation_head = nn.Conv2d(4, classes, 1)


class ExternalPretrainingTests(unittest.TestCase):
    def test_supcon_rewards_same_roi_embeddings(self):
        features = torch.tensor([[1.0, 0], [1.0, 0], [0, 1.0], [0, 1.0]])
        aligned = supervised_contrastive_loss(features, torch.tensor([0, 0, 1, 1]))
        wrong = supervised_contrastive_loss(features, torch.tensor([0, 1, 0, 1]))
        self.assertTrue(torch.isfinite(aligned))
        self.assertLess(float(aligned), float(wrong))

    def test_cross_pair_prefers_same_magnification_different_microscope(self):
        with tempfile.TemporaryDirectory(prefix="komap_cross_pair_") as temporary:
            root = Path(temporary)
            rows = []
            variants = (("A", "50x"), ("B", "50x"), ("C", "20x"), ("C", "20x"))
            for index, (microscope, magnification) in enumerate(variants):
                path = root / f"image{index}.png"
                cv2.imwrite(str(path), np.full((40, 40, 3), 40 + index * 50, np.uint8))
                rows.append({"path": path.name, "roi_id": "sample|phase|1", "split": "train",
                             "microscope": microscope, "magnification": magnification})
            manifest = root / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            dataset = CrossMicroscopePairs(manifest, size=32, repeats=12, seed=3)
            for index in range(len(dataset)):
                sample = dataset[index]
                self.assertEqual(sample["magnification1"], sample["magnification2"])
                if sample["magnification1"] == "50x":
                    self.assertNotEqual(sample["microscope1"], sample["microscope2"])
                self.assertEqual(tuple(sample["view1"].shape), (3, 32, 32))

    def test_nasa_overlay_uses_official_exact_colours(self):
        with tempfile.TemporaryDirectory(prefix="komap_nasa_mask_") as temporary:
            path = Path(temporary) / "mask.tif"
            overlay = np.full((8, 8, 4), (127, 127, 127, 255), np.uint8)
            overlay[1:3, 1:3, :3] = (255, 0, 0)  # BGR blue: secondary
            overlay[5:7, 5:7, :3] = (0, 0, 255)  # BGR red: tertiary
            cv2.imwrite(str(path), overlay)
            result = nasa_mask(path)
            self.assertTrue(np.all(result[1:3, 1:3] == 1))
            self.assertTrue(np.all(result[5:7, 5:7] == 2))
            self.assertEqual(int(result[0, 0]), 0)

    def test_checkpoint_contracts_are_strict_and_reset_head(self):
        with tempfile.TemporaryDirectory(prefix="komap_transfer_") as temporary:
            root = Path(temporary)
            source_encoder = nn.Conv2d(3, 4, 1)
            with torch.no_grad():
                source_encoder.weight.fill_(0.25)
            encoder_path = root / "encoder.pth"
            torch.save({"format": "komap_encoder_v1", "encoder": source_encoder.state_dict()}, encoder_path)
            target_encoder = nn.Conv2d(3, 4, 1)
            load_encoder_checkpoint(target_encoder, encoder_path)
            self.assertTrue(torch.equal(source_encoder.weight, target_encoder.weight))

            source, target = TinyTransferModel(3), TinyTransferModel(4)
            before = target.segmentation_head.weight.detach().clone()
            transfer_path = root / "transfer.pth"
            torch.save({"format": "komap_segmentation_transfer_v1", "model": source.state_dict()},
                       transfer_path)
            load_transfer_checkpoint(target, transfer_path)
            self.assertTrue(torch.equal(source.encoder.weight, target.encoder.weight))
            self.assertTrue(torch.equal(source.decoder.weight, target.decoder.weight))
            self.assertTrue(torch.equal(before, target.segmentation_head.weight))

            bad = root / "bad.pth"
            state = source_encoder.state_dict()
            state.pop("bias")
            torch.save({"format": "komap_encoder_v1", "encoder": state}, bad)
            with self.assertRaises(RuntimeError):
                load_encoder_checkpoint(target_encoder, bad)

    def test_zip_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="komap_zip_") as temporary:
            root = Path(temporary)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as value:
                value.writestr("../escape.txt", "no")
            with self.assertRaises(ValueError):
                safe_unzip(archive, root / "out")


if __name__ == "__main__":
    unittest.main()
