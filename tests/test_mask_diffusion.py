import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from komap import index_to_rgb
from mask_diffusion import MaskUNet, loss, mask_condition, phase_style, sample
from mask_diffusion_data import remodel, split_records
from summarize_mask_diffusion import paired_summary


class MaskDiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.mask = np.zeros((32, 32), np.uint8)
        self.mask[:, 7:10] = 2
        self.mask[4:9, 14:25] = 1
        self.mask[19:28, 21:29] = 3
        self.image = np.array([210, 145, 95, 60], np.uint8)[self.mask]

    def test_condition_retains_thin_phase_and_absent_class(self):
        mask = self.mask.copy()
        mask[mask == 3] = 0
        condition = mask_condition(mask)
        np.testing.assert_array_equal(condition[:4].argmax(0), mask)
        self.assertTrue((condition[6][mask == 2] > 0).all())
        self.assertTrue((condition[7] == -1).all())
        self.assertTrue(np.isfinite(condition).all())

    def test_group_split_keeps_repeated_prefixes_together(self):
        records = [dict(name=f"A{a}_step{s}_image.png") for a in range(10) for s in (1, 2)]
        train, holdout = split_records(records)
        a = {r["name"].split("_")[0] for r in train}
        b = {r["name"].split("_")[0] for r in holdout}
        self.assertFalse(a & b)
        self.assertEqual(len(train) + len(holdout), len(records))

    def test_training_and_seeded_sampling_keep_condition_unchanged(self):
        condition = torch.from_numpy(mask_condition(self.mask))[None]
        original = condition.clone()
        style = torch.from_numpy(phase_style(self.image, self.mask))[None]
        model = MaskUNet(base=32)
        image = torch.from_numpy(self.image.astype(np.float32)[None, None] / 127.5 - 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad()
            value = loss(model, image, condition, style)
            value.backward()
            optimizer.step()
        self.assertTrue(torch.isfinite(value))
        self.assertGreater(float(model.conditions[0].weight.grad.abs().sum()), 0)
        model.eval()
        a = sample(model, condition, style, steps=3, generator=torch.Generator().manual_seed(7))
        b = sample(model, condition, style, steps=3, generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(condition, original))

    def test_remodel_is_bounded_and_never_mutates_input(self):
        original = self.mask.copy()
        for seed in range(10):
            result, _ = remodel(self.mask, np.random.default_rng(seed))
            np.testing.assert_array_equal(self.mask, original)
            self.assertTrue(np.isin(result, np.arange(4)).all())
            for c in range(4):
                before, after = (original == c).astype(np.uint8), (result == c).astype(np.uint8)
                self.assertEqual(cv2.connectedComponents(before)[0], cv2.connectedComponents(after)[0])

    def test_paired_summary_matches_images_not_iteration_order(self):
        reference = {"A1_a": 75.0, "A1_b": 78.0, "A2_a": 80.0}
        new = {"A2_a": 80.5, "A1_b": 78.5, "A1_a": 75.5}
        result = paired_summary(reference, new, repeats=100)
        self.assertEqual(result["delta_pp"], 0.5)
        self.assertEqual(result["image_bootstrap_95ci_pp"], [0.5, 0.5])
        self.assertEqual(result["filename_groups"], 2)
        with self.assertRaises(ValueError):
            paired_summary(reference, {"A1_a": 75})

    def test_cli_select_refit_generate_and_segmentation_loader(self):
        with tempfile.TemporaryDirectory(prefix="komap_mask_diff_") as temporary:
            directory = Path(temporary)
            for kind in ("images", "masks"):
                (directory / "train/train" / kind).mkdir(parents=True)
            for i in range(6):
                cv2.imwrite(str(directory / "train/train/images" / f"A{i}_image.png"), self.image)
                cv2.imwrite(str(directory / "train/train/masks" / f"A{i}_mask.png"),
                            cv2.cvtColor(index_to_rgb(self.mask), cv2.COLOR_RGB2BGR))

            def run(script, *flags):
                result = subprocess.run([sys.executable, str(ROOT / script), *map(str, flags)],
                                        cwd=ROOT, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            select, refit = directory / "select", directory / "refit"
            run("train_mask_diffusion.py", "--data_root", directory, "--out", select,
                "--patch", 32, "--base", 32, "--steps", 2, "--batch_size", 1,
                "--accum", 1, "--workers", 0, "--device", "cpu", "--eval_every", 1,
                "--preview_every", 0)
            run("train_mask_diffusion.py", "--data_root", directory, "--out", refit,
                "--mode", "refit", "--selection", select / "best.pt", "--device", "cpu",
                "--workers", 0, "--preview_every", 0)
            split = json.loads((refit / "split.json").read_text())
            self.assertEqual(len(split["train"]), 6)
            self.assertEqual(split["holdout"], [])
            generated, control = directory / "generated", directory / "control"
            run("sample_mask_diffusion.py", "--ckpt", refit / "best.pt", "--out", generated,
                "--control_out", control, "--data_root", directory, "--count", 2,
                "--batch_size", 2, "--steps", 3, "--device", "cpu")
            for path in (generated / "masks").glob("*.png"):
                np.testing.assert_array_equal(cv2.imread(str(path)), cv2.imread(str(control / "masks" / path.name)))
            # Patch dimensions must not select the small-scan scale for data
            # generated from a large source scan (the cs2_plus path).
            manifest = [json.loads(row) for row in (generated / "manifest.jsonl").read_text().splitlines()]
            for row in manifest:
                row["source_height"] = 1024
            (generated / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
            from train_v2 import CropDataset
            ds = CropDataset(str(directory / "train/train"), tile=16, samples_per_image=1,
                             input_scale=2, scale_table=[(800, 4)], synth_dir=str(generated), synth_ratio=1,
                             geo_aug="none", color_aug="none", class_ratio=0)
            self.assertEqual(ds.scales[-1], 2)
            self.assertEqual(ds[0]["image"].shape, (3, 32, 32))


if __name__ == "__main__":
    unittest.main()
