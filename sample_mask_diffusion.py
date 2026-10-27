"""Generate image-mask pairs and a matched real-patch control from TRAIN only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from komap import index_to_rgb
from mask_diffusion import MaskUNet, mask_condition, sample
from mask_diffusion_data import crop, read_train, remodel
from train_mask_diffusion import MODEL_KEYS


def prepare_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output: {path}")
    for sub in ("images", "masks"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path


def save_pair(directory, name, image, mask):
    if not cv2.imwrite(str(directory / "images" / name), image):
        raise OSError(f"failed to save {name}")
    path = directory / "masks" / name.replace("_image", "_mask")
    if not cv2.imwrite(str(path), cv2.cvtColor(index_to_rgb(mask), cv2.COLOR_RGB2BGR)):
        raise OSError(f"failed to save {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--control_out", required=True)
    ap.add_argument("--data_root", default=str(ROOT))
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--mode", choices=["appearance", "remodel"], default="appearance")
    ap.add_argument("--style_mode", choices=["source", "donor"], default="donor")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = ap.parse_args()
    if min(args.count, args.batch_size) < 1 or args.steps < 2:
        ap.error("count/batch must be positive and steps >= 2")
    if Path(args.out).resolve() == Path(args.control_out).resolve():
        ap.error("synthetic and real-control outputs must differ")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    recipe = checkpoint["args"]
    model = MaskUNet(**{k: recipe[k] for k in MODEL_KEYS}).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    records = read_train(args.data_root)
    expected = {(r["name"], r["sha256"]) for r in checkpoint["sources"]}
    if expected != {(r["name"], r["sha256"]) for r in records}:
        raise ValueError("training sources changed since generator training")
    # Selection-run smoke samples may use only the generator's actual train subset.
    allowed = set(checkpoint["split"]["train"])
    records = [r for r in records if r["name"] in allowed]
    out, control = prepare_output(args.out), prepare_output(args.control_out)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device).manual_seed(args.seed)
    rows, controls, tone_errors, previews = [], [], [], []
    for first in range(0, args.count, args.batch_size):
        pending = []
        for i in range(first, min(args.count, first + args.batch_size)):
            source = records[int(rng.integers(len(records)))]
            real, original_mask, location = crop(source, recipe["patch"], rng)
            mask, remodeled = (remodel(original_mask, rng) if args.mode == "remodel"
                                else (original_mask, False))
            present = np.unique(mask)
            candidates = [r for r in records
                          if (r["mask"].shape[0] <= 800) == (source["mask"].shape[0] <= 800)
                          and np.all(r["style"][8 + present] > 0) and r["name"] != source["name"]]
            donor = (candidates[int(rng.integers(len(candidates)))]
                     if args.style_mode == "donor" and candidates else source)
            meta = dict(name=f"M{i:05d}_image.png", source=source["name"],
                        source_sha256=source["sha256"], source_height=source["mask"].shape[0],
                        crop=location, style_source=donor["name"],
                        style=donor["style"].tolist(), remodeled=remodeled,
                        mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
            pending.append((real, original_mask, mask, donor["style"], meta))
        conditions = torch.from_numpy(np.stack([mask_condition(p[2]) for p in pending])).to(device)
        styles = torch.from_numpy(np.stack([p[3] for p in pending])).to(device)
        generated = sample(model, conditions, styles, steps=args.steps, generator=generator).cpu().numpy()
        for image, (real, original_mask, mask, style, meta) in zip(generated, pending):
            save_pair(out, meta["name"], image, mask)
            # This control repeats exactly the same source crop, including D4.
            # For remodel, it keeps the ORIGINAL matching mask, never a false label.
            save_pair(control, meta["name"], real, original_mask)
            control_meta = dict(meta, style_source=meta["source"], style=None, remodeled=False,
                                mask_sha256=hashlib.sha256(original_mask.tobytes()).hexdigest())
            rows.append(meta)
            controls.append(control_meta)
            errors = [float(image[mask == c].mean() - (style[c] + 1) * 127.5)
                      if np.any(mask == c) else None for c in range(4)]
            tone_errors.append(errors)
            if len(previews) < 4:
                previews.append(np.concatenate([np.repeat(real[..., None], 3, 2), index_to_rgb(mask),
                                                np.repeat(image[..., None], 3, 2)], 0))
        print(f"sampled {len(rows)}/{args.count}", flush=True)
    common = dict(args=vars(args), generator_step=checkpoint["step"], generator_mode=recipe["mode"],
                  ckpt_sha256=hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest(),
                  source_split="train/train", size=recipe["patch"], count=len(rows))
    for directory, items, kind in ((out, rows, "mask_conditional"), (control, controls, "real_repeat")):
        with open(directory / "manifest.jsonl", "w") as f:
            for row in items:
                f.write(json.dumps(row) + "\n")
        (directory / "meta.json").write_text(json.dumps(dict(**common, kind=kind), indent=2))
    diagnostic = dict(remodeled_count=sum(r["remodeled"] for r in rows),
                      mean_phase_gray_error_to_requested_style=[
                          float(np.mean([e[c] for e in tone_errors if e[c] is not None]))
                          if any(e[c] is not None for e in tone_errors) else None for c in range(4)],
                      note="diagnostic only; condition masks are targets, not proof of image alignment")
    (out / "diagnostics.json").write_text(json.dumps(diagnostic, indent=2))
    cv2.imwrite(str(out / "preview.png"), cv2.cvtColor(np.concatenate(previews, 1), cv2.COLOR_RGB2BGR))
    print(json.dumps(diagnostic), flush=True)


if __name__ == "__main__":
    main()
