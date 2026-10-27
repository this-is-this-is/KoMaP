"""One augmentation at a time, on the team protocol: B3, batch 4, 12000 steps, seed 42.

Every row of the team's augmentation table is trained on its own on top of
the same bare recipe -- plain random 512 crops and nothing else -- so its score
against that reference is that augmentation's effect alone. Two rows have a
different reference: "Crop" is compared with fixed grid tiles (no crop
augmentation at all), and "Small-scan Scale x4" with "Input Scale x2", which
it builds on.

    ./.conda/bin/python aug_table.py names             run names, in training order
    ./.conda/bin/python aug_table.py flags hflip       train_v2.py flags of one run
    ./.conda/bin/python aug_table.py summary           results table -> AUG_TABLE_RESULTS.md
    (run_aug_table.sh trains every row and then prints the summary)

Runs are runs/t_<name>_s<seed>.
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
# The model of every other experiment here: MicroNet EfficientNet-B3 U-Net++
# with the full-resolution branch. 12000 steps of batch 4 are 85.7 epochs of
# 140 steps; validation every 4 epochs (560 steps) and at the end.
COMMON = ("--arch unetpp --encoder efficientnet-b3 --pretrain micronet --fullres_skip "
          "--batch_size 4 --max_steps 12000 --val_every 4")
BARE = "--class_ratio 0 --geo_aug none --color_aug none"

# name, row of the team table, what is switched on, flags on top of BARE, reference run
TABLE = [
    ("grid", "(기준) 증강 없음", "고정 격자 타일 (매 에폭 같은 타일, 무작위 crop 없음)", "--crop_mode grid", None),
    ("hflip", "Horizontal Flip", "p=0.5", "--geo_aug hflip", "crop"),
    ("vflip", "Vertical Flip", "p=0.5", "--geo_aug vflip", "crop"),
    ("rot90", "90° Rotation", "0/90/180/270도 균등", "--geo_aug rot90", "crop"),
    ("d4", "D4 Augmentation", "8방향 균등", "--geo_aug d4", "crop"),
    ("brightness", "Brightness", "±15%, p=0.5", "--color_aug brightness", "crop"),
    ("contrast", "Contrast", "±15%, p=0.5", "--color_aug contrast", "crop"),
    ("gamma", "Gamma", "0.9–1.1, p=0.5", "--color_aug gamma", "crop"),
    ("noise", "Gaussian Noise", "표준편차 2–5%, p=0.5", "--color_aug noise", "crop"),
    ("rscale", "Random Scale", "0.7–1.4배", "--scale_aug 0.7 1.4", "crop"),
    ("crop", "Crop", "무작위 위치 512 crop (다른 행의 바탕)", "", "grid"),
    ("classcrop", "Class-balanced Crop", "crop의 50%를 희소 클래스 픽셀 중심으로", "--class_ratio 0.5", "crop"),
    ("centercrop", "Microstructure-centered Crop", "crop의 50%를 Eutectic Si 픽셀 중심으로", "--focus_ratio 0.5", "crop"),
    ("morphcrop", "Morphology-aware Crop", "crop의 30%를 얇고 조밀한 Eutectic Si 위치로", "--thin_ratio 0.3", "crop"),
    ("cutmix", "CutMix", "p=0.2", "--cutmix_p 0.2", "crop"),
    ("copypaste", "Copy-Paste", "Si가 많은 상자를 다른 crop에서 복사, p=0.1", "--context_paste_p 0.1", "crop"),
    ("prisi", "Primary-Si Particle Shift", "같은 스캔 안 입자 이동, p=0.25", "--prisi_paste_p 0.25", "crop"),
    ("scale2", "Input Scale ×2", "학습·추론 모두 2배", "--input_scale 2", "crop"),
    ("small4", "Small-scan Scale ×4", "2배 위에 작은 스캔만 4배", "--input_scale 2 --scale_by_height 800:4", "scale2"),
    ("mildcolor", "Mild Color Augmentation", "밝기·대비·감마 중 하나 ±10%, p=0.5", "--color_aug oneof", "crop"),
    ("illum", "Illumination Field", "±5%, p=0.2", "--illumination_p 0.2", "crop"),
    ("fourier", "Fourier Mixing", "저주파 진폭 10–20% 혼합, p=0.1", "--fourier_p 0.1", "crop"),
    # Not part of the suite's first pass: run_diffusion.sh trains it once synth/d1 exists.
    ("diffusion", "Diffusion 합성 데이터", "디퓨전으로 만든 이미지·마스크 쌍을 crop의 30%로",
     "--synth_dir synth/d1 --synth_ratio 0.3", "crop"),
]
ROW = {r[0]: r for r in TABLE}
# References first, then the two long scale-2 runs so they overlap with short ones.
ORDER = ["grid", "crop", "scale2", "small4"]
ORDER += [r[0] for r in TABLE if r[0] not in ORDER]
CLASSES = ["Al", "Al3Ni", "EutecticSi", "PrimarySi"]


def flags(name):
    return " ".join(f"{COMMON} {BARE} {ROW[name][3]}".split())


def final_eval(run_dir, device, data):
    """mIoU of best.pth under the final inference setting; cached next to the checkpoint."""
    import numpy as np
    import torch

    from komap import NUM_CLASSES, per_image_miou
    from postprocess import si_particle_vote
    from predict import load_model
    from preprocess import parse_scale_table, prepare, scan_scale
    from tiling import predict_tiled, scaled_tile

    ckpt, cache = f"{run_dir}/best.pth", f"{run_dir}/final_eval.json"
    if os.path.exists(cache):
        with open(cache) as f:
            out = json.load(f)
        if out.get("ckpt_mtime") == os.path.getmtime(ckpt):
            return out
    model, a = load_model(ckpt, device)
    s = a.get("input_scale", 1.0)
    table = parse_scale_table(a.get("scale_by_height"))
    scores, ious = [], []
    for img, gt in data:
        prob = predict_tiled(model, prepare(img, a).to(device), tile=scaled_tile(768, s), overlap=0.5,
                             num_classes=NUM_CLASSES, flip_tta=True, transpose_tta=True,
                             scale=scan_scale(img.shape[0], s, table))
        miou, iou = per_image_miou(si_particle_vote(prob), gt)
        scores.append(miou)
        ious.append(iou)
    del model
    torch.cuda.empty_cache()
    out = {"ckpt_mtime": os.path.getmtime(ckpt), "miou": 100 * float(np.mean(scores)),
           "class_iou": (100 * np.nanmean(ious, 0)).tolist()}
    with open(cache, "w") as f:
        json.dump(out, f)
    return out


def summary(seeds, final, runs_dir="runs", out="AUG_TABLE_RESULTS.md"):
    import numpy as np

    runs_dir, out = os.path.abspath(runs_dir), os.path.abspath(out)
    os.chdir(ROOT)
    sys.path.insert(0, os.path.join(ROOT, "src"))
    data = device = None
    if final:
        import cv2
        import torch

        from komap import rgb_to_index
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = []
        for p in sorted(glob.glob("valid/valid/images/*.png")):
            mp = p.replace("/images/", "/masks/").replace("_image.png", "_mask.png")
            data.append((cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB),
                         rgb_to_index(cv2.cvtColor(cv2.imread(mp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))))

    stats = {}
    for name in ROW:
        per_seed = []
        for s in seeds:
            run = f"{runs_dir}/t_{name}_s{s}"
            if not os.path.exists(f"{run}/history.json"):
                continue
            with open(f"{run}/history.json") as f:
                h = json.load(f)
            row = {"best": 100 * h["best_miou"], "last": 100 * h["history"][-1]["miou"]}
            if final:
                e = final_eval(run, device, data)
                row["final"], row["cls"] = e["miou"], e["class_iou"]
                print(f"evaluated {os.path.basename(run)}", flush=True)
            else:
                with open(f"{run}/val_per_image.json") as f:
                    rows = json.load(f)["per_image"]
                row["cls"] = [100 * np.nanmean([np.nan if r[c] is None else r[c] for r in rows])
                              for c in CLASSES]
            per_seed.append(row)
        if per_seed:
            stats[name] = {k: np.mean([r[k] for r in per_seed], 0) for k in per_seed[0]}
            stats[name]["n"] = len(per_seed)

    metrics = [("best", "학습 중 best"), ("last", "마지막 에폭")] + ([("final", "최종 추론")] if final else [])

    def cell(name, key):
        v, ref = stats[name][key], ROW[name][4]
        if ref is None or ref not in stats:
            return f"{v:.2f}", "–"
        return f"{v:.2f}", f"{v - stats[ref][key]:+.2f}"

    lines = ["# 증강별 비교 실험 결과", "",
             f"- 조건: MicroNet EfficientNet-B3 U-Net++ + 원본 해상도 branch, 배치 4, 12000 step, seed {', '.join(map(str, seeds))}",
             "- 각 행은 무작위 crop만 쓰는 바탕(`crop`)에 그 증강 하나만 더한 학습입니다. `차이`는 비교 기준 대비 pp입니다.",
             "- 학습 중 best와 마지막 에폭은 valid 20장, tile 512, TTA 없음. "
             + ("최종 추론은 best checkpoint에 8방향 TTA + tile 768 + Si 입자 투표. 클래스별 IoU는 최종 추론 기준입니다."
                if final else "클래스별 IoU는 best checkpoint 기준입니다."),
             "- seed가 1개면 ±0.3pp 안의 차이는 seed 노이즈와 구분되지 않습니다.", "",
             "| 증강 | 설정 | 비교 기준 | " + " | ".join(f"{label} | 차이" for _, label in metrics)
             + " | Al | Al3Ni | Eutectic Si | Primary Si |",
             "|---|---|---|" + "---:|---:|" * len(metrics) + "---:|---:|---:|---:|"]
    for name, label, setting, _, ref in TABLE:
        if name not in stats:
            lines.append(f"| {label} | {setting} | {ROW[ref][1] if ref else '–'} | " + "미완료 | – | " * len(metrics) + "– | – | – | – |")
            continue
        cells = [x for key, _ in metrics for x in cell(name, key)]
        lines.append(f"| {label} | {setting} | {ROW[ref][1] if ref else '–'} | " + " | ".join(cells) + " | "
                     + " | ".join(f"{v:.1f}" for v in stats[name]["cls"]) + " |")
    key = "final" if final else "best"
    ranked = sorted(((stats[n][key] - stats[ROW[n][4]][key], ROW[n][1]) for n in stats
                     if ROW[n][4] in stats), reverse=True)
    if ranked:
        lines += ["", f"{dict(metrics)[key]} 기준 차이 순위: "
                  + ", ".join(f"{label} {d:+.2f}" for d, label in ranked)]
    text = "\n".join(lines) + "\n"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text + f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["names", "flags", "summary"])
    ap.add_argument("name", nargs="?", help="run name, for flags")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--no_final", action="store_true",
                    help="skip the final-inference evaluation (no GPU needed)")
    ap.add_argument("--runs_dir", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "AUG_TABLE_RESULTS.md"))
    args = ap.parse_args()
    if args.command == "names":
        print(" ".join(ORDER))
    elif args.command == "flags":
        if args.name not in ROW:
            raise SystemExit(f"unknown run {args.name}; one of: {' '.join(ORDER)}")
        print(flags(args.name))
    else:
        summary(args.seeds, not args.no_final, args.runs_dir, args.out)


if __name__ == "__main__":
    main()
