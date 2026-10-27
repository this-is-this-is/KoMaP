"""Paired scan-level uncertainty for fixed-checkpoint diffusion experiments.

Accept compare_groups.py output or two predict.py scores.json files. Reports
image bootstrap and A<number>-prefix cluster bootstrap; neither accounts for
repeated validation-based model selection or all training-seed uncertainty.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np


def paired_summary(reference, generated, repeats=10000, seed=17):
    if set(reference) != set(generated) or not reference:
        raise ValueError("paired evaluation requires the same nonempty set of images")
    names = sorted(reference)
    a, b = (np.array([values[n] for n in names], dtype=float) for values in (reference, generated))
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite image score")
    delta = b - a
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(len(names), size=(repeats, len(names)))].mean(1)
    grouped = {}
    for i, name in enumerate(names):
        match = re.match(r"(A\d+)_", name)
        grouped.setdefault(match.group(1) if match else name, []).append(i)
    clusters = [delta[idx] for idx in grouped.values()]
    picks = rng.integers(len(clusters), size=(repeats, len(clusters)))
    sums, counts = np.array([c.sum() for c in clusters]), np.array([len(c) for c in clusters])
    cluster_boot = sums[picks].sum(1) / counts[picks].sum(1)
    return dict(reference_miou=float(a.mean()), generated_miou=float(b.mean()),
                delta_pp=float(delta.mean()), images=len(names), filename_groups=len(clusters),
                better=int((delta > 1e-4).sum()), worse=int((delta < -1e-4).sum()),
                image_bootstrap_95ci_pp=np.percentile(boot, [2.5, 97.5]).tolist(),
                filename_cluster_bootstrap_95ci_pp=np.percentile(cluster_boot, [2.5, 97.5]).tolist(),
                per_image_delta_pp=dict(zip(names, delta.tolist())),
                caveat="CIs condition on fitted checkpoints; repeated validation selection and full seed "
                       "uncertainty are excluded. A<number> filename groups are not verified specimens.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparison", help="compare_groups JSON (scores already in percent)")
    ap.add_argument("--ref_group")
    ap.add_argument("--new_group")
    ap.add_argument("--ref_scores", help="predict.py scores.json (scores in [0,1])")
    ap.add_argument("--new_scores")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    extra = {}
    if args.comparison:
        if not args.ref_group or not args.new_group or args.ref_scores or args.new_scores:
            ap.error("--comparison needs two group names and cannot mix with --ref_scores/--new_scores")
        rows = json.loads(Path(args.comparison).read_text())

        def ensemble(group):
            matches = [r for r in rows if re.fullmatch(re.escape(group) + r" x\d+", r)]
            if len(matches) != 1:
                raise ValueError(f"expected one ensemble row for {group}")
            return rows[matches[0]]

        ref, new = ensemble(args.ref_group), ensemble(args.new_group)
        reference, generated = ref["per_image"], new["per_image"]
        extra["class_delta_pp"] = {k: new["class_iou"][k] - v for k, v in ref["class_iou"].items()}
        extra["error_px_delta"] = {k: new["error_px"][k] - v for k, v in ref["error_px"].items()}

        def singles(group):
            return {int(r.rsplit("_s", 1)[1]): v["miou"] for r, v in rows.items()
                    if re.fullmatch(re.escape(group) + r"_s\d+", r)}

        a, b = singles(args.ref_group), singles(args.new_group)
        if set(a) == set(b):
            extra["paired_seed_delta_pp"] = {str(s): b[s] - a[s] for s in sorted(a)}
        extra["sources"] = vars(args)
    else:
        if not args.ref_scores or not args.new_scores:
            ap.error("provide --comparison or both --ref_scores and --new_scores")
        a, b = (json.loads(Path(p).read_text()) for p in (args.ref_scores, args.new_scores))
        reference, generated = ({r["name"]: 100 * r["miou"] for r in data["per_image"]} for data in (a, b))
        classes = [k for k in a["per_image"][0] if k not in ("name", "miou")]
        means = [{c: np.mean([r[c] for r in data["per_image"] if r[c] is not None]) for c in classes}
                 for data in (a, b)]
        extra["class_delta_pp"] = {c: float(100 * (means[1][c] - means[0][c])) for c in classes}
        extra["sources"] = vars(args)
    result = dict(**paired_summary(reference, generated), **extra)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"mIoU {result['reference_miou']:.2f} -> {result['generated_miou']:.2f}; "
          f"delta {result['delta_pp']:+.2f}pp; image-bootstrap 95% CI "
          f"{result['image_bootstrap_95ci_pp']}; wrote {args.out}")


if __name__ == "__main__":
    main()
