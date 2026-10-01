"""Where the errors are: boundary placement versus whole-structure mislabelling.

"Within 2 px of a GT boundary" over-counts boundary errors on this data:
Eutectic Si is ~3 px thick, so a lamella labelled as the wrong phase lies
entirely inside that band. Measuring each error pixel's distance to the
nearest GT pixel *of the class it was predicted as* separates the two:
<=1 px means the boundary between two touching phases moved; >5 px means the
model saw a phase that is not there at all.
"""
import cv2
import numpy as np

from komap import NUM_CLASSES, per_image_miou

BUCKETS = (("<=1px", 0.0, 1.5), ("2-5px", 1.5, 5.0), (">5px", 5.0, np.inf))


def error_breakdown(pred, gt, num_classes=NUM_CLASSES):
    """Error pixels per distance bucket, and the image's mIoU if that bucket were fixed.

    Returns a dict: bucket -> (error pixel count, mIoU with those pixels set to GT).
    """
    dist = np.zeros(gt.shape, np.float32)
    for c in range(num_classes):
        m = pred == c
        if m.any():
            d = cv2.distanceTransform((gt != c).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            dist[m] = d[m]
    wrong = pred != gt
    out = {}
    for name, lo, hi in BUCKETS:
        sel = wrong & (dist > lo) & (dist <= hi)
        fixed = pred.copy()
        fixed[sel] = gt[sel]
        out[name] = (int(sel.sum()), per_image_miou(fixed, gt, num_classes)[0])
    return out


def summarise(rows, base_scores):
    """Print the bucket shares and headroom over a list of error_breakdown() dicts."""
    total = sum(n for r in rows for n, _ in r.values())
    base = float(np.mean(base_scores))
    print("  error breakdown (distance to the nearest GT pixel of the predicted class):")
    for name, _, _ in BUCKETS:
        n = sum(r[name][0] for r in rows)
        fixed = float(np.mean([r[name][1] for r in rows]))
        print(f"    {name:6s} {100 * n / max(total, 1):5.1f}% of error pixels   "
              f"fixing all of them: {100 * (fixed - base):+5.2f}pp")
