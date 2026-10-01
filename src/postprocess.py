"""Post-processing of full-image class probabilities."""
import cv2
import numpy as np


def si_particle_vote(prob, eutectic=2, primary=3):
    """Give every connected Si particle a single label.

    Eutectic and Primary Si are the same phase told apart by size and shape,
    and on this data 99% of the confusion between them is whole-particle
    mislabelling rather than boundary placement. So each 8-connected blob of
    predicted Si takes whichever of the two has the larger summed probability
    over the blob. Parameter-free; +0.3-0.4pp on valid together with 8-way TTA.

    Args:
        prob: (K, H, W) class probabilities (torch tensor or numpy array).
    Returns:
        (H, W) uint8 label map.
    """
    if hasattr(prob, "detach"):
        prob = prob.detach().float().cpu().numpy()
    pred = prob.argmax(0).astype(np.uint8)
    n, cc = cv2.connectedComponents(np.isin(pred, (eutectic, primary)).astype(np.uint8),
                                    connectivity=8)
    if n <= 1:
        return pred
    pe = np.bincount(cc.ravel(), weights=prob[eutectic].ravel(), minlength=n)
    pp = np.bincount(cc.ravel(), weights=prob[primary].ravel(), minlength=n)
    label = np.where(pp > pe, primary, eutectic).astype(np.uint8)
    blob = cc > 0
    pred[blob] = label[cc[blob]]
    return pred
