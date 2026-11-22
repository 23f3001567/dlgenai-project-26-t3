"""RLE helpers, Dice and the official SurfaceGuard AI competition score.

Pure NumPy, no GPU needed. Run `python scripts/rle_metric.py` to execute the self-tests.
Images are 128 (high) x 800 (wide); RLE is column-major and 1-based ("start length ...").
"""
import numpy as np

H, W = 128, 800


def rle_to_mask(rle, shape=(H, W)):
    """'start length start length ...' -> 0/1 array of `shape` (empty string -> all zeros)."""
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if isinstance(rle, str) and rle.strip():
        nums = np.array(rle.split(), dtype=np.int64)
        for start, length in zip(nums[0::2] - 1, nums[1::2]):   # "- 1": RLE counts from 1
            flat[start:start + length] = 1
    return flat.reshape(shape, order="F")                        # "F" = column-major


def mask_to_rle(mask):
    """0/1 array -> 'start length ...' ('' if the mask is empty)."""
    pixels = np.concatenate([[0], np.asarray(mask).flatten(order="F"), [0]])  # padding handles runs touching the border
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1                         # positions where the value changes
    runs[1::2] -= runs[::2]                                                    # end positions -> lengths
    return " ".join(str(x) for x in runs)


def dice(pred, true):
    """Dice = 2|X and Y| / (|X| + |Y|) for two 0/1 arrays; two empty masks count as 1."""
    pred, true = np.asarray(pred, bool), np.asarray(true, bool)
    total = pred.sum() + true.sum()
    return 1.0 if total == 0 else 2.0 * (pred & true).sum() / total


def postprocess(prob, thr=0.4, min_area=40):
    """probability map (H, W) -> 0/1 mask: threshold, then drop masks smaller than min_area pixels."""
    mask = prob > thr
    return mask if mask.sum() >= min_area else np.zeros_like(mask)


def competition_score(inter, pred_area, true_area):
    """Official score = 0.7 * foreground Dice + 0.2 * false-positive score + 0.1 * empty accuracy.

    inter / pred_area / true_area: arrays (N pairs, ...) of overlap, predicted and true pixel counts.
    """
    inter, pa, ga = [np.asarray(a, dtype=np.float64) for a in (inter, pred_area, true_area)]
    has_defect = ga > 0
    d = 2 * inter / np.maximum(pa + ga, 1)
    fg_dice = d[has_defect].mean() if has_defect.any() else 0.0
    if (~has_defect).any():
        fp_score = 1 - (pa[~has_defect] / (H * W)).mean()       # penalise false pixels in proportion to their size
        empty_acc = (pa[~has_defect] == 0).mean()               # share of empty pairs predicted exactly empty
    else:
        fp_score = empty_acc = 1.0
    return dict(score=0.7 * fg_dice + 0.2 * fp_score + 0.1 * empty_acc, fg_dice=fg_dice, fp_score=fp_score, empty_acc=empty_acc)


def _self_test():
    rng = np.random.RandomState(0)
    # 1) RLE round trip on random blob masks, including a run that touches the last pixel
    for _ in range(50):
        m = (rng.rand(H, W) > 0.97).astype(np.uint8)
        m[-1, -1] = 1
        assert (rle_to_mask(mask_to_rle(m)) == m).all()
    assert mask_to_rle(np.zeros((H, W), np.uint8)) == "" and rle_to_mask("").sum() == 0
    assert rle_to_mask("10 5").sum() == 5 and rle_to_mask("10 5").flatten(order="F")[9] == 1   # 1-based start
    # 2) Dice
    a = np.array([1, 1, 0, 0]); b = np.array([1, 0, 1, 0])
    assert abs(dice(a, b) - 0.5) < 1e-9 and dice(a * 0, b * 0) == 1.0
    # 3) post-processing
    p = np.zeros((H, W)); p[:5, :5] = 0.9
    assert postprocess(p, 0.4, 40).sum() == 0 and postprocess(p, 0.4, 20).sum() == 25
    # 4) the official worked example: 10 positive pairs (6 x Dice 0.8, 4 x Dice 0.4) + 40 empty pairs (4 false blobs of 1024 px)
    ga = np.r_[np.full(10, 1000), np.zeros(40)]
    pa = np.r_[np.full(10, 1000), np.full(4, 1024), np.zeros(36)]
    inter = np.r_[np.full(6, 800), np.full(4, 400), np.zeros(40)]
    r = competition_score(inter, pa, ga)
    assert abs(r["score"] - 0.7378) < 5e-4, r
    print("all self-tests passed; worked example score =", round(r["score"], 4))


if __name__ == "__main__":
    _self_test()
