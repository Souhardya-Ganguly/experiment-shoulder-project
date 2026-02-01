import os
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Paths (edit only if needed)
# -----------------------------
DATA_DIR = Path("Preprocessed Data")
OUT_DIR = Path("outputs")

X_PATH = DATA_DIR / "data_test_b_full.npy"
Y_PATH = DATA_DIR / "data_mask_test_b_full.npy"
P_PATH = OUT_DIR / "pred_test_b.npy"   # probabilities from the training script

THR = 0.5  # threshold for binary prediction


# -----------------------------
# Metrics
# -----------------------------
def dice_iou_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7):
    """
    pred, gt: boolean arrays (H,W)
    """
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    p_sum = pred.sum()
    g_sum = gt.sum()
    dice = (2.0 * inter + eps) / (p_sum + g_sum + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou)


def mask_area(mask_bool: np.ndarray):
    return int(mask_bool.sum())


# -----------------------------
# Visualization helpers
# -----------------------------
def save_overlay_grid(
    X, Y, P, indices, out_path, title,
    thr=0.5, ncols=5
):
    """
    Saves a grid where each sample shows:
      - base grayscale image
      - GT contour-ish overlay (GT mask alpha)
      - Pred mask overlay (alpha)
    """
    n = len(indices)
    nrows = math.ceil(n / ncols)

    plt.figure(figsize=(ncols * 3.2, nrows * 3.2))
    for k, idx in enumerate(indices):
        ax = plt.subplot(nrows, ncols, k + 1)
        img = X[idx]
        gt = Y[idx].astype(bool)
        pr = (P[idx] >= thr)

        ax.imshow(img, cmap="gray")
        # overlay GT and prediction with different alpha layers
        ax.imshow(gt, alpha=0.25)
        ax.imshow(pr, alpha=0.25)

        d, j = dice_iou_binary(pr, gt)
        ax.set_title(f"idx={idx}\nD={d:.3f} IoU={j:.3f}")
        ax.axis("off")

    plt.suptitle(title, y=0.99)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_area_histogram(gt_areas, pr_areas, out_path):
    plt.figure(figsize=(8, 5))
    bins = 50
    plt.hist(gt_areas, bins=bins, alpha=0.6, label="GT area (pixels)")
    plt.hist(pr_areas, bins=bins, alpha=0.6, label="Pred area (pixels)")
    plt.xlabel("mask area (pixels)")
    plt.ylabel("count")
    plt.title("Mask area distribution (GT vs Pred) on test/b")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    # Basic existence checks
    for p in [X_PATH, Y_PATH, P_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load arrays
    X = np.load(X_PATH)  # (N,256,256) float in [0,1]
    Y = np.load(Y_PATH)  # (N,256,256) bool
    P = np.load(P_PATH)  # (N,256,256) float prob in [0,1]

    if X.shape != Y.shape or X.shape != P.shape:
        raise ValueError(f"Shape mismatch: X={X.shape}, Y={Y.shape}, P={P.shape}")

    N = X.shape[0]
    print(f"[Info] Loaded test/b: N={N}, shape={X.shape[1:]}")

    # Compute per-sample metrics + areas
    dice_list = np.zeros(N, dtype=np.float32)
    iou_list = np.zeros(N, dtype=np.float32)
    gt_area = np.zeros(N, dtype=np.int64)
    pr_area = np.zeros(N, dtype=np.int64)

    for i in range(N):
        gt = Y[i].astype(bool)
        pr = (P[i] >= THR)

        d, j = dice_iou_binary(pr, gt)
        dice_list[i] = d
        iou_list[i] = j
        gt_area[i] = mask_area(gt)
        pr_area[i] = mask_area(pr)

    # Summary
    print(f"[Summary test/b @thr={THR}]")
    print(f"  Dice: mean={dice_list.mean():.4f}, median={np.median(dice_list):.4f}, min={dice_list.min():.4f}, max={dice_list.max():.4f}")
    print(f"  IoU : mean={iou_list.mean():.4f}, median={np.median(iou_list):.4f}, min={iou_list.min():.4f}, max={iou_list.max():.4f}")

    # Save CSV of per-sample metrics
    csv_path = OUT_DIR / "qa_testb_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "dice", "iou", "gt_area_px", "pred_area_px"])
        for i in range(N):
            w.writerow([i, float(dice_list[i]), float(iou_list[i]), int(gt_area[i]), int(pr_area[i])])
    print(f"[Saved] {csv_path}")

    # Best/Worst 10 by Dice
    order = np.argsort(dice_list)  # ascending
    worst10 = order[:10].tolist()
    best10 = order[-10:][::-1].tolist()

    # Random-ish overlay set: pick 20 evenly spaced indices
    # (avoids needing RNG; deterministic)
    k = 20
    overlay_idx = np.linspace(0, N - 1, num=min(k, N), dtype=int).tolist()

    # Save visuals
    save_area_histogram(gt_area, pr_area, OUT_DIR / "qa_testb_mask_area_hist.png")
    print(f"[Saved] {OUT_DIR / 'qa_testb_mask_area_hist.png'}")

    save_overlay_grid(
        X, Y, P, overlay_idx,
        OUT_DIR / "qa_testb_overlays.png",
        title="Test/b overlays (evenly spaced samples) | GT + Pred (thr=0.5)",
        thr=THR,
        ncols=5,
    )
    print(f"[Saved] {OUT_DIR / 'qa_testb_overlays.png'}")

    save_overlay_grid(
        X, Y, P, best10,
        OUT_DIR / "qa_testb_best10.png",
        title="Best-10 test/b cases by Dice | GT + Pred (thr=0.5)",
        thr=THR,
        ncols=5,
    )
    print(f"[Saved] {OUT_DIR / 'qa_testb_best10.png'}")

    save_overlay_grid(
        X, Y, P, worst10,
        OUT_DIR / "qa_testb_worst10.png",
        title="Worst-10 test/b cases by Dice | GT + Pred (thr=0.5)",
        thr=THR,
        ncols=5,
    )
    print(f"[Saved] {OUT_DIR / 'qa_testb_worst10.png'}")

    print("[Done] QA report generated.")


if __name__ == "__main__":
    main()
