# explain/viz.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return x


def save_overlay(
    out_path: Path,
    image: np.ndarray,
    pred_mask: np.ndarray,
    heatmap: np.ndarray,
    title: str = "",
):
    """
    image: (H,W) in [0,1]
    pred_mask: (H,W) {0,1}
    heatmap: (H,W) float
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize heatmap for visualization
    hm = heatmap.astype(np.float32)
    hm = hm - hm.min()
    if hm.max() > 0:
        hm = hm / hm.max()

    fig = plt.figure(figsize=(12, 4))

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(image, cmap="gray")
    ax1.set_title("Input")
    ax1.axis("off")

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(image, cmap="gray")
    ax2.imshow(pred_mask, alpha=0.35)
    ax2.set_title("Pred mask overlay")
    ax2.axis("off")

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(image, cmap="gray")
    ax3.imshow(hm, alpha=0.55)
    ax3.set_title("Explanation heatmap")
    ax3.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def save_counterfactual_panel(
    out_path: Path,
    image: np.ndarray,
    pred_mask: np.ndarray,
    x_cf: np.ndarray,
    p0: np.ndarray,
    p_cf: np.ndarray,
    delta: np.ndarray,
    title: str = "",
):
    """
    image: (H,W)
    pred_mask: (H,W)
    x_cf: (H,W)
    p0, p_cf: (H,W) probabilities
    delta: (H,W)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # normalize delta for display
    d = delta.astype(np.float32)
    d = d - d.min()
    if d.max() > 0:
        d = d / d.max()

    fig = plt.figure(figsize=(14, 7))

    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(image, cmap="gray")
    ax1.set_title("Original x")
    ax1.axis("off")

    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(image, cmap="gray")
    ax2.imshow(pred_mask, alpha=0.35)
    ax2.set_title("Original predicted region")
    ax2.axis("off")

    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(p0, cmap="gray")
    ax3.set_title("p0 = sigmoid(logits(x))")
    ax3.axis("off")

    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(x_cf, cmap="gray")
    ax4.set_title("Counterfactual x_cf")
    ax4.axis("off")

    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(p_cf, cmap="gray")
    ax5.set_title("p_cf = sigmoid(logits(x_cf))")
    ax5.axis("off")

    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(image, cmap="gray")
    ax6.imshow(d, alpha=0.55)
    ax6.set_title("Delta heatmap (scaled)")
    ax6.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
