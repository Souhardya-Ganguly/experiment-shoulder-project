# explain/counterfactuals.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any

import torch
import torch.nn.functional as F


def tv_loss(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Total variation loss for smoothness.
    x: (1,1,H,W)
    """
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]
    return torch.sqrt(dh * dh + eps).mean() + torch.sqrt(dw * dw + eps).mean()


@dataclass
class CFConfig:
    # Optimization
    steps: int = 200
    lr: float = 0.05

    # Regularization
    lam_l1: float = 0.02
    lam_tv: float = 0.10

    # Range constraints for the counterfactual image
    clamp_min: float = 0.0
    clamp_max: float = 1.0

    # NEW: ROI-gated edits + hard perturbation budget
    roi_kernel: int = 41        # dilation kernel size (odd). bigger -> more allowed area.
    max_abs_delta: float = 0.15 # hard cap on per-pixel change in [0,1] space

    # Reporting
    print_every: int = 25

    # For "changed fraction" reporting (auditable)
    changed_thr_small: float = 0.02
    changed_thr_large: float = 0.05


@torch.no_grad()
def predicted_region_from_logits(logits: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs >= thr)  # bool mask (1,1,H,W)


@torch.no_grad()
def dilate_mask(mask: torch.Tensor, k: int = 41) -> torch.Tensor:
    """
    Binary dilation via maxpool.
    mask: (1,1,H,W) bool
    Returns: (1,1,H,W) bool
    """
    if k <= 1:
        return mask
    if k % 2 == 0:
        k = k + 1  # enforce odd
    x = mask.float()
    pad = k // 2
    x = F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
    return (x > 0.0)


def counterfactual_reduce_region_confidence(
    forward_logits: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    region_mask: torch.Tensor,
    cfg: CFConfig = CFConfig(),
) -> Dict[str, Any]:
    """
    Optimize a perturbation delta to reduce mean predicted probability inside region_mask,
    while keeping edits localized (ROI-gated), small (L1 + hard cap), and smooth (TV).

    forward_logits(x)->logits (1,1,H,W)
    x: (1,1,H,W) float32 in [0,1]
    region_mask: (1,1,H,W) bool  (fixed region from original prediction)

    Returns dict with:
      x_cf, p0, p_cf, delta, history, roi, metrics
    """
    device = x.device
    x0 = x.detach()

    # If region is empty, fall back to global region (whole image)
    if region_mask.sum() == 0:
        region_mask = torch.ones_like(region_mask, dtype=torch.bool, device=device)

    # NEW: ROI-gated edits around region
    roi = dilate_mask(region_mask, k=cfg.roi_kernel)
    # If dilation still yields empty (shouldn't), fall back to whole image
    if roi.sum() == 0:
        roi = torch.ones_like(roi, dtype=torch.bool, device=device)

    # Trainable perturbation (delta), not the image directly
    delta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=cfg.lr)

    # Cache original prediction (for reporting)
    with torch.no_grad():
        logits0 = forward_logits(x0)
        p0 = torch.sigmoid(logits0)
        region_mean0 = float(p0[region_mask].mean().item())

    history = []

    for t in range(1, cfg.steps + 1):
        opt.zero_grad(set_to_none=True)

        # Apply ROI gate (only edit in roi)
        x_cf = x0 + roi.float() * delta
        x_cf = torch.clamp(x_cf, cfg.clamp_min, cfg.clamp_max)

        logits_cf = forward_logits(x_cf)
        p_cf = torch.sigmoid(logits_cf)

        # Objective: reduce mean prob in region
        region_mean = p_cf[region_mask].mean()

        # Regularizers
        l1 = (roi.float() * delta).abs().mean()   # only penalize actual editable region
        tv = tv_loss(x_cf)

        loss = region_mean + cfg.lam_l1 * l1 + cfg.lam_tv * tv
        loss.backward()
        opt.step()

        # NEW: hard cap the perturbation magnitude (prevents spikes/cheating)
        with torch.no_grad():
            delta.clamp_(-cfg.max_abs_delta, cfg.max_abs_delta)

        if t == 1 or (t % cfg.print_every == 0) or (t == cfg.steps):
            with torch.no_grad():
                rm = float(region_mean.item())
                l1v = float(l1.item())
                tvv = float(tv.item())
                history.append((t, rm, l1v, tvv))
                print(f"[CF {t:04d}/{cfg.steps}] region_mean={rm:.4f} l1={l1v:.4f} tv={tvv:.4f}")

    with torch.no_grad():
        x_cf = x0 + roi.float() * delta
        x_cf = torch.clamp(x_cf, cfg.clamp_min, cfg.clamp_max)

        logits_cf = forward_logits(x_cf)
        p_cf = torch.sigmoid(logits_cf)

        # Actual applied delta after clamp and gating
        applied_delta = (x_cf - x0)

        # --- NEW: auditable strength metrics ---
        abs_d = applied_delta.abs()
        metrics = {
            "region_px": int(region_mask.sum().item()),
            "roi_px": int(roi.sum().item()),
            "mean_prob_in_region_before": float(region_mean0),
            "mean_prob_in_region_after": float(p_cf[region_mask].mean().item()),
            "l1_mean": float(abs_d.mean().item()),
            "linf": float(abs_d.max().item()),
            "tv": float(tv_loss(x_cf).item()),
            "changed_frac_gt_small": float((abs_d > cfg.changed_thr_small).float().mean().item()),
            "changed_frac_gt_large": float((abs_d > cfg.changed_thr_large).float().mean().item()),
        }

    return {
        "x_cf": x_cf.detach(),
        "p0": p0.detach(),
        "p_cf": p_cf.detach(),
        "delta": applied_delta.detach(),  # this is the *applied* delta
        "roi": roi.detach(),
        "history": history,              # python list, not tensor
        "metrics": metrics,              # python dict
    }
