# explain/targets.py
from __future__ import annotations
import torch


def predicted_region_mask_from_logits(logits: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """
    logits: (1,1,H,W)
    returns mask: (1,1,H,W) bool
    """
    probs = torch.sigmoid(logits)
    return (probs >= thr)


def target_mean_prob_in_region(logits: torch.Tensor, region_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scalar target for attribution/counterfactuals:
    mean(sigmoid(logits)) inside region_mask.
    If region is empty, falls back to global mean prob.
    """
    probs = torch.sigmoid(logits)
    if region_mask.sum() == 0:
        return probs.mean()
    return (probs[region_mask].mean() + eps)
