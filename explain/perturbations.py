# explain/perturbations.py
from __future__ import annotations
import torch
from typing import Callable


@torch.no_grad()
def occlusion_sensitivity(
    forward_logits: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    patch: int = 32,
    stride: int = 16,
    occlude_value: float = 0.0,
) -> torch.Tensor:
    """
    Returns sensitivity map (1,1,H,W) where higher means occluding that region reduces target more.
    """
    assert x.ndim == 4 and x.shape[0] == 1, "Expected x shape (1,1,H,W)"
    device = x.device

    # Base score
    base_logits = forward_logits(x)
    base_score = target_fn(base_logits).item()

    _, _, H, W = x.shape
    sens = torch.zeros((1, 1, H, W), device=device)
    counts = torch.zeros((1, 1, H, W), device=device)

    for top in range(0, H - patch + 1, stride):
        for left in range(0, W - patch + 1, stride):
            x_occ = x.clone()
            x_occ[:, :, top:top+patch, left:left+patch] = occlude_value

            logits_occ = forward_logits(x_occ)
            score_occ = target_fn(logits_occ).item()

            delta = base_score - score_occ  # drop in target
            sens[:, :, top:top+patch, left:left+patch] += delta
            counts[:, :, top:top+patch, left:left+patch] += 1.0

    sens = sens / torch.clamp(counts, min=1.0)
    return sens
