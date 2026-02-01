# explain/attributions.py
from __future__ import annotations
import torch
from typing import Callable


def integrated_gradients(
    forward_logits: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    baseline: torch.Tensor | None = None,
    steps: int = 32,
) -> torch.Tensor:
    """
    forward_logits: function(x)->logits (1,1,H,W)
    x: (1,1,H,W) float32
    target_fn: function(logits)->scalar tensor
    baseline: if None, zeros
    returns attribution map: (1,1,H,W)
    """
    assert x.ndim == 4 and x.shape[0] == 1, "Expected x shape (1,1,H,W)"

    device = x.device
    if baseline is None:
        baseline = torch.zeros_like(x, device=device)

    # Scale inputs from baseline to x
    alphas = torch.linspace(0.0, 1.0, steps, device=device).view(steps, 1, 1, 1)
    x_scaled = baseline + alphas * (x - baseline)  # (steps,1,H,W)

    # We’ll accumulate gradients across steps
    grads = torch.zeros_like(x_scaled)

    for i in range(steps):
        xi = x_scaled[i:i+1].clone().detach().requires_grad_(True)  # (1,1,H,W)
        logits = forward_logits(xi)
        scalar = target_fn(logits)  # scalar
        scalar.backward()

        gi = xi.grad.detach()  # (1,1,H,W)
        grads[i:i+1] = gi

    # Average gradients and scale by (x - baseline)
    avg_grads = grads.mean(dim=0, keepdim=True)  # (1,1,H,W)
    attr = (x - baseline) * avg_grads
    return attr
