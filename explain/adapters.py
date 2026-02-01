# explain/adapters.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import importlib
import torch
import torch.nn as nn


@dataclass
class ModelSpec:
    # Name of the python file (module) where UNet is defined (WITHOUT .py)
    training_module: str = "train_shoulder_unet_pytorch"   # <-- change if your file name differs
    model_class_name: str = "UNet"
    in_ch: int = 1
    out_ch: int = 1
    base: int = 32


def _load_model_class(training_module: str, model_class_name: str):
    mod = importlib.import_module(training_module)
    cls = getattr(mod, model_class_name)
    return cls


def load_model_from_ckpt(
    ckpt_path: Path,
    device: str,
    spec: ModelSpec = ModelSpec(),
) -> nn.Module:
    """
    Loads model weights from your checkpoint that stores: {"model_state": state_dict, ...}
    """
    ModelCls = _load_model_class(spec.training_module, spec.model_class_name)
    model = ModelCls(in_ch=spec.in_ch, out_ch=spec.out_ch, base=spec.base).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    x: (1,1,H,W) float32
    returns logits: (1,1,H,W)
    """
    return model(x)
