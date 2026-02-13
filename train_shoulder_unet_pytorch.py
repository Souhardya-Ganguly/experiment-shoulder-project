import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Config
# ----------------------------
@dataclass
class CFG:
    data_dir: Path = Path("Preprocessed Data")
    # Option A: train only on b
    train_img: str = "data_train_b_full.npy"
    train_msk: str = "data_mask_train_b_full.npy"
    test_b_img: str = "data_test_b_full.npy"
    test_b_msk: str = "data_mask_test_b_full.npy"
    test_h_img: str = "data_test_h_full.npy"
    test_h_msk: str = "data_mask_test_h_full.npy"

    # Training defaults
    epochs: int = 300
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42

    # Experiment / Aug
    exp_name: str = "e0"          # e0 | e1 | e2
    aug_profile: str = "none"     # none | classical
    deterministic: bool = False

    # E2 placeholder (diffusion) – not used yet
    use_synth: bool = False
    p_synth: float = 0.10
    synth_img: str = ""
    synth_msk: str = ""

    # Model / data
    in_ch: int = 1
    out_ch: int = 1
    img_size: int = 256

    # Threshold for binary metrics
    thr: float = 0.5

    # Output (will be overridden per experiment)
    out_dir: Path = Path("outputs")
    ckpt_dir: Path = Path("checkpoints")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", type=str, default="e0", choices=["e0", "e1", "e2"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--p_synth", type=float, default=0.10)
    p.add_argument("--synth_img", type=str, default="")
    p.add_argument("--synth_msk", type=str, default="")
    p.add_argument("--arch", type=str, default="unet_small",
               choices=["unet_small", "smp_unet_r34", "smp_unetpp_r34", "smp_deeplabv3p_r34", "smp_segformer_b0"])


    return p.parse_args()


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int, deterministic: bool = False) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For controlled ablations: prefer deterministic=True
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def ensure_dirs(cfg: CFG) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)


def _rand_uniform(rng: np.random.Generator, a: float, b: float) -> float:
    return float(rng.random() * (b - a) + a)


def apply_classical_aug(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    x: (H,W) float32 in [0,1]
    y: (H,W) float32 in {0,1}
    Returns augmented (x,y) with geometry preserved exactly (no interpolation).
    Suitable for ultrasound-like data.
    """
    # --- spatial (safe, no interpolation) ---
    if rng.random() < 0.5:
        x = np.fliplr(x)
        y = np.fliplr(y)
    if rng.random() < 0.2:   # vertical flips less common in US, keep lower
        x = np.flipud(x)
        y = np.flipud(y)

    # random 90-degree rotations
    k = int(rng.integers(0, 4))
    if k:
        x = np.rot90(x, k)
        y = np.rot90(y, k)

    # --- intensity (image only) ---
    # gamma jitter (mild)
    gamma = _rand_uniform(rng, 0.85, 1.15)
    x = np.clip(x, 0.0, 1.0) ** gamma

    # contrast + brightness
    contrast = _rand_uniform(rng, 0.9, 1.1)
    brightness = _rand_uniform(rng, -0.05, 0.05)
    x = np.clip((x - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)

    # additive Gaussian noise
    if rng.random() < 0.5:
        sigma = _rand_uniform(rng, 0.0, 0.03)
        x = np.clip(x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32), 0.0, 1.0)

    # multiplicative "speckle" (mild)
    if rng.random() < 0.4:
        speckle = _rand_uniform(rng, 0.0, 0.06)
        x = np.clip(x * (1.0 + rng.normal(0.0, speckle, size=x.shape).astype(np.float32)), 0.0, 1.0)

    return x.astype(np.float32), y.astype(np.float32)


# ----------------------------
# Dataset
# ----------------------------
class NPYSegDataset(Dataset):
    """
    Expects:
      X: (N, H, W) float in [0,1] (will cast to float32)
      Y: (N, H, W) bool or {0,1} (will cast to float32)
    Returns:
      x: torch.float32 (1, H, W)
      y: torch.float32 (1, H, W)
    """
    def __init__(self, x_path: Path, y_path: Path, *, is_train: bool, aug_profile: str, seed: int):
        self.X = np.load(x_path, mmap_mode="r")
        self.Y = np.load(y_path, mmap_mode="r")

        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(f"Misaligned N: X={self.X.shape}, Y={self.Y.shape}")
        if self.X.ndim != 3 or self.Y.ndim != 3:
            raise ValueError(f"Expected (N,H,W). Got X.ndim={self.X.ndim}, Y.ndim={self.Y.ndim}")

        self.is_train = bool(is_train)
        self.aug_profile = str(aug_profile)
        self.seed = int(seed)

        self._printed = False

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx].astype(np.float32)

        # Per-sample RNG: reproducible across runs, independent of dataloader shuffling
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)

        if self.is_train and self.aug_profile == "classical":
            x, y = apply_classical_aug(x, y, rng)

        # Add channel dim: (H,W)->(1,H,W)
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)

        if not self._printed and idx == 0:
            self._printed = True
            print(f"[Dataset] aug_profile={self.aug_profile} is_train={self.is_train}")
            print(f"[Dataset] X: shape={self.X.shape}, dtype={self.X.dtype}, min={float(np.min(self.X))}, max={float(np.max(self.X))}")
            print(f"[Dataset] Y: shape={self.Y.shape}, dtype={self.Y.dtype}, unique(sample0)={np.unique(self.Y[0])[:10]}")
            print(f"[Dataset] Returned tensors: x={x.shape} float32, y={y.shape} float32")

        return torch.from_numpy(x), torch.from_numpy(y)


class NPYArraysSegDataset(Dataset):
    """
    Dataset backed by in-memory or memmapped arrays.
    X: (N,H,W) float32 in [0,1]
    Y: (N,H,W) float32 or bool in {0,1}
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, *, is_train: bool, aug_profile: str, seed: int, name: str = "arrays"):
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"[{name}] Misaligned N: X={X.shape}, Y={Y.shape}")
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError(f"[{name}] Expected (N,H,W). Got X.ndim={X.ndim}, Y.ndim={Y.ndim}")

        self.X = X
        self.Y = Y
        self.is_train = bool(is_train)
        self.aug_profile = str(aug_profile)
        self.seed = int(seed)
        self.name = name
        self._printed = False

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx].astype(np.float32, copy=False)
        y = self.Y[idx].astype(np.float32, copy=False)

        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        if self.is_train and self.aug_profile == "classical":
            x, y = apply_classical_aug(x, y, rng)

        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)

        if not self._printed and idx == 0:
            self._printed = True
            print(f"[Dataset:{self.name}] aug_profile={self.aug_profile} is_train={self.is_train} N={len(self)}")

        return torch.from_numpy(x), torch.from_numpy(y)


class MixedSegDataset(Dataset):
    """
    Mix real + synthetic with probability p_synth per sample draw.
    Length is defined as len(real) to keep epoch budgets comparable to E0.
    """
    def __init__(self, real_ds: Dataset, synth_ds: Dataset, p_synth: float, seed: int):
        if not (0.0 <= p_synth <= 1.0):
            raise ValueError("p_synth must be in [0,1]")
        self.real_ds = real_ds
        self.synth_ds = synth_ds
        self.p_synth = float(p_synth)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.real_ds)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed * 9_999_991 + idx)
        if rng.random() < self.p_synth:
            j = int(rng.integers(0, len(self.synth_ds)))
            return self.synth_ds[j]
        else:
            i = idx % len(self.real_ds)
            return self.real_ds[i]


# ----------------------------
# Model: UNet (small, solid baseline)
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.outc(d1)
        return logits


# ----------------------------
# Model Factory
# ----------------------------
def build_model(arch: str, in_ch: int, out_ch: int):
    if arch == "unet_small":
        return UNet(in_ch=in_ch, out_ch=out_ch, base=32)

    import segmentation_models_pytorch as smp

    if arch == "smp_unet_r34":
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_ch,
            classes=out_ch,
        )

    if arch == "smp_unetpp_r34":
        return smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_ch,
            classes=out_ch,
        )

    if arch == "smp_deeplabv3p_r34":
        return smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_ch,
            classes=out_ch,
        )

    if arch == "smp_segformer_b0":
        # SegFormer supported via smp with timm encoders
        return smp.Unet(
            encoder_name="mit_b0",
            encoder_weights="imagenet",
            in_channels=in_ch,
            classes=out_ch,
        )

    raise ValueError(f"Unknown arch: {arch}")


# ----------------------------
# Losses & Metrics
# ----------------------------
def soft_dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2.0 * torch.sum(probs * targets, dim=(1, 2, 3)) + eps
    den = torch.sum(probs + targets, dim=(1, 2, 3)) + eps
    dice = num / den
    return 1.0 - dice.mean()


@torch.no_grad()
def dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-7) -> Tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).to(torch.uint8)
    t = (targets >= 0.5).to(torch.uint8)

    inter = torch.sum(preds & t).item()
    union = torch.sum(preds | t).item()
    p_sum = torch.sum(preds).item()
    t_sum = torch.sum(t).item()

    dice = (2.0 * inter + eps) / (p_sum + t_sum + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou)


# ----------------------------
# Train / Eval
# ----------------------------
def train_one_epoch(model, loader, optimizer, scaler, device) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    bce = nn.BCEWithLogitsLoss()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.startswith("cuda"))):
            logits = model(x)
            loss = 0.5 * bce(logits, y) + 0.5 * soft_dice_loss_with_logits(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def eval_model(model, loader, device, thr: float) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0

    bce = nn.BCEWithLogitsLoss()
    dice_acc = 0.0
    iou_acc = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = 0.5 * bce(logits, y) + 0.5 * soft_dice_loss_with_logits(logits, y)

        dice, iou = dice_iou_from_logits(logits, y, thr=thr)

        total_loss += loss.item() * x.size(0)
        dice_acc += dice * x.size(0)
        iou_acc += iou * x.size(0)
        n += x.size(0)

    return {
        "loss": total_loss / max(n, 1),
        "dice": dice_acc / max(n, 1),
        "iou": iou_acc / max(n, 1),
    }


@torch.no_grad()
def predict_full(model, loader, device) -> np.ndarray:
    model.eval()
    preds = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(1)  # (B,H,W)
        preds.append(probs.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0)


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    cfg = CFG()

    cfg.exp_name = args.exp
    cfg.seed = args.seed
    cfg.deterministic = bool(args.deterministic)
    cfg.p_synth = args.p_synth
    if not (0.0 <= cfg.p_synth <= 1.0):
        raise ValueError(f"--p_synth must be in [0,1], got {cfg.p_synth}")

    cfg.synth_img = args.synth_img
    cfg.synth_msk = args.synth_msk


    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr

    # Map exp -> augmentation
    if cfg.exp_name == "e0":
        cfg.aug_profile = "none"
    elif cfg.exp_name == "e1":
        cfg.aug_profile = "classical"
    elif cfg.exp_name == "e2":
        # placeholder: behave like e1 until we add diffusion samples
        cfg.aug_profile = "none"
        cfg.use_synth = True

    # Per-experiment output dirs (prevents overwriting)
    # Per-experiment + per-architecture + per-seed output dirs (prevents overwriting)
    arch_tag = args.arch
    seed_tag = f"seed{cfg.seed}"

    cfg.out_dir = Path("outputs") / cfg.exp_name / arch_tag / seed_tag
    cfg.ckpt_dir = Path("checkpoints") / cfg.exp_name / arch_tag / seed_tag


    ensure_dirs(cfg)
    set_seed(cfg.seed, deterministic=cfg.deterministic)

    device = cfg.device
    print(f"[Info] device={device} exp={cfg.exp_name} arch={args.arch} aug={cfg.aug_profile} deterministic={cfg.deterministic}")


    # Paths
    train_x = cfg.data_dir / cfg.train_img
    train_y = cfg.data_dir / cfg.train_msk
    testb_x = cfg.data_dir / cfg.test_b_img
    testb_y = cfg.data_dir / cfg.test_b_msk
    testh_x = cfg.data_dir / cfg.test_h_img
    testh_y = cfg.data_dir / cfg.test_h_msk

    for p in [train_x, train_y, testb_x, testb_y, testh_x, testh_y]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p.resolve()}")

    # Datasets / loaders
    ds_train_real = NPYSegDataset(train_x, train_y, is_train=True, aug_profile=cfg.aug_profile, seed=cfg.seed)

    if cfg.exp_name != "e2":
        ds_train = ds_train_real
    else:
        if not cfg.synth_img or not cfg.synth_msk:
            raise ValueError("E2 requires --synth_img and --synth_msk")

        synth_x_path = Path(cfg.synth_img)
        synth_y_path = Path(cfg.synth_msk)
        if not synth_x_path.exists() or not synth_y_path.exists():
            raise FileNotFoundError(f"Missing synth files: {synth_x_path} / {synth_y_path}")

        Xs = np.load(synth_x_path, mmap_mode="r")
        Ys = np.load(synth_y_path, mmap_mode="r")

        ds_train_synth = NPYArraysSegDataset(Xs, Ys, is_train=True, aug_profile="none", seed=cfg.seed, name="synth")
        ds_train = MixedSegDataset(ds_train_real, ds_train_synth, p_synth=cfg.p_synth, seed=cfg.seed)

        print(f"[Info] E2 mixing: p_synth={cfg.p_synth:.2f} | N_real={len(ds_train_real)} | N_synth={len(ds_train_synth)}")

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )

    ds_testb = NPYSegDataset(testb_x, testb_y, is_train=False, aug_profile="none", seed=cfg.seed)
    dl_testb = DataLoader(
        ds_testb,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )

    ds_testh = NPYSegDataset(testh_x, testh_y, is_train=False, aug_profile="none", seed=cfg.seed)
    dl_testh = DataLoader(
        ds_testh,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )

    # Model
    # model = UNet(in_ch=cfg.in_ch, out_ch=cfg.out_ch, base=32).to(device)

    model = build_model(args.arch, cfg.in_ch, cfg.out_ch).to(device)


    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    best_score = -1.0
    history = {
        "train_loss": [],
        "testb_loss": [],
        "testb_dice": [],
        "testb_iou": [],
        "testh_loss": [],
        "testh_dice": [],
        "testh_iou": [],
    }

    print(f"[Info] epochs={cfg.epochs}, batch_size={cfg.batch_size}, lr={cfg.lr}")

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, dl_train, optimizer, scaler, device)

        metrics_b = eval_model(model, dl_testb, device, thr=cfg.thr)
        metrics_h = eval_model(model, dl_testh, device, thr=cfg.thr)

        history["train_loss"].append(tr_loss)
        history["testb_loss"].append(metrics_b["loss"])
        history["testb_dice"].append(metrics_b["dice"])
        history["testb_iou"].append(metrics_b["iou"])
        history["testh_loss"].append(metrics_h["loss"])
        history["testh_dice"].append(metrics_h["dice"])
        history["testh_iou"].append(metrics_h["iou"])

        # Select best by test/b dice (kept consistent with your original script)
        score = metrics_b["dice"]
        if score > best_score:
            best_score = score
            ckpt_path = cfg.ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_score": best_score,
                    "cfg": cfg.__dict__,
                },
                ckpt_path,
            )

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            print(
                f"[Epoch {epoch:03d}/{cfg.epochs}] "
                f"train_loss={tr_loss:.4f} | "
                f"test/b loss={metrics_b['loss']:.4f} dice={metrics_b['dice']:.4f} iou={metrics_b['iou']:.4f} | "
                f"test/h loss={metrics_h['loss']:.4f} dice={metrics_h['dice']:.4f} iou={metrics_h['iou']:.4f} | "
                f"best_dice_b={best_score:.4f}"
            )

    # Load best checkpoint
    ckpt = torch.load(cfg.ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[Info] Loaded best checkpoint from epoch={ckpt['epoch']} with best_dice_b={ckpt['best_score']:.4f}")

    # Predict probabilities for both test sets
    pred_b = predict_full(model, dl_testb, device)
    pred_h = predict_full(model, dl_testh, device)

    np.save(cfg.out_dir / "pred_test_b.npy", pred_b)
    np.save(cfg.out_dir / "pred_test_h.npy", pred_h)

    # Final metrics
    final_b = eval_model(model, dl_testb, device, thr=cfg.thr)
    final_h = eval_model(model, dl_testh, device, thr=cfg.thr)

    results = {
        "experiment": cfg.exp_name,
        "aug_profile": cfg.aug_profile,
        "final_test_b": final_b,
        "final_test_h": final_h,
        "best_epoch": int(ckpt["epoch"]),
        "best_score_test_b_dice": float(ckpt["best_score"]),
        "threshold": cfg.thr,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
        "arch": args.arch,
    }

    with open(cfg.out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot loss curve
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["testb_loss"], label="test_b_loss")
    plt.plot(history["testh_loss"], label="test_h_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Training / Test Loss ({cfg.exp_name}, arch={args.arch}, aug={cfg.aug_profile})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.out_dir / "loss_curve.png", dpi=300)

    # If you're on a headless server, plt.show() can hang; keep it off by default.
    # plt.show()

    print("[Done] Saved predictions + metrics:")
    print(f"  - {cfg.out_dir / 'pred_test_b.npy'}")
    print(f"  - {cfg.out_dir / 'pred_test_h.npy'}")
    print(f"  - {cfg.out_dir / 'metrics.json'}")
    print(f"  - {cfg.out_dir / 'loss_curve.png'}")
    print(f"  - {cfg.ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
