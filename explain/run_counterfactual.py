# explain/run_counterfactual.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from explain.adapters import ModelSpec, load_model_from_ckpt
from explain.targets import predicted_region_mask_from_logits
from explain.counterfactuals import CFConfig, counterfactual_reduce_region_confidence
from explain.viz import save_counterfactual_panel


def load_npy_sample(data_dir: Path, split: str, idx: int):
    if split == "test_b":
        x_path = data_dir / "data_test_b_full.npy"
    elif split == "test_h":
        x_path = data_dir / "data_test_h_full.npy"
    else:
        raise ValueError("split must be test_b or test_h")

    X = np.load(x_path, mmap_mode="r")  # (N,H,W)
    img = X[idx].astype(np.float32)
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return x, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="Preprocessed Data")
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    ap.add_argument("--training_module", type=str, default="train_shoulder_unet_pytorch")
    ap.add_argument("--split", type=str, default="test_b", choices=["test_b", "test_h"])
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5)

    # CF params (defaults are reasonable starters)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lam_l1", type=float, default=0.02)
    ap.add_argument("--lam_tv", type=float, default=0.10)

    ap.add_argument("--out_dir", type=str, default="explain_outputs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    spec = ModelSpec(training_module=args.training_module)
    model = load_model_from_ckpt(Path(args.ckpt), device=device, spec=spec)

    def forward_logits(x):
        return model(x)

    x, img = load_npy_sample(data_dir, args.split, args.idx)
    x = x.to(device)

    # Region fixed from original prediction
    with torch.no_grad():
        logits0 = forward_logits(x)
        region = predicted_region_mask_from_logits(logits0, thr=args.thr)

    cfg = CFConfig(
        steps=args.steps,
        lr=args.lr,
        lam_l1=args.lam_l1,
        lam_tv=args.lam_tv,
        print_every=max(1, args.steps // 8),
    )

    out = counterfactual_reduce_region_confidence(
        forward_logits=forward_logits,
        x=x,
        region_mask=region,
        cfg=cfg,
    )

    x_cf = out["x_cf"].squeeze().cpu().numpy().astype(np.float32)
    p0 = out["p0"].squeeze().cpu().numpy().astype(np.float32)
    p_cf = out["p_cf"].squeeze().cpu().numpy().astype(np.float32)
    delta = out["delta"].squeeze().cpu().numpy().astype(np.float32)
    pred_mask = region.squeeze().cpu().numpy().astype(np.uint8)

    # Save arrays
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{args.split}_idx{args.idx:04d}_x_cf.npy", x_cf)
    np.save(out_dir / f"{args.split}_idx{args.idx:04d}_p0.npy", p0)
    np.save(out_dir / f"{args.split}_idx{args.idx:04d}_p_cf.npy", p_cf)
    np.save(out_dir / f"{args.split}_idx{args.idx:04d}_delta.npy", delta)

    # Save visualization
    save_counterfactual_panel(
        out_dir / f"{args.split}_idx{args.idx:04d}_CF.png",
        image=img,
        pred_mask=pred_mask,
        x_cf=x_cf,
        p0=p0,
        p_cf=p_cf,
        delta=delta,
        title=f"{args.split} idx={args.idx} | Counterfactual reduce region confidence",
    )

    # Print summary numbers
    region_px = pred_mask.sum()
    region_mean_before = float(p0[pred_mask > 0].mean()) if region_px > 0 else float(p0.mean())
    region_mean_after = float(p_cf[pred_mask > 0].mean()) if region_px > 0 else float(p_cf.mean())
    print(f"[Summary] region_px={int(region_px)}")
    print(f"[Summary] mean prob in region: before={region_mean_before:.4f} after={region_mean_after:.4f}")
    print(f"[Saved] {out_dir.resolve()}")


if __name__ == "__main__":
    main()
