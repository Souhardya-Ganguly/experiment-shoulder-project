# explain/run_explain.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from explain.adapters import ModelSpec, load_model_from_ckpt
from explain.targets import predicted_region_mask_from_logits, target_mean_prob_in_region
from explain.attributions import integrated_gradients
from explain.perturbations import occlusion_sensitivity
from explain.viz import save_overlay


def load_npy_sample(data_dir: Path, split: str, idx: int):
    """
    split: 'test_b' or 'test_h' (we default to test_b)
    Returns x (1,1,H,W) torch float32 and raw image (H,W) numpy
    """
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
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--ig_steps", type=int, default=32)
    ap.add_argument("--occ_patch", type=int, default=32)
    ap.add_argument("--occ_stride", type=int, default=16)
    ap.add_argument("--out_dir", type=str, default="explain_outputs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    spec = ModelSpec(training_module=args.training_module)
    model = load_model_from_ckpt(Path(args.ckpt), device=device, spec=spec)

    def forward_logits(x):
        return model(x)

    for idx in args.indices:
        x, img = load_npy_sample(data_dir, args.split, idx)
        x = x.to(device)

        # 1) Get predicted region for the DEFAULT target
        with torch.no_grad():
            logits = forward_logits(x)
            region = predicted_region_mask_from_logits(logits, thr=args.thr)

        # Define scalar target: mean prob in predicted region
        def target_fn(lgts):
            return target_mean_prob_in_region(lgts, region)

        # 2) Integrated Gradients
        attr = integrated_gradients(
            forward_logits=forward_logits,
            x=x,
            target_fn=target_fn,
            baseline=None,        # default zeros baseline
            steps=args.ig_steps,
        )
        attr_map = attr.abs().squeeze().detach().cpu().numpy()  # (H,W)

        # 3) Occlusion sensitivity
        occ = occlusion_sensitivity(
            forward_logits=forward_logits,
            x=x,
            target_fn=target_fn,
            patch=args.occ_patch,
            stride=args.occ_stride,
            occlude_value=0.0,
        )
        occ_map = occ.squeeze().detach().cpu().numpy()  # (H,W)

        pred_mask = region.squeeze().detach().cpu().numpy().astype(np.uint8)  # (H,W)

        # Save overlays
        save_overlay(
            out_dir / f"{args.split}_idx{idx:04d}_IG.png",
            image=img,
            pred_mask=pred_mask,
            heatmap=attr_map,
            title=f"{args.split} idx={idx} | Integrated Gradients",
        )

        save_overlay(
            out_dir / f"{args.split}_idx{idx:04d}_OCC.png",
            image=img,
            pred_mask=pred_mask,
            heatmap=occ_map,
            title=f"{args.split} idx={idx} | Occlusion Sensitivity",
        )

        print(f"[Saved] {args.split} idx={idx} -> {out_dir}")

    print("[Done] Explainability outputs saved in:", out_dir.resolve())


if __name__ == "__main__":
    main()
