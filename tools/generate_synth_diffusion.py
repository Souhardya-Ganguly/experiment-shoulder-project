#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import time

import numpy as np
import torch
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_x", type=str, required=True, help="Input train images .npy (N,H,W) float in [0,1]")
    p.add_argument("--in_y", type=str, required=True, help="Input train masks .npy (N,H,W) bool or {0,1}")
    p.add_argument("--out_x", type=str, required=True, help="Output synth images .npy (N,H,W) float32 in [0,1]")
    p.add_argument("--out_y", type=str, required=True, help="Output synth masks .npy (N,H,W) bool (copied)")
    p.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--strength", type=float, default=0.20)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=1.0, help="Keep low for structure preservation")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", type=int, default=0, help="Start index (resume support)")
    p.add_argument("--end", type=int, default=-1, help="End index exclusive; -1 means full length")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--meta", type=str, default="", help="Optional JSON metadata output path")
    return p.parse_args()


def to_pil_rgb_from_gray01(x_hw: np.ndarray) -> Image.Image:
    """x_hw float in [0,1], shape (H,W) -> PIL RGB"""
    x8 = np.clip(x_hw * 255.0, 0, 255).astype(np.uint8)
    pil_l = Image.fromarray(x8, mode="L")
    return pil_l.convert("RGB")


def gray01_from_pil(pil_img: Image.Image) -> np.ndarray:
    """PIL -> float32 gray in [0,1], shape (H,W)"""
    g = pil_img.convert("L")
    a = np.asarray(g).astype(np.float32) / 255.0
    return np.clip(a, 0.0, 1.0)


def main():
    args = parse_args()

    in_x = Path(args.in_x)
    in_y = Path(args.in_y)
    out_x = Path(args.out_x)
    out_y = Path(args.out_y)
    out_x.parent.mkdir(parents=True, exist_ok=True)
    out_y.parent.mkdir(parents=True, exist_ok=True)

    X = np.load(in_x, mmap_mode="r")  # (N,H,W) float
    Y = np.load(in_y, mmap_mode="r")  # (N,H,W) bool / {0,1}

    if X.ndim != 3 or Y.ndim != 3:
        raise ValueError(f"Expected (N,H,W). Got X.ndim={X.ndim}, Y.ndim={Y.ndim}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Misaligned N: X={X.shape}, Y={Y.shape}")
    if X.shape[1:] != (256, 256):
        print(f"[Warn] Expected 256x256 but got {X.shape[1:]}; script will still run.")

    N = X.shape[0]
    start = int(args.start)
    end = N if args.end == -1 else int(args.end)
    if not (0 <= start < end <= N):
        raise ValueError(f"Invalid start/end: start={start}, end={end}, N={N}")

    # Create output .npy as memory-mapped files (no huge RAM spike)
    Xs = np.lib.format.open_memmap(out_x, mode="w+", dtype=np.float32, shape=(end - start, X.shape[1], X.shape[2]))
    Ys = np.lib.format.open_memmap(out_y, mode="w+", dtype=np.bool_, shape=(end - start, Y.shape[1], Y.shape[2]))

    # Copy masks (1:1 mapping) – assumes structure preserved by low-strength img2img
    print(f"[Info] Copying masks -> {out_y}")
    for i in range(start, end):
        Ys[i - start] = Y[i].astype(np.bool_, copy=False)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    print(f"[Info] Loading diffusion pipeline: {args.model_id}")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        safety_checker=None,   # medical-like grayscale; we aren't generating unsafe content
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    # Slight memory optimizations
    if device.startswith("cuda"):
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    # Prompts: keep empty and guidance low to preserve structure
    prompt = ""
    negative_prompt = ""

    bs = int(args.batch_size)
    strength = float(args.strength)
    steps = int(args.steps)
    guidance = float(args.guidance)
    base_seed = int(args.seed)

    print(f"[Info] Generating synth images -> {out_x}")
    print(f"[Info] Params: strength={strength}, steps={steps}, guidance={guidance}, batch_size={bs}, seed={base_seed}")
    t0 = time.time()

    out_idx = 0
    for i0 in range(start, end, bs):
        i1 = min(i0 + bs, end)
        batch = []
        gens = []

        for i in range(i0, i1):
            x = X[i].astype(np.float32, copy=False)
            batch.append(to_pil_rgb_from_gray01(x))

            # Seed per image for reproducibility + resumability
            gens.append(torch.Generator(device=device).manual_seed(base_seed * 1_000_003 + i))

        # diffusers supports list inputs
        with torch.inference_mode():
            result = pipe(
                prompt=[prompt] * len(batch),
                image=batch,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                negative_prompt=[negative_prompt] * len(batch),
                generator=gens,
            )

        images = result.images  # list of PIL
        for j, pil_img in enumerate(images):
            Xs[out_idx + j] = gray01_from_pil(pil_img)

        out_idx += len(images)

        if (i0 == start) or ((i0 - start) // bs) % 10 == 0:
            done = i1 - start
            total = end - start
            elapsed = time.time() - t0
            print(f"[Progress] {done}/{total} done ({100.0 * done / total:.1f}%) | elapsed={elapsed/60:.1f} min")

    # Flush memmaps
    del Xs
    del Ys

    meta = {
        "in_x": str(in_x),
        "in_y": str(in_y),
        "out_x": str(out_x),
        "out_y": str(out_y),
        "model_id": args.model_id,
        "strength": strength,
        "steps": steps,
        "guidance": guidance,
        "batch_size": bs,
        "seed": base_seed,
        "start": start,
        "end": end,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    meta_path = Path(args.meta) if args.meta else (out_x.parent / (out_x.stem + "_meta.json"))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[Done] Wrote:")
    print(f"  - {out_x}")
    print(f"  - {out_y}")
    print(f"  - {meta_path}")


if __name__ == "__main__":
    main()
