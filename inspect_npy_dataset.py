from pathlib import Path
import re
import numpy as np

DATA_DIR = Path("Preprocessed Data")  # change to Path("preprocessed_data") if you rename it

PATTERN = re.compile(
    r"^(data)(_mask)?_(train|test)_([a-zA-Z]+)_full\.npy$"
)

def summarize_array(arr: np.ndarray, name: str) -> None:
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, min={arr_min}, max={arr_max}")
    if np.issubdtype(arr.dtype, np.floating):
        print(f"    has_nan={np.isnan(arr).any()}, has_inf={np.isinf(arr).any()}")

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Could not find data directory: {DATA_DIR.resolve()}")

    files = sorted(DATA_DIR.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in: {DATA_DIR.resolve()}")

    # Index discovered files
    index = {}  # (split, domain) -> {"image": path, "mask": path}
    unmatched = []

    for f in files:
        m = PATTERN.match(f.name)
        if not m:
            unmatched.append(f.name)
            continue

        _, mask_flag, split, domain = m.groups()
        kind = "mask" if mask_flag else "image"
        key = (split, domain.lower())

        index.setdefault(key, {})
        index[key][kind] = f

    print("Discovered dataset entries:")
    for (split, domain), kinds in sorted(index.items()):
        img = kinds.get("image")
        msk = kinds.get("mask")
        print(f"- split={split:5s}, domain={domain:6s} | image={'Y' if img else 'N'} | mask={'Y' if msk else 'N'}")

    if unmatched:
        print("\nUnmatched .npy files (naming does not follow expected pattern):")
        for name in unmatched:
            print(f"  - {name}")

    # Validate pairing + basic stats
    print("\nValidation + quick stats:")
    for (split, domain), kinds in sorted(index.items()):
        print(f"\n[{split}/{domain}]")
        if "image" not in kinds:
            print("  ERROR: missing image file for this split/domain.")
            continue
        if "mask" not in kinds:
            print("  WARNING: missing mask file for this split/domain (ok for inference-only).")

        X = np.load(kinds["image"], mmap_mode="r")
        summarize_array(X, "image")

        if "mask" in kinds:
            Y = np.load(kinds["mask"], mmap_mode="r")
            summarize_array(Y, "mask")
            # First-dimension check (N alignment)
            if X.shape[0] != Y.shape[0]:
                print(f"  ERROR: image N={X.shape[0]} but mask N={Y.shape[0]} (misaligned!)")
            else:
                print(f"  OK: N aligned ({X.shape[0]})")

if __name__ == "__main__":
    main()
