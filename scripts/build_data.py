#!/usr/bin/env python3
"""
Build ArSSR HR training patches from a directory of high-res NIfTI volumes.

Configurable version of the upstream data_bulid.py (which hardcoded the
HCP-1200 split). For each HR volume -> P random patch³ crops (kept away from the
black background border) -> {out_dir}/hr_train|hr_val/*.nii.gz, ready for
ArSSR's train.py.

IMPORTANT: ArSSR is self-supervised (HR patch is the target, LR is downsampled
from it inside data.py). So the HR volumes here should be genuinely HIGH-res
(near-isotropic) abdominal scans. Our anisotropic clinical data is the very
thing we want to SR; point --hr_dir at proper high-res abdominal volumes.

Run (from ArSSR root, venv active):
    python scripts/build_data.py --hr_dir /path/to/hr_volumes --out_dir data
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # ArSSR root (for `import utils`)
import utils  # ArSSR top-level utils.write_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr_dir", required=True, help="dir of HR .nii.gz volumes")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--patch", type=int, default=40, help="patch side (default 40)")
    ap.add_argument("--patches_per_vol", type=int, default=6)
    ap.add_argument("--train_frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(f"{args.out_dir}/hr_train", exist_ok=True)
    os.makedirs(f"{args.out_dir}/hr_val", exist_ok=True)

    vols = sorted(f for f in os.listdir(args.hr_dir)
                  if f.endswith((".nii.gz", ".nii")))
    assert vols, f"no NIfTI volumes found in {args.hr_dir}"
    n_train = int(round(len(vols) * args.train_frac))
    split = {"train": vols[:n_train], "val": vols[n_train:]}
    print(f"volumes: {len(vols)} -> train {len(split['train'])}, val {len(split['val'])}")

    p = args.patch
    margin = 40  # avoid black border (same as upstream)
    for phase, flist in split.items():
        for f in tqdm(flist, desc=phase):
            ref = os.path.join(args.hr_dir, f)
            try:
                arr = sitk.GetArrayFromImage(sitk.ReadImage(ref))
            except Exception as e:
                print(f"  skip {f}: {e}")
                continue
            if any(s < p + 2 * margin for s in arr.shape):
                print(f"  skip {f}: too small {arr.shape} for patch {p}+margin")
                continue
            name = f.split(".nii")[0]
            for i in range(args.patches_per_vol):
                x0 = int(rng.integers(margin, arr.shape[0] - p - margin))
                y0 = int(rng.integers(margin, arr.shape[1] - p - margin))
                z0 = int(rng.integers(margin, arr.shape[2] - p - margin))
                patch = arr[x0:x0 + p, y0:y0 + p, z0:z0 + p]
                utils.write_img(
                    vol=patch, ref_path=ref,
                    out_path=f"{args.out_dir}/hr_{phase}/{name}_{i}.nii.gz",
                )
    counts = {ph: len(os.listdir(f"{args.out_dir}/hr_{ph}")) for ph in split}
    print("patches written:", counts)


if __name__ == "__main__":
    main()
