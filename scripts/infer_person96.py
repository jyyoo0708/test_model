#!/usr/bin/env python3
"""Infer a prepared square image patch with the exported floating point model."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--model", type=Path, default=Path("models/person96/run01/model_float32.ts"))
    ap.add_argument("--threshold", type=float)
    args = ap.parse_args()
    threshold = args.threshold
    if threshold is None:
        threshold = json.loads((args.model.parent / "results.json").read_text())["threshold"]
    torch.set_num_threads(1)
    with Image.open(args.image) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        if im.width != im.height:
            raise ValueError("Provide a square region patch; see report for full-frame inference limitations.")
        im = im.resize((96, 96), Image.Resampling.LANCZOS)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).view(1, 1, 96, 96) / 255.
    model = torch.jit.load(str(args.model), map_location="cpu").eval()
    with torch.inference_mode():
        p = float(model(x).sigmoid().item())
    print(json.dumps({"image": str(args.image), "probability_person": p, "threshold": threshold,
                      "label": int(p >= threshold)}, indent=2))


if __name__ == "__main__":
    main()
