#!/usr/bin/env python3
"""Verify the saved inference artifacts against every held-out prediction."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models/person96"))
from model import Person96CNN


def main():
    run = ROOT / "models/person96/run01"
    data = ROOT / "data/processed/person96"
    torch.set_num_threads(8)
    result = json.loads((run / "results.json").read_text())
    model = torch.jit.load(str(run / "model_float32.ts"), map_location="cpu").eval()
    with np.load(data / "test.npz", allow_pickle=False) as pack:
        x, y, ids = pack["x"], pack["y"], pack["sample_id"]
    with (run / "test_predictions.csv").open() as f:
        predictions = list(csv.DictReader(f))
    assert len(x) == len(predictions) == 5461
    assert np.array_equal(ids, [r["sample_id"] for r in predictions])
    saved_p = np.array([float(r["probability_person"]) for r in predictions])
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(x), 256):
            batch = torch.from_numpy(x[start:start + 256].copy()).float() / 255.
            probabilities.append(model(batch).sigmoid().numpy())
    p = np.concatenate(probabilities)
    assert np.isfinite(p).all()
    error = float(np.max(np.abs(p - saved_p)))
    assert error < 1e-5, error
    assert np.array_equal(p >= result["threshold"], saved_p >= result["threshold"])
    accuracy = float(((p >= result["threshold"]) == y).mean())
    assert abs(accuracy - result["metrics"]["test"]["accuracy"]) < 1e-12
    state = torch.load(run / "model_float32.pt", map_location="cpu", weights_only=False)["state_dict"]
    reconstructed = Person96CNN().eval().fused()
    reconstructed.load_state_dict(state, strict=True)
    with np.load(run / "weights_float32.npz", allow_pickle=False) as weights:
        assert set(weights.files) == set(state)
        for key in state:
            assert weights[key].dtype == np.float32 and np.isfinite(weights[key]).all()
            assert np.array_equal(weights[key], state[key].numpy())
    with torch.inference_mode():
        batch = torch.from_numpy(x[:64].copy()).float() / 255.
        assert torch.allclose(reconstructed(batch), model(batch), atol=2e-5, rtol=1e-5)
    # Decode two PNGs separately and compare end-to-end probabilities to the CSV.
    smoke = []
    for label in (0, 1):
        row = next(r for r in predictions if int(r["label"]) == label)
        with Image.open(data / row["image"]) as im:
            inputs = torch.from_numpy(np.asarray(im.convert("L"), dtype=np.float32).copy()).view(1, 1, 96, 96) / 255.
        with torch.inference_mode():
            prob = float(model(inputs).sigmoid())
        assert abs(prob - float(row["probability_person"])) < 1e-5
        smoke.append({"sample_id": row["sample_id"], "true_label": label, "probability_person": prob})
    audit = {"status": "PASS", "test_examples": len(y), "max_abs_export_probability_error": error,
        "all_test_labels_match_saved_predictions": True, "test_accuracy_reproduced": accuracy,
        "float32_npz_equals_checkpoint": True, "fused_state_dict_reload": True, "png_smoke": smoke}
    (ROOT / "report/person96_training/model_validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    files = [run / name for name in ("best.pt", "model_float32.pt", "model_float32.ts", "weights_float32.npz", "results.json")]
    (run / "SHA256SUMS").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + str(p.relative_to(ROOT)) + "\n" for p in files))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
