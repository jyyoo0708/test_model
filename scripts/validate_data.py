#!/usr/bin/env python3
"""Audit every exported patch, label, geometry, split, and NPZ row."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

POPCOUNT = getattr(int, "bit_count", lambda x: bin(x).count("1"))


def xyxy(b):
    return [b[0], b[1], b[0] + max(0, b[2]), b[1] + max(0, b[3])]


def overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/person96"))
    args = ap.parse_args()
    p = args.data
    c = json.loads((p / "config.json").read_text())
    sources = {r["ID"]: r for r in map(json.loads, (p / "source_annotations.jsonl").open())}
    groups = {}
    source_sets, pixel_sets, id_sets = {}, {}, {}
    checks = Counter()
    for s in sources.values():
        assert groups.setdefault(s["group_id"], s["split"]) == s["split"]
    for split in ("train", "val", "test"):
        rows = [json.loads(line) for line in (p / f"{split}.jsonl").open()]
        with np.load(p / f"{split}.npz", allow_pickle=False) as pack:
            x, y, ids = pack["x"], pack["y"], pack["sample_id"]
        assert x.shape == (len(rows), 1, 96, 96) and x.dtype == np.uint8
        assert y.shape == (len(rows),) and y.dtype == np.int64
        assert set(y.tolist()) == {0, 1}
        source_sets[split], pixel_sets[split], id_sets[split] = set(), set(), set()
        for i, row in enumerate(rows):
            s = sources[row["source_id"]]
            assert s["split"] == row["split"] == split
            assert s["group_id"] == row["group_id"]
            assert row["sample_id"] == ids[i] and row["label"] == y[i]
            with Image.open(p / row["image"]) as im:
                assert im.mode == "L" and im.size == (96, 96)
                pixels = np.asarray(im)
                assert np.array_equal(pixels, x[i, 0])
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            assert digest == row["pixel_sha256"] and digest not in pixel_sets[split]
            pixel_sets[split].add(digest)
            id_sets[split].add(row["sample_id"])
            source_sets[split].add(row["source_id"])
            crop = row["crop_xyxy"]
            assert crop[2] - crop[0] == crop[3] - crop[1] > 0
            real = [max(0, crop[0]), max(0, crop[1]), min(s["width"], crop[2]), min(s["height"], crop[3])]
            if row["label"] == 1:
                b = s["gtboxes"][row["target_index"]]
                assert b["tag"] == "person" and not b.get("extra", {}).get("ignore", 0)
                f, v = xyxy(b["fbox"]), xyxy(b["vbox"])
                a = b["fbox"][2] * b["fbox"][3]
                assert b["fbox"][3] >= c["min_source_height"] and b["fbox"][2] >= c["min_source_width"]
                visible_rect = [max(v[0], f[0]), max(v[1], f[1]), min(v[2], f[2]), min(v[3], f[3])]
                visible = overlap(visible_rect, real) / a
                retained = overlap(f, real) / a
                assert visible >= c["visible_ratio_min"] and retained >= c["retained_ratio_min"]
                assert abs(visible - row["visible_ratio"]) < 1e-9
                assert abs(retained - row["retained_ratio"]) < 1e-9
                side = crop[2] - crop[0]
                expected_box = [(max(f[0], real[0]) - crop[0]) * 96 / side,
                                (max(f[1], real[1]) - crop[1]) * 96 / side,
                                (min(f[2], real[2]) - crop[0]) * 96 / side,
                                (min(f[3], real[3]) - crop[1]) * 96 / side]
                assert np.allclose(expected_box, row["bbox_xyxy_96"])
                x1, y1, x2, y2 = expected_box
                assert 0 <= x1 < x2 <= 96 and 0 <= y1 < y2 <= 96
                assert x2 - x1 >= c["min_input_width"] and y2 - y1 >= c["min_input_height"]
                checks["positive_geometry"] += 1
            else:
                assert crop == real  # Negative crops have no synthetic padding.
                target = row["target_index"]
                margin = c["background_margin"] if row["negative_kind"] == "background" else 0
                if row["negative_kind"] == "partial_person":
                    b = s["gtboxes"][target]
                    assert b["derived"]["eligible"]
                    retained = overlap(crop, xyxy(b["fbox"])) / (b["fbox"][2] * b["fbox"][3])
                    assert 0 < retained <= c["partial_retained_max"]
                    assert abs(retained - row["retained_ratio"]) < 1e-9
                else:
                    assert row["negative_kind"] == "background" and target is None
                for j, b in enumerate(s["gtboxes"]):
                    if j == target:
                        continue
                    for key in ("fbox", "vbox", "hbox"):
                        if key not in b or b[key][2] <= 0 or b[key][3] <= 0:
                            continue
                        q = xyxy(b[key])
                        q = [q[0] - margin, q[1] - margin, q[2] + margin, q[3] + margin]
                        assert overlap(crop, q) == 0, (row["sample_id"], j)
                checks[row["negative_kind"] + "_geometry"] += 1
            checks["png_npz_pixel_match"] += 1
        assert len(id_sets[split]) == len(rows)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not source_sets[a] & source_sets[b]
        assert not pixel_sets[a] & pixel_sets[b]
        assert not id_sets[a] & id_sets[b]
    source_list = list(sources.values())
    signatures = [int(s["dhash"], 16) for s in source_list]
    for i, a in enumerate(source_list):
        for j in range(i):
            if a["split"] != source_list[j]["split"]:
                assert POPCOUNT(signatures[i] ^ signatures[j]) > c["dhash_distance"]
    report = {"status": "PASS", "source_images": len(sources), "checks": dict(checks),
              "no_cross_split_source_or_pixel_duplicates": True,
              "no_cross_split_dhash_neighbors_within_threshold": True,
              "note": "Geometry audit uses original bounding boxes; it is not a human visibility segmentation audit."}
    Path("report/data_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
