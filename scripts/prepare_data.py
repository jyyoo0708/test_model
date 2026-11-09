#!/usr/bin/env python3
"""Deterministic CrowdHuman -> 96x96 single-channel binary classification data.

Run from the repository root. Raw input is never modified. Refuses to overwrite
an existing output directory, so experiments cannot silently mix configurations.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

RESAMPLING = getattr(Image, "Resampling", Image)
POPCOUNT = getattr(int, "bit_count", lambda x: bin(x).count("1"))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def rect(box):
    x, y, w, h = box
    return [x, y, x + max(w, 0), y + max(h, 0)]


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection(a, b):
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def dhash(image):
    a = np.asarray(image.convert("L").resize((9, 8), RESAMPLING.BILINEAR))
    return int.from_bytes(np.packbits(a[:, 1:] > a[:, :-1]).tobytes(), "big")


def grouped_splits(records, signatures, seed, distance):
    # Compare all pairs (only 4,370 images), then take connected components.
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, h in enumerate(signatures):
        for j in range(i):
            if POPCOUNT(h ^ signatures[j]) <= distance:
                a, b = root(i), root(j)
                parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[root(i)].append(i)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    counts = Counter()
    # Greedy fill at group boundaries: source images never cross splits.
    for key in keys:
        split = "train" if counts["train"] < round(len(records) * .8) else (
            "val" if counts["val"] < round(len(records) * .1) else "test")
        for i in groups[key]:
            records[i]["split"] = split
            records[i]["group_id"] = records[key]["ID"]
        counts[split] += len(groups[key])
    return {"source_counts": dict(counts), "groups": len(groups),
            "multi_image_groups": sum(len(v) > 1 for v in groups.values()),
            "largest_group": max(map(len, groups.values()))}


def source_reasons(box, width, height, c):
    f = rect(box["fbox"])
    v = rect(box["vbox"])
    reasons = []
    if box.get("tag") != "person" or box.get("extra", {}).get("ignore", 0):
        reasons.append("source_ignore")
    if area(f) == 0:
        reasons.append("invalid_box")
    if box["fbox"][3] < c["min_source_height"] or box["fbox"][2] < c["min_source_width"]:
        reasons.append("small_source")
    visible = area(intersection(intersection(v, f), [0, 0, width, height])) / max(area(f), 1)
    retained = area(intersection(f, [0, 0, width, height])) / max(area(f), 1)
    if visible < c["visible_ratio_min"]:
        reasons.append("visible_ratio_below_threshold")
    if retained < c["retained_ratio_min"]:
        reasons.append("truncated_in_source")
    return reasons, visible, retained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("data/config.json"))
    ap.add_argument("--output", type=Path, default=Path("data/processed/person96"))
    args = ap.parse_args()
    c = json.loads(args.config.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "config.json", c)
    raw = Path("data/raw/crowdhuman")
    records = sorted([json.loads(line) for line in (raw / "annotation_val.odgt").open()], key=lambda r: r["ID"])
    archive = zipfile.ZipFile(raw / "CrowdHuman_val.zip")
    members = {Path(n).stem: n for n in archive.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))}
    assert len(records) == 4370 and all(r["ID"] in members for r in records)
    signatures = []
    for i, r in enumerate(records):
        with archive.open(members[r["ID"]]) as f, Image.open(f) as im:
            signatures.append(dhash(im))
            r["width"], r["height"] = im.size
        r["zip_member"] = members[r["ID"]]
        r["dhash"] = f"{signatures[-1]:016x}"
        if (i + 1) % 500 == 0:
            print(f"Source scan {i + 1}/{len(records)}", flush=True)
    grouping = grouped_splits(records, signatures, c["seed"], c["dhash_distance"])
    print(f"Grouping: {grouping}", flush=True)
    reason_counts = Counter()
    pixel_hashes = set()
    skipped = Counter()
    data = {s: {"x": [], "y": [], "ids": []} for s in ("train", "val", "test")}
    manifests = {s: (out / f"{s}.jsonl").open("w") for s in data}
    sources_file = (out / "source_annotations.jsonl").open("w")
    n = c["input_size"]
    preview = defaultdict(list)

    def save(im, crop, source, label, details):
        patch = im.crop(crop).resize((n, n), RESAMPLING.LANCZOS)
        pixels = np.asarray(patch, dtype=np.uint8)
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        if digest in pixel_hashes:
            skipped["duplicate_pixels"] += 1
            return False
        if pixels.std() < 3:
            skipped["nearly_constant"] += 1
            return False
        pixel_hashes.add(digest)
        split = source["split"]
        sample_id = f'{source["ID"]}_{details["suffix"]}'
        cls = "person" if label else "non_person"
        rel = Path("images") / split / cls / f"{sample_id}.png"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        patch.save(out / rel)
        row = {"sample_id": sample_id, "image": str(rel), "label": label,
               "class_name": cls, "source_id": source["ID"], "group_id": source["group_id"],
               "split": split, "crop_xyxy": crop, "pixel_sha256": digest, **details}
        manifests[split].write(json.dumps(row) + "\n")
        data[split]["x"].append(pixels[None])
        data[split]["y"].append(label)
        data[split]["ids"].append(sample_id)
        if len(preview[(split, label)]) < 48:
            preview[(split, label)].append((patch.copy(), sample_id))
        return True

    for i, r in enumerate(records):
        width, height = r["width"], r["height"]
        rng = random.Random(f'{c["seed"]}:{r["ID"]}')
        with archive.open(r["zip_member"]) as f, Image.open(f) as original:
            im = ImageOps.exif_transpose(original).convert("L")
        assert im.size == (width, height), "Unexpected EXIF rotation"
        eligible = []
        for j, b in enumerate(r["gtboxes"]):
            reasons, visible, retained = source_reasons(b, width, height, c)
            b["derived"] = {"eligible": not reasons, "ignore_reasons": reasons,
                            "visible_ratio": visible, "source_retained_ratio": retained}
            reason_counts.update(reasons)
            if not reasons:
                eligible.append(j)
        sources_file.write(json.dumps(r) + "\n")
        rng.shuffle(eligible)
        positive_count = 0
        for j in eligible[:c["max_positive_instances_per_source"]]:
            b = r["gtboxes"][j]
            fbox, vbox = rect(b["fbox"]), rect(b["vbox"])
            for k, scale in enumerate(c["context_scales"]):
                side = int(np.ceil(max(b["fbox"][2:]) * scale))
                cx, cy = (fbox[0] + fbox[2]) / 2, (fbox[1] + fbox[3]) / 2
                x = int(round(cx - side / 2))
                y = int(round(cy - side / 2))
                crop = [x, y, x + side, y + side]
                actual = intersection(crop, [0, 0, width, height])
                clipped = intersection(fbox, actual)
                retained = area(clipped) / max(area(fbox), 1)
                visible = area(intersection(intersection(vbox, fbox), actual)) / max(area(fbox), 1)
                bbox96 = [(clipped[0] - x) * n / side, (clipped[1] - y) * n / side,
                          (clipped[2] - x) * n / side, (clipped[3] - y) * n / side]
                if retained < c["retained_ratio_min"] or visible < c["visible_ratio_min"]:
                    skipped["crop_visibility"] += 1
                    continue
                if bbox96[3] - bbox96[1] < c["min_input_height"] or bbox96[2] - bbox96[0] < c["min_input_width"]:
                    skipped["small_at_input"] += 1
                    continue
                positive_count += save(im, crop, r, 1, {"suffix": f"p{j:03d}s{k}",
                    "target_index": j, "bbox_xyxy_96": bbox96,
                    "visible_ratio": visible, "retained_ratio": retained,
                    "context_scale": scale})
        # Even ignored, tiny, and heavily occluded people block background crops.
        blocked = []
        margin = c["background_margin"]
        for b in r["gtboxes"]:
            for key in ("fbox", "vbox", "hbox"):
                if key in b and area(rect(b[key])) > 0:
                    q = rect(b[key])
                    blocked.append([q[0] - margin, q[1] - margin, q[2] + margin, q[3] + margin])
        partial_count = 0
        for j in eligible:
            if partial_count >= c["max_partial_per_source"]:
                break
            fbox = rect(r["gtboxes"][j]["fbox"])
            side = max(32, int((fbox[3] - fbox[1]) * .45))
            x = int(round((fbox[0] + fbox[2] - side) / 2))
            for y in (int(fbox[1]), int(fbox[3] - side)):
                crop = [x, y, x + side, y + side]
                if x < 0 or y < 0 or x + side > width or y + side > height:
                    continue
                retained = area(intersection(crop, fbox)) / max(area(fbox), 1)
                if retained > c["partial_retained_max"]:
                    continue
                if any(area(intersection(crop, rect(b[key]))) > 0
                       for idx, b in enumerate(r["gtboxes"]) if idx != j
                       for key in ("fbox", "vbox", "hbox") if key in b):
                    continue
                partial_count += save(im, crop, r, 0, {"suffix": f"t{j:03d}_{y}",
                    "negative_kind": "partial_person", "target_index": j,
                    "bbox_xyxy_96": None, "visible_ratio": None,
                    "retained_ratio": retained})
                break
        target_bg = min(c["max_background_per_source"], max(2, positive_count - partial_count))
        bg_count = 0
        for attempt in range(c["background_attempts"]):
            side = rng.randint(max(32, int(min(width, height) * .12)), max(32, int(min(width, height) * .55)))
            if side > min(width, height):
                continue
            x, y = rng.randint(0, width - side), rng.randint(0, height - side)
            crop = [x, y, x + side, y + side]
            if any(area(intersection(crop, b)) > 0 for b in blocked):
                continue
            bg_count += save(im, crop, r, 0, {"suffix": f"b{attempt:03d}",
                "negative_kind": "background", "target_index": None, "bbox_xyxy_96": None,
                "visible_ratio": None, "retained_ratio": None})
            if bg_count >= target_bg:
                break
        if bg_count < target_bg:
            skipped["sources_background_shortfall"] += 1
        if (i + 1) % 250 == 0:
            print(f"Prepared {i + 1}/{len(records)} sources; {len(pixel_hashes)} patches", flush=True)
    sources_file.close()
    stats = {"dataset": "CrowdHuman official validation subset; custom research split",
             "grouping": grouping, "source_box_count": sum(len(r["gtboxes"]) for r in records),
             "eligible_source_people": sum(b["derived"]["eligible"] for r in records for b in r["gtboxes"]),
             "ignore_reason_counts_nonexclusive": dict(reason_counts), "skipped": dict(skipped), "splits": {}}
    for split, d in data.items():
        manifests[split].close()
        x = np.stack(d["x"]) if d["x"] else np.empty((0, 1, n, n), dtype=np.uint8)
        y = np.array(d["y"], dtype=np.int64)
        np.savez_compressed(out / f"{split}.npz", x=x, y=y, sample_id=np.array(d["ids"]))
        kinds = Counter(json.loads(line).get("negative_kind") for line in (out / f"{split}.jsonl").open())
        stats["splits"][split] = {"total": len(y), "person": int(y.sum()), "non_person": int((y == 0).sum()),
            "background": kinds["background"], "partial_person": kinds["partial_person"],
            "shape": list(x.shape), "majority_accuracy": float(max(y.sum(), (y == 0).sum()) / len(y))}
    preview_dir = out / "preview"
    preview_dir.mkdir()
    for (split, label), patches in preview.items():
        sheet = Image.new("RGB", (8 * 112, 6 * 120), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (patch, sample_id) in enumerate(patches):
            x, y = (i % 8) * 112, (i // 8) * 120
            sheet.paste(patch, (x, y))
            draw.text((x, y + 97), f"{i}: y={label}", fill="black")
        sheet.save(preview_dir / f"{split}_{label}.jpg")
        dump(preview_dir / f"{split}_{label}_index.json", [sid for _, sid in patches])
    dump(out / "stats.json", stats)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
