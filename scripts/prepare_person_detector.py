#!/usr/bin/env python3
"""Letterbox original frames and retain all qualified bounding boxes."""
import json
from pathlib import Path
from collections import Counter
import zipfile
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/person96_detection"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    sources = [json.loads(line) for line in (ROOT / "data/processed/person96/source_annotations.jsonl").open()]
    config = {"input_size": 96, "grid_size": 24, "stride": 4, "padding_value": 114,
              "visible_ratio_min": .8, "retained_ratio_min": .8,
              "min_source_height": 48, "min_source_width": 12,
              "min_input_height": 24, "min_input_width": 8,
              "split": "reuse original-source/dHash groups from person96, never crop-level split",
              "visibility_rule": "intersection(vbox,fbox,image) area / full fbox area",
              "ignore_policy": "source ignore/mask and tiny people ignored; known large visibility/truncation failures are negative supervision"}
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    arrays = {s: [] for s in ("train", "val", "test")}
    records = {s: [] for s in arrays}
    stats = {s: Counter() for s in arrays}
    reasons = Counter()
    with zipfile.ZipFile(ROOT / "data/raw/crowdhuman/CrowdHuman_val.zip") as archive:
        for i, source in enumerate(sources):
            width, height = source["width"], source["height"]
            scale = 96 / max(width, height)
            nw, nh = max(1, round(width * scale)), max(1, round(height * scale))
            left, top = (96 - nw) // 2, (96 - nh) // 2
            sx, sy = nw / width, nh / height
            with archive.open(source["zip_member"]) as stream, Image.open(stream) as image:
                assert image.size == (width, height)
                image = image.convert("L").resize((nw, nh), Image.Resampling.LANCZOS)
                canvas = Image.new("L", (96, 96), 114)
                canvas.paste(image, (left, top))
            accepted, ignored, negatives = [], [], []
            for index, box in enumerate(source["gtboxes"]):
                x, y, w, h = box["fbox"]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(width, x + w), min(height, y + h)
                if x2 <= x1 or y2 <= y1:
                    reasons["outside_or_invalid"] += 1
                    continue
                transformed = [x1 * sx + left, y1 * sy + top, x2 * sx + left, y2 * sy + top]
                vx, vy, vw, vh = box["vbox"]
                visible = max(0, min(x2, vx + vw) - max(x1, vx)) * max(0, min(y2, vy + vh) - max(y1, vy)) / max(w * h, 1)
                retained = (x2 - x1) * (y2 - y1) / max(w * h, 1)
                why = []
                if box.get("tag") != "person" or box.get("extra", {}).get("ignore", 0):
                    why.append("source_ignore")
                if h < 48 or w < 12:
                    why.append("small_source")
                if transformed[3] - transformed[1] < 24 or transformed[2] - transformed[0] < 8:
                    why.append("small_input")
                if visible < .8:
                    why.append("visibility_below_80")
                if retained < .8:
                    why.append("truncated_below_80")
                item = {"bbox_xyxy": transformed, "source_index": index,
                        "source_fbox_xywh": box["fbox"], "visible_ratio": visible,
                        "retained_ratio": retained, "reasons": why}
                reasons.update(why)
                if not why:
                    accepted.append(item)
                elif any(r in why for r in ("source_ignore", "small_source", "small_input")):
                    ignored.append(item)
                else:
                    negatives.append(item)
            split = source["split"]
            relative = Path("images") / split / (source["ID"] + ".png")
            (OUT / relative).parent.mkdir(parents=True, exist_ok=True)
            canvas.save(OUT / relative)
            row = {"source_id": source["ID"], "group_id": source["group_id"], "split": split,
                   "image": str(relative), "zip_member": source["zip_member"],
                   "original_width": width, "original_height": height,
                   "transform": {"scale_x": sx, "scale_y": sy, "pad_left": left, "pad_top": top,
                                 "resized_width": nw, "resized_height": nh},
                   "boxes": accepted, "ignore_boxes": ignored, "negative_people": negatives}
            records[split].append(row)
            arrays[split].append(np.array(canvas, dtype=np.uint8)[None])
            stats[split].update(images=1, people=len(accepted), positive_images=int(bool(accepted)),
                                negative_images=int(not accepted), ignored_boxes=len(ignored), negative_people=len(negatives))
            if (i + 1) % 500 == 0:
                print(f"Prepared full frames {i+1}/{len(sources)}", flush=True)
    for split in arrays:
        with (OUT / f"{split}.jsonl").open("w") as stream:
            for row in records[split]:
                stream.write(json.dumps(row) + "\n")
        np.savez_compressed(OUT / f"{split}.npz", x=np.stack(arrays[split]),
                            source_id=np.array([r["source_id"] for r in records[split]]))
    summary = {"splits": {s: dict(v) for s, v in stats.items()}, "exclusion_reasons_nonexclusive": dict(reasons)}
    (OUT / "stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
