#!/usr/bin/env python3
"""Independent scalar fixed-point reference for bbox_overlay.v.

All inputs are plain Python integers. The RTL stream is channel-major raw
unsigned bytes, with two's-complement logits interpreted by the exported LUT.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def postprocess(raw, lut, threshold=21299, nms_iou=32768,
                min_width=8, min_height=24, nms=True, topk=50):
    if len(raw) != 3456 or len(lut) != 1536:
        raise ValueError('expected 3456 output bytes and 1536 LUT entries')
    prob = [[lut[c * 256 + (int(raw[c * 576 + k]) & 255)]
             for k in range(576)] for c in range(6)]
    candidates = []
    for k in range(576):
        x, y = k % 24, k // 24
        heat = prob[0][k]
        if any(prob[0][ny * 24 + nx] > heat
               for ny in range(max(0, y - 1), min(24, y + 2))
               for nx in range(max(0, x - 1), min(24, x + 2))):
            continue
        score = heat * prob[5][k] >> 16
        if score < threshold:
            continue
        cx = x * 262144 + prob[1][k] * 8 - 131072
        cy = y * 262144 + prob[2][k] * 8 - 131072
        w, h = prob[3][k] * 96, prob[4][k] * 96
        q8 = [max(0, min(6291456, v)) >> 8
              for v in (cx - w // 2, cy - h // 2,
                        cx + w // 2, cy + h // 2)]
        candidates.append(dict(cell=k, score=score, q8=q8))
    candidates.sort(key=lambda c: (-c['score'], c['cell']))
    kept = []
    for candidate in candidates[:topk]:
        a = candidate['q8']
        if a[2] <= a[0] or a[3] <= a[1]:
            continue
        if a[2] - a[0] < min_width * 256 or a[3] - a[1] < min_height * 256:
            continue
        suppressed = False
        if nms:
            for previous in kept:
                b = previous['q8']
                inter = (max(0, min(a[2], b[2]) - max(a[0], b[0])) *
                         max(0, min(a[3], b[3]) - max(a[1], b[1])))
                union = ((a[2] - a[0]) * (a[3] - a[1]) +
                         (b[2] - b[0]) * (b[3] - b[1]) - inter)
                if inter * 65536 > union * nms_iou:
                    suppressed = True
                    break
        if not suppressed:
            candidate['xyxy'] = [a[0] // 256, a[1] // 256,
                                 (a[2] + 255) // 256 - 1,
                                 (a[3] + 255) // 256 - 1]
            kept.append(candidate)
    return kept


def sensor_boxes(boxes, width=96, height=96):
    """Invert the capture block's centered letterbox transform."""
    longest = max(width, height)
    rw = max(1, (width * 96 + longest // 2) // longest)
    rh = max(1, (height * 96 + longest // 2) // longest)
    px, py = (96 - rw) // 2, (96 - rh) // 2
    result = []
    for b in boxes:
        a = b['q8']
        # Truncation toward zero matches Verilog signed division. Negative
        # coordinates are subsequently clipped to zero.
        x1 = max(0, min(width - 1, int((a[0] - px * 256) * width / (rw * 256))))
        y1 = max(0, min(height - 1, int((a[1] - py * 256) * height / (rh * 256))))
        x2 = max(0, min(width - 1, int(((a[2] - px * 256) * width + rw * 256 - 1) / (rw * 256)) - 1))
        y2 = max(0, min(height - 1, int(((a[3] - py * 256) * height + rh * 256 - 1) / (rh * 256)) - 1))
        result.append([x1, y1, x2, y2])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_hex', type=Path)
    parser.add_argument('lut_hex', type=Path)
    parser.add_argument('--threshold', type=int, default=21299)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    raw = [int(v, 16) for v in args.raw_hex.read_text().split()]
    lut = [int(v, 16) for v in args.lut_hex.read_text().split()]
    encoded = json.dumps(postprocess(raw, lut, threshold=args.threshold), indent=2) + '\n'
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
