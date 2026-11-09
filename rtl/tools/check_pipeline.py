#!/usr/bin/env python3
"""Independent fixed-point decode and image verification of RTL stream dumps."""
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / 'rtl/generated'
OUT = ROOT / 'rtl/results'


def decode(raw, lut, threshold=21299, iou=32768):
    p = [[lut[c*256+raw[c*576+i]] for i in range(576)] for c in range(6)]
    candidates = []
    for i in range(576):
        y, x = divmod(i, 24)
        if any(p[0][yy*24+xx] > p[0][i]
               for yy in range(max(0, y-1), min(24, y+2))
               for xx in range(max(0, x-1), min(24, x+2))):
            continue
        score = (p[0][i]*p[5][i]) >> 16
        if score < threshold:
            continue
        cx, cy = x*262144+p[1][i]*8-131072, y*262144+p[2][i]*8-131072
        w, h = p[3][i]*96, p[4][i]*96
        box = tuple(max(0, min(96*65536, v)) >> 8
                    for v in (cx-w//2, cy-h//2, cx+w//2, cy+h//2))
        candidates.append((score, i, box))
    candidates.sort(key=lambda r: (-r[0], r[1]))
    kept = []
    for score, i, b in candidates[:50]:
        x1, y1, x2, y2 = b
        if x2-x1 < 8*256 or y2-y1 < 24*256:
            continue
        area = (x2-x1)*(y2-y1)
        suppress = False
        for _, _, k in kept:
            inter = max(0, min(x2,k[2])-max(x1,k[0]))*max(0,min(y2,k[3])-max(y1,k[1]))
            union = area+(k[2]-k[0])*(k[3]-k[1])-inter
            if inter*65536 > union*iou:
                suppress = True
                break
        if not suppress:
            kept.append((score,i,b))
    return [(b[0]//256,b[1]//256,(b[2]+255)//256-1,(b[3]+255)//256-1,score)
            for score, _, b in kept]


def main():
    raw = [int(v,16) for v in (GEN/'golden_output.hex').read_text().split()]
    lut = [int(v,16) for v in (GEN/'post_lut.hex').read_text().split()]
    expected = decode(raw,lut)
    (GEN/'golden_boxes.txt').write_text(''.join(' '.join(map(str,b))+'\n' for b in expected))
    print('Integer post reference:',expected)
    if not (OUT/'rtl_boxes.txt').exists():
        return
    actual = [tuple(map(int,line.split())) for line in (OUT/'rtl_boxes.txt').read_text().splitlines()]
    if expected != actual:
        raise AssertionError(f'RTL bbox mismatch: {actual} != {expected}')
    image = Image.open(OUT/'rtl_overlay.ppm').convert('RGB')
    pixels = [int(v,16) for v in (GEN/'input_pixels.hex').read_text().split()]
    changed = 0
    for y in range(96):
        for x in range(96):
            hit = any(x1<=x<=x2 and y1<=y<=y2 and (x in (x1,x2) or y in (y1,y2))
                      for x1,y1,x2,y2,_ in expected)
            gray = pixels[y*96+x]
            value = (255,0,0) if hit else (gray,gray,gray)
            if image.getpixel((x,y)) != value:
                raise AssertionError(f'pixel mismatch ({x},{y}): {image.getpixel((x,y))} != {value}')
            changed += int(hit)
    image.save(OUT/'rtl_overlay.png')
    Image.frombytes('L',(96,96),bytes(pixels)).save(OUT/'rtl_input96.png')
    zoom = image.resize((576,576),Image.Resampling.NEAREST)
    zoom.save(OUT/'rtl_overlay_6x.png')
    manifest = json.loads((GEN/'manifest.json').read_text())
    report = dict(image=manifest['image'], npu_output_bytes_checked=3456,
                  boxes_xyxy_score_q16=actual, red_border_pixels=changed,
                  output_pixels_checked=9216, threshold_q16=21299,
                  npu_sram_bytes=manifest['npu_sram_bytes'],
                  post_sram_bytes=9984,
                  total_sram_bytes=manifest['npu_sram_bytes']+9984,
                  note='Dual-clock RTL pixel27MHz/NPU100MHz; actual inference/decode/NMS/overlay, frozen parameters; following bypass frame. INT8 dataset accuracy not evaluated.')
    report.update(json.loads((OUT/'cycles.json').read_text()))
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS: {len(actual)} boxes and all 9216 output pixels; {changed} red border pixels')


if __name__ == '__main__':
    main()
