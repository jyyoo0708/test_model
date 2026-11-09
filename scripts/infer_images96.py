#!/usr/bin/env python3
"""Infer image/ at 96x96 grayscale and save grayscale bbox images in result/."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ensure_runtime():
    """Use the local environment, including when invoked with system python."""
    environment = ROOT / '.venv'
    executable = environment / 'bin/python'
    if not executable.is_file():
        raise SystemExit(f'실행 환경이 없습니다: {environment}')
    if Path(sys.prefix).resolve() != environment.resolve() or 'LD_LIBRARY_PATH' in os.environ:
        env = os.environ.copy()
        env.pop('LD_LIBRARY_PATH', None)
        env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT / 'tmp'),
                   XDG_CACHE_HOME=str(ROOT / '.cache'), TORCH_HOME=str(ROOT / '.cache/torch'))
        (ROOT / 'tmp').mkdir(exist_ok=True)
        os.execve(str(executable), [str(executable), str(Path(__file__).resolve()), *sys.argv[1:]], env)


if __name__ == '__main__':
    ensure_runtime()

import argparse
import json
import math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(ROOT / 'models'))
from advanced_utils import decode


def input_image(path, padding):
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source).convert('L')
        width, height = source.size
        scale = 96 / max(width, height)
        nw, nh = max(1, round(width * scale)), max(1, round(height * scale))
        left, top = (96 - nw) // 2, (96 - nh) // 2
        canvas = Image.new('L', (96, 96), padding)
        canvas.paste(source.resize((nw, nh), Image.Resampling.LANCZOS), (left, top))
    return canvas


def main():
    parser = argparse.ArgumentParser(description='image/ 이미지를 96x96 gray로 추론하고 result/에 박스 결과를 저장합니다.')
    parser.add_argument('--input', type=Path, default=ROOT / 'image', help='입력 이미지 또는 폴더 (기본: image/)')
    parser.add_argument('--output', type=Path, default=ROOT / 'result', help='결과 폴더 (기본: result/)')
    parser.add_argument('--threshold', type=float, help='검출 점수 기준 0~1 (생략 시 모델 설정값)')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'inference/config.json').read_text())
    threshold = cfg['threshold'] if args.threshold is None else args.threshold
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        parser.error('--threshold는 0~1 사이의 유한한 값이어야 합니다.')
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(ROOT)
    except ValueError:
        parser.error('결과는 advance_network 안에 저장해야 합니다.')
    if not source.exists():
        parser.error(f'입력 경로가 없습니다: {source}')
    if output == source or source in output.parents or (source.is_file() and output == source.parent):
        parser.error('입력과 결과 폴더를 분리하세요.')
    extensions = {'.jpg', '.jpeg', '.jfif', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    files = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in extensions) if source.is_dir() else [source]
    if not files:
        parser.error('입력 폴더에 이미지가 없습니다.')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    model = torch.jit.load(str(ROOT / cfg['model']), map_location='cpu').eval()
    results, errors = [], []
    for path in files:
        relative = path.relative_to(source) if source.is_dir() else Path(path.name)
        # Preserve extensions so a.jpg and a.png have distinct outputs.
        basename = output / relative
        basename.parent.mkdir(parents=True, exist_ok=True)
        try:
            canvas = input_image(path, cfg['padding'])
            pixels = np.array(canvas, dtype=np.uint8)
            inputs = torch.from_numpy(pixels.copy())[None, None].float() / 255
            with torch.inference_mode():
                raw = model(inputs)
                if tuple(raw.shape) != (1, 6, 24, 24):
                    raise ValueError(f'예상하지 못한 모델 출력: {tuple(raw.shape)}')
                boxes = decode(raw, confidence=threshold, nms_iou=cfg['nms_iou'],
                               topk=cfg['topk'], min_width=cfg['min_width'], min_height=cfg['min_height'])[0]
            input_path = Path(str(basename) + '_input96.png')
            result_path = Path(str(basename) + '_bbox96.png')
            canvas.save(input_path)
            # Keep actual mode L; only the grayscale box outlines change pixels.
            result = canvas.copy()
            draw = ImageDraw.Draw(result)
            for x1, y1, x2, y2, score in boxes:
                rect = (max(0, int(np.floor(x1))), max(0, int(np.floor(y1))),
                        min(95, int(np.ceil(x2)) - 1), min(95, int(np.ceil(y2)) - 1))
                draw.rectangle(rect, outline=0, width=2)
                draw.rectangle(rect, outline=255, width=1)
            result.save(result_path)
            record = dict(input=str(relative), input_image=str(input_path.relative_to(output)),
                          result_image=str(result_path.relative_to(output)), image_size=[96, 96], mode='L',
                          threshold=threshold, person_present=bool(len(boxes)), count=len(boxes),
                          boxes_xyxy_score_96=boxes.tolist())
            Path(str(basename) + '.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
            results.append(record)
            print(f'{relative}: bbox {len(boxes)}개 -> {result_path.name}')
        except Exception as error:
            errors.append(dict(input=str(relative), error=str(error)))
            print(f'{relative}: 실패: {error}', file=sys.stderr)
    summary = dict(model=cfg['model'], threshold=threshold, processed=len(results), failed=len(errors),
                   results=results, errors=errors)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(f'완료: {len(results)}장, 실패: {len(errors)}장, threshold={threshold:g}\n결과: {output}')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
