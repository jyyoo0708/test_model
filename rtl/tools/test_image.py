#!/usr/bin/env python3
"""Run an image through the existing RTL TB using frozen quantized parameters."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from export_int8 import ROOT, NAMES, SOURCES, prepare, integer_conv, hexfile
import check_pipeline


def read_hex(path, bits):
    a = np.array([int(x, 16) for x in path.read_text().split()], dtype=np.int64)
    return np.where(a >= (1 << (bits-1)), a-(1 << bits), a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--phase', type=int, default=17321, help='pixel clock phase in units1/27ps')
    args = parser.parse_args()
    image = args.image.resolve()
    source = ROOT/'rtl/generated'
    case = ROOT/'rtl/build/cases'/image.stem
    gen = case/'rtl/generated'
    out = case/'rtl/results'
    gen.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    (case/'rtl/build').mkdir(parents=True, exist_ok=True)
    for filename in ('weights.hex', 'bias.hex', 'requant.hex', 'descriptors.hex', 'post_lut.hex'):
        shutil.copy2(source/filename, gen/filename)
    manifest = json.loads((source/'manifest.json').read_text())
    manifest['image'] = str(image.relative_to(ROOT))
    manifest['parameter_source'] = 'rtl/generated (frozen; no recalibration)'
    pixels = prepare(image)
    acts = {'input': ((pixels.astype(np.int32)*127+127)//255).astype(np.int8)[None]}
    weights = read_hex(gen/'weights.hex', 8)
    biases = read_hex(gen/'bias.hex', 32)
    multipliers = read_hex(gen/'requant.hex', 32)
    for name, layer, (src, skip) in zip(NAMES, manifest['layers'], SOURCES):
        d = layer['descriptor']
        x = acts[src]
        if skip:
            x = np.concatenate([np.repeat(np.repeat(x, 2, 1), 2, 2), acts[skip]], axis=0)
        count = d[11]*(d[7]+d[8])*d[12]*d[12]
        w = weights[d[3]:d[3]+count].reshape(d[11],d[7]+d[8],d[12],d[12])
        b = biases[d[4]:d[4]+d[11]]
        mult = multipliers[d[4]:d[4]+d[11]]
        acc = integer_conv(x,w,b,d[13],d[12]//2)
        assert np.max(np.abs(acc)) < 1 << 31
        acts[name] = np.clip((acc*mult[:,None,None]+(1<<23))>>24,
                             0 if d[15] else -128,127).astype(np.int8)
        layer['max_accumulator'] = int(np.max(np.abs(acc)))
    output = np.concatenate([acts[n] for n in NAMES[9:]],axis=0)
    hexfile(gen/'input_pixels.hex', pixels,8)
    hexfile(gen/'input_q.hex', acts['input'],8)
    hexfile(gen/'golden_output.hex',output,8)
    (gen/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    executable = ROOT/'rtl/build/pipeline/test'
    if not executable.exists():
        raise SystemExit('Build pipeline first: python3 rtl/tools/run_tests.py')
    env = os.environ.copy()
    env.pop('LD_LIBRARY_PATH',None)
    command = [str(executable),'--phase',str(args.phase)]+(['--trace'] if args.trace else [])
    print(f'Running RTL TB on {image.name}; frozen parameters',flush=True)
    result = subprocess.run(command,cwd=case,env=env,capture_output=True,text=True)
    (out/'simulation.log').write_text(result.stdout+result.stderr)
    print(result.stdout+result.stderr,flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    check_pipeline.GEN,check_pipeline.OUT = gen,out
    check_pipeline.main()
    destination = ROOT/'rtl/results'/image.stem
    destination.mkdir(parents=True,exist_ok=True)
    for file in out.iterdir():
        if file.is_file():
            shutil.copy2(file,destination/file.name)
    print(f'Results: {destination}',flush=True)


if __name__ == '__main__':
    main()
