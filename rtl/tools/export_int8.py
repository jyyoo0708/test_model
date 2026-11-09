#!/usr/bin/env python3
"""Export the trained fused student and an integer reference; no RTL shortcuts."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'models'))
from advanced_model import AdvancedDetector

NAMES = ['c1', 'c2', 'c3', 'c4', 'context', 'fuse1', 'fuse2', 'head',
         'regression', 'heatmap', 'offset', 'size', 'quality']
SOURCES = [('input', None), ('c1', None), ('c2', None), ('c3', None),
           ('c4', None), ('context', 'c3'), ('fuse1', 'c2'),
           ('fuse2', None), ('head', None), ('head', None),
           ('regression', None), ('regression', None), ('regression', None)]

def prepare(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('L')
        s = 96 / max(im.size)
        w, h = [max(1, round(v * s)) for v in im.size]
        out = Image.new('L', (96, 96), 114)
        out.paste(im.resize((w, h), Image.Resampling.LANCZOS), ((96-w)//2, (96-h)//2))
    return np.array(out)

def hexfile(path, values, bits):
    values = np.asarray(values).reshape(-1)
    path.write_text(''.join(f'{int(x) & ((1 << bits)-1):0{bits//4}x}\n' for x in values))

def integer_conv(x, w, b, stride, padding):
    """INT64 NumPy accumulation independent of RTL scheduling/address generation."""
    x = np.pad(x.astype(np.int64), ((0, 0), (padding, padding), (padding, padding)))
    k = w.shape[-1]
    windows = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(1, 2))
    windows = windows[:, ::stride, ::stride]
    return np.einsum('cyxij,ocij->oyx', windows, w.astype(np.int64), optimize=True) + b[:, None, None]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=Path, default=ROOT/'image/frame_20261021_085042_877.jpg')
    ap.add_argument('--calibration', type=int, default=32)
    ap.add_argument('--out', type=Path, default=ROOT/'rtl/generated')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    ckpt = ROOT/'models/student_kd/model_float32.pt'
    state = torch.load(ckpt, map_location='cpu', weights_only=False)['state_dict']
    model = AdvancedDetector().eval()
    if 'c1.0.bias' in state:
        model = model.fused()
    model.load_state_dict(state)
    if 'c1.1.weight' in state:
        model = model.fused()
    convs = [getattr(model, n)[0] if n not in NAMES[9:] else getattr(model, n) for n in NAMES]
    maxima = dict.fromkeys(NAMES, 0.)
    shapes = {}
    hooks = []
    def capture(n):
        def hook(m, inputs, out):
            a = out.detach().numpy()
            maxima[n] = max(maxima[n], float(np.quantile(np.abs(a), .9999)))
            shapes[n] = list(a.shape[1:])
        return hook
    for n, c in zip(NAMES, convs):
        hooks.append(c.register_forward_hook(capture(n)))
    images = sorted((ROOT/'image').glob('*'))
    data = sorted((ROOT/'data/processed/advanced_detection/images/train').glob('*.png'))
    if data:
        step = max(1, len(data)//max(1, args.calibration))
        images += data[::step][:args.calibration]
    images = list(dict.fromkeys([args.image] + images))
    calibrated = 0
    with torch.inference_mode():
        for p in images:
            try:
                x = torch.from_numpy(prepare(p).copy())[None, None].float()/255
                model(x)
                calibrated += 1
            except (OSError, ValueError):
                continue
        train_archive = ROOT/'data/processed/advanced_detection/train.npz'
        if train_archive.exists() and args.calibration:
            with np.load(train_archive) as archive:
                train = archive['x']
                indices = np.linspace(0, len(train)-1, min(args.calibration,len(train)), dtype=int)
                for i in indices:
                    model(torch.from_numpy(train[i:i+1].copy()).float()/255)
                    calibrated += 1
    for h in hooks:
        h.remove()
    scales = {n: max(maxima[n]/127, 1e-8) for n in NAMES}
    for a, b in [('context', 'c3'), ('fuse1', 'c2')]:
        scales[a] = scales[b] = max(scales[a], scales[b])
    scales['input'] = 1/127
    bases = {'input': 0}
    cursor = 9216
    for n in NAMES:
        bases[n] = cursor
        cursor += int(np.prod(shapes[n]))
    weights, biases, multipliers, descs, layers = [], [], [], [], []
    pixels = prepare(args.image)
    acts = {'input': ((pixels.astype(np.int32)*127+127)//255).astype(np.int8)[None]}
    macs = 0
    for n, c, (src, skip) in zip(NAMES, convs, SOURCES):
        ins = scales[src]
        ws = np.maximum(np.max(np.abs(c.weight.detach().numpy()), axis=(1,2,3))/127, 1e-12)
        w = np.clip(np.rint(c.weight.detach().numpy()/ws[:,None,None,None]), -127,127).astype(np.int8)
        b = np.rint(c.bias.detach().numpy()/(ins*ws)).astype(np.int64)
        mult = np.rint(ins*ws/scales[n]*(1<<24)).astype(np.int64)
        assert np.max(np.abs(b)) < (1<<31) and np.min(mult)>0 and np.max(mult)<(1<<31)
        x = acts[src]
        up = int(skip is not None)
        inc1 = x.shape[0]
        if up:
            x = np.repeat(np.repeat(x, 2, 1), 2, 2)
            x = np.concatenate([x, acts[skip]], axis=0)
        co, ho, wo = shapes[n]
        descriptor = [bases[src], bases[skip] if skip else 0, bases[n], len(weights), len(biases),
                      x.shape[2], x.shape[1], inc1, acts[skip].shape[0] if skip else 0,
                      wo, ho, co, c.kernel_size[0], c.stride[0], up, int(n not in NAMES[9:])]
        descs.extend(descriptor)
        acc = integer_conv(x, w, b, c.stride[0], c.padding[0])
        assert np.max(np.abs(acc)) < 1<<31
        q = np.clip((acc*mult[:,None,None]+(1<<23))>>24, 0 if descriptor[15] else -128, 127).astype(np.int8)
        acts[n] = q
        layer_macs = int(q.size*np.prod(w.shape[1:]))
        macs += layer_macs
        layers.append(dict(name=n, shape=shapes[n], input_scale=ins, output_scale=scales[n],
                           weight_scales=ws.tolist(), descriptor=descriptor, macs=layer_macs,
                           max_accumulator=int(np.max(np.abs(acc))), saturation=int(np.sum(np.abs(q.astype(int))>=127))))
        weights.extend(w.reshape(-1).tolist())
        biases.extend(b.tolist())
        multipliers.extend(mult.tolist())
    output = np.concatenate([acts[n] for n in NAMES[9:]], axis=0)
    np.save(args.out/'golden_output.npy', output)
    np.save(args.out/'float_scales.npy', np.array([scales[n] for n in ['heatmap','offset','offset','size','size','quality']]))
    head_scales = [scales[n] for n in ['heatmap','offset','offset','size','size','quality']]
    lut = []
    for channel, scale in enumerate(head_scales):
        z = np.arange(256)
        z = np.where(z<128, z, z-256)*scale
        sig = 1/(1+np.exp(-z))
        if channel==5:
            sig=np.sqrt(sig)
        lut.extend(np.minimum(65535,np.floor(sig*65536+.5)).astype(int).tolist())
    for filename, values, bits in [('weights',weights,8),('bias',biases,32),('requant',multipliers,32),
                                  ('descriptors',descs,32),('input_pixels',pixels,8),('input_q',acts['input'],8),
                                  ('golden_output',output,8),('post_lut',lut,16)]:
        hexfile(args.out/(filename+'.hex'), values,bits)
    # SRAM accounting covers actual declared depths; postprocessor is added at integration.
    manifest = dict(checkpoint=str(ckpt.relative_to(ROOT)), checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                    image=str(args.image.resolve().relative_to(ROOT)), calibration_images=calibrated,
                    activation_depth=cursor, input_frame_depth=9216, input_frame_bytes=9216, clock_pixel_hz=27000000, clock_npu_hz=100000000, weight_depth=len(weights), parameter_depth=len(biases),
                    descriptor_depth=len(descs), macs=macs, input_scale=1/127,
                    bases=bases, head_scales=head_scales, layers=layers,
                    npu_sram_bytes=9216+cursor+len(weights)+8*len(biases)+4*len(descs),
                    requant='clip((acc*mult+2^23) >>>24)',lut_address='channel*256+uint8(logit)',
                    load_counts=[len(weights),len(biases),len(descs),len(multipliers)])
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['image','calibration_images','activation_depth','macs','npu_sram_bytes']}))

if __name__=='__main__':
    main()
