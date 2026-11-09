#!/usr/bin/env python3
"""Audit accumulator ranges and compare selected-image float/INT8 detections."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'models'))
from advanced_model import AdvancedDetector
from advanced_utils import decode
from export_int8 import prepare

def readhex(path):
    a=np.array([int(x,16) for x in path.read_text().split()],dtype=np.int64)
    return np.where(a >= 1<<31,a-(1<<32),a)

def main():
    generated=ROOT/'rtl/generated'
    m=json.loads((generated/'manifest.json').read_text())
    bias=readhex(generated/'bias.hex')
    mult=readhex(generated/'requant.hex')
    bounds=[]
    for layer in m['layers']:
        d=layer['descriptor']
        fanin=(d[7]+d[8])*d[12]**2
        b=bias[d[4]:d[4]+d[11]]
        k=mult[d[4]:d[4]+d[11]]
        bound=np.abs(b)+fanin*128*127
        rq_bound=bound*np.abs(k)+(1<<23)
        assert np.max(bound) < 1<<31 and np.max(rq_bound)<(1<<63)-1
        bounds.append(dict(name=layer['name'],fanin=fanin,
                           conservative_abs_accumulator=int(np.max(bound)),
                           conservative_abs_requant_product=int(np.max(rq_bound)),
                           signed_int32_safe=True,signed_int64_safe=True))
    torch.set_num_threads(4)
    model=AdvancedDetector().eval().fused()
    model.load_state_dict(torch.load(ROOT/m['checkpoint'],map_location='cpu',weights_only=False)['state_dict'])
    pixels=prepare(ROOT/m['image'])
    with torch.inference_mode():
        fp=model(torch.from_numpy(pixels.copy())[None,None].float()/255)
    q=np.load(generated/'golden_output.npy')
    dequant=q.astype(np.float64)*np.array(m['head_scales'])[:,None,None]
    qout=torch.from_numpy(dequant[None].astype(np.float32))
    cfg=json.loads((ROOT/'inference/config.json').read_text())
    kwargs=dict(confidence=cfg['threshold'],nms_iou=cfg['nms_iou'],topk=cfg['topk'],
                min_width=cfg['min_width'],min_height=cfg['min_height'])
    trainpath=ROOT/'data/processed/advanced_detection/train.npz'
    with np.load(trainpath) as train:
        indices=np.linspace(0,len(train['x'])-1,32,dtype=int).tolist()
    report=dict(image=m['image'],threshold=cfg['threshold'],
                calibration=dict(selected_image=True,image_directory_files=sorted(p.name for p in (ROOT/'image').glob('*') if p.is_file()),
                                 train_archive=str(trainpath.relative_to(ROOT)),
                                 train_archive_sha256=hashlib.sha256(trainpath.read_bytes()).hexdigest(),
                                 train_indices=indices,observed_images=m['calibration_images'],
                                 rule='Maximum across image-wise abs(conv_output) percentile99.99; concat source scales tied to larger value'),
                float_boxes_xyxy_score=decode(fp,**kwargs)[0].tolist(),
                int8_boxes_xyxy_score=decode(qout,**kwargs)[0].tolist(),
                logit_rmse=float(np.sqrt(np.mean((fp.numpy()[0]-dequant)**2))),
                output_saturation_count=int(np.sum((q==-128)|(q==127))),
                accumulator_bounds=bounds,
                interpretation='INT8 quality tested on selected image only; dataset AP50/accuracy after PTQ not evaluated.')
    (generated/'quantization_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['float_boxes_xyxy_score','int8_boxes_xyxy_score','logit_rmse']}))

if __name__=='__main__':main()
