#!/usr/bin/env python3
"""Save exact 96x96 grayscale inputs, infer them, and draw boxes at 96x96."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'models/person_detector'))
from utils import decode
from infer_person_detector import prepare


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',type=Path,default=ROOT/'test_image')
    ap.add_argument('--model',type=Path,default=ROOT/'models/person_detector/run02/model_float32.ts')
    args=ap.parse_args()
    extensions={'.jpg','.jpeg','.jfif','.png','.bmp','.webp','.tif','.tiff'}
    photos=sorted(p for p in args.directory.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if not photos:raise ValueError('No input images found')
    if len({p.stem for p in photos})!=len(photos):raise ValueError('Image stems must be unique')
    inputs_dir=args.directory/'input_96';results_dir=args.directory/'result_96'
    for p in (inputs_dir,results_dir):
        if p.exists():raise FileExistsError(f'Output directory already exists: {p}')
    inputs_dir.mkdir();results_dir.mkdir()
    cfg=json.loads((args.model.parent/'config.json').read_text())
    threshold=json.loads((args.model.parent/'results.json').read_text())['threshold']
    torch.set_num_threads(1)
    model=torch.jit.load(str(args.model),map_location='cpu').eval()
    records=[]
    for photo in photos:
        with Image.open(photo) as source:
            image=ImageOps.exif_transpose(source).convert('RGB')
        prepared,transform=prepare(image)
        pixels=np.rint(prepared.numpy()[0,0]*255).astype(np.uint8)
        input_file=inputs_dir/(photo.stem+'.png');result_file=results_dir/(photo.stem+'.png')
        Image.fromarray(pixels).save(input_file)
        with Image.open(input_file) as saved:
            assert saved.mode=='L' and saved.size==(96,96)
            pixels=np.array(saved,dtype=np.uint8)
        # The network reads these saved pixels directly, with float32 /255 normalization.
        tensor=torch.from_numpy(pixels.astype(np.float32)).view(1,1,96,96)/255.
        assert torch.equal(tensor,prepared)
        with torch.inference_mode():
            boxes=decode(model(tensor),confidence=threshold,
                min_width=cfg.get('min_pred_width',8),min_height=cfg.get('min_pred_height',24))[0]
        annotated=Image.fromarray(pixels).convert('RGB');draw=ImageDraw.Draw(annotated)
        for x1,y1,x2,y2,score in boxes:
            assert 0<=x1<x2<=96 and 0<=y1<y2<=96
            draw.rectangle([max(0,int(np.floor(x1))),max(0,int(np.floor(y1))),
                            min(95,int(np.ceil(x2))-1),min(95,int(np.ceil(y2))-1)],outline=(255,0,0),width=1)
        annotated.save(result_file)
        with Image.open(result_file) as output:
            assert output.size==(96,96)
            rgb=np.array(output);expected=np.repeat(pixels[:,:,None],3,2)
            changed=(rgb!=expected).any(2)
            assert np.all(rgb[changed]==np.array([255,0,0]))
        record={'source':str(photo),'source_oriented_size':[image.width,image.height],
                'input':str(input_file),'result':str(result_file),'input_size':[96,96],
                'input_mode':'L','result_size':[96,96],'model':str(args.model),
                'threshold':threshold,'coordinate_format':'xyxy on 96x96 input; right/bottom exclusive',
                'letterbox_transform':transform['transform'],
                'person_present':bool(len(boxes)),'count':len(boxes),
                'detections':[{'bbox_xyxy':b[:4].tolist(),'score':float(b[4])} for b in boxes]}
        result_file.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
        records.append(record)
    (args.directory/'inference_summary.json').write_text(json.dumps({'validation':'PASS','images':len(records),
        'actual_model_input':'saved input_96 PNG pixels, float32 /255','results':records},indent=2)+'\n')
    print(json.dumps([{'source':Path(r['source']).name,'count':r['count'],'input':r['input'],'result':r['result']} for r in records],indent=2))


if __name__=='__main__':main()
