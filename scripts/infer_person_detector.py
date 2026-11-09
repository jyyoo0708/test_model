#!/usr/bin/env python3
"""Full image -> person presence, original-image xyxy boxes, annotated image."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"models/person_detector"))
from utils import decode,to_original


def prepare(image):
    width,height=image.size;scale=96/max(width,height)
    nw,nh=max(1,round(width*scale)),max(1,round(height*scale))
    left,top=(96-nw)//2,(96-nh)//2
    canvas=Image.new('L',(96,96),114)
    canvas.paste(image.convert('L').resize((nw,nh),Image.Resampling.LANCZOS),(left,top))
    tensor=torch.from_numpy(np.asarray(canvas,dtype=np.float32).copy()).view(1,1,96,96)/255
    row={'original_width':width,'original_height':height,
         'transform':{'scale_x':nw/width,'scale_y':nh/height,'pad_left':left,'pad_top':top}}
    return tensor,row


def draw_boxes(image,boxes,color=(255,50,30),prefix='person'):
    image=image.copy().convert('RGB');draw=ImageDraw.Draw(image)
    line=max(2,round(max(image.size)/350))
    font=ImageFont.load_default(size=max(12,min(24,round(max(image.size)/60))))
    for box in boxes:
        x1,y1,x2,y2=map(float,box[:4])
        if x2<=x1 or y2<=y1:continue
        draw.rectangle((x1,y1,x2,y2),outline=color,width=line)
        label=prefix+(f" {float(box[4]):.2f}" if len(box)>4 else '')
        draw.text((max(0,x1),max(0,y1-18)),label,fill=color,font=font,stroke_width=1,stroke_fill='black')
    return image


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('image',type=Path)
    ap.add_argument('--model',type=Path,default=ROOT/'models/person_detector/run02/model_float32.ts')
    ap.add_argument('--output',type=Path)
    ap.add_argument('--threshold',type=float)
    args=ap.parse_args();torch.set_num_threads(1)
    threshold=args.threshold
    if threshold is None:threshold=json.loads((args.model.parent/'results.json').read_text())['threshold']
    config=json.loads((args.model.parent/'config.json').read_text())
    if not 0<=threshold<=1:raise ValueError('threshold must be in [0,1]')
    output=args.output or ROOT/'output/person_detector'/(args.image.stem+'_bbox.jpg')
    if output.resolve()==args.image.resolve():raise ValueError('output must differ from input')
    with Image.open(args.image) as source:image=ImageOps.exif_transpose(source).convert('RGB')
    x,row=prepare(image)
    model=torch.jit.load(str(args.model),map_location='cpu').eval()
    with torch.inference_mode():pred=decode(model(x),confidence=threshold,min_width=config.get('min_pred_width',8),min_height=config.get('min_pred_height',24))[0]
    original=to_original(pred,row)
    good=(original[:,2]>original[:,0])&(original[:,3]>original[:,1]);original=original[good];pred=pred[good]
    result={'image':str(args.image),'width':image.width,'height':image.height,'coordinate_format':'xyxy; original oriented-image pixels; right/bottom exclusive',
            'threshold':threshold,'person_present':bool(len(original)),'count':len(original),
            'detections':[{'bbox_xyxy':[float(v) for v in b[:4]],'score':float(b[4]),'bbox_xyxy_96':[float(v) for v in p[:4]]} for b,p in zip(original,pred)]}
    output.parent.mkdir(parents=True,exist_ok=True)
    draw_boxes(image,original).save(output)
    output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'annotated_image':str(output)},indent=2))


if __name__=='__main__':main()
