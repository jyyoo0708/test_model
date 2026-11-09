"""Save original/augmented labels for local visual review."""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'models'))
from advanced_utils import load_split,augment

def main():
    torch.set_num_threads(4);x,rows=load_split(ROOT/'data/processed/advanced_detection','train')
    native=[i for i,r in enumerate(rows) if r['dataset']=='crowdhuman' and r['boxes']][:6]
    mot=[i for i,r in enumerate(rows) if r['dataset']=='mot17' and r['boxes']][:6];ids=native+mot
    original=x[ids];rr=[rows[i] for i in ids];aug,ar=augment(original,rr,np.random.default_rng(33))
    sheet=Image.new('RGB',(12*192,2*216),'white');draw=ImageDraw.Draw(sheet)
    for rownum,(xx,rs) in enumerate(((original,rr),((aug*255).byte(),ar))):
        for j,(a,r) in enumerate(zip(xx,rs)):
            im=Image.fromarray(a[0].numpy()).convert('RGB');d=ImageDraw.Draw(im)
            for key,color in (('ignore_boxes','yellow'),('negative_people','red'),('boxes','lime')):
                for b in r[key]:
                    q=np.clip(b['bbox_xyxy'],0,95);d.rectangle(tuple(q),outline=color,width=1)
            sheet.paste(im.resize((192,192),Image.Resampling.NEAREST),(j*192,rownum*216))
            draw.text((j*192+2,rownum*216+194),('original' if rownum==0 else 'augmented')+f' GT={len(r["boxes"])}',fill='black')
    (ROOT/'report').mkdir(exist_ok=True);sheet.save(ROOT/'report/augmentation_preview.jpg')
if __name__=='__main__':main()
