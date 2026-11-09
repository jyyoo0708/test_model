"""Add MOT17 surveillance images with entire-sequence splits and audited eligibility."""
import json,hashlib,zipfile
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/processed/advanced_detection'

def make_view(image,annotations,seq,split,frame,crop,label):
    W,H=image.size;cx,cy,cw,ch=crop;nw=round(cw*96/max(cw,ch));nh=round(ch*96/max(cw,ch))
    left,top=(96-nw)//2,(96-nh)//2;sx,sy=nw/cw,nh/ch
    canvas=Image.new('L',(96,96),114)
    canvas.paste(image.crop((cx,cy,cx+cw,cy+ch)).convert('L').resize((nw,nh),Image.Resampling.LANCZOS),(left,top))
    r=dict(source_id=f'{seq}_{frame:06}_{label}',group_id=f'mot17_scene_{seq[6:8]}',split=split,dataset='mot17',
           sequence=seq,frame=frame,source_crop_xywh=list(crop),original_width=cw,original_height=ch,
           transform=dict(scale_x=sx,scale_y=sy,pad_left=left,pad_top=top,resized_width=nw,resized_height=nh),
           boxes=[],ignore_boxes=[],negative_people=[])
    for a in annotations:
        _,identity,x,y,w,h,conf,cls,visibility=a[:9];x-=1+cx;y-=1+cy
        clipped=[max(0,x),max(0,y),min(cw,x+w),min(ch,y+h)]
        if clipped[2]<=clipped[0] or clipped[3]<=clipped[1]:continue
        retained=(clipped[2]-clipped[0])*(clipped[3]-clipped[1])/max(w*h,1)
        # Conservative lower bound: cropping can remove as much visible area as total removed area.
        effective_visible=max(0,float(visibility)-(1-retained))
        box=[clipped[0]*sx+left,clipped[1]*sy+top,clipped[2]*sx+left,clipped[3]*sy+top]
        why=[]
        if conf!=1 or cls!=1:why.append('source_ignore')
        if w<12 or h<48:why.append('small_source')
        if box[2]-box[0]<8 or box[3]-box[1]<24:why.append('small_input')
        if effective_visible<.8:why.append('visibility_below_80')
        if retained<.8:why.append('truncated_below_80')
        item=dict(bbox_xyxy=box,visible_ratio=effective_visible,retained_ratio=retained,reasons=why,
                  source_visibility=float(visibility),source_fbox_xywh=[x,y,w,h],source_index=int(identity))
        key='ignore_boxes' if any(k in why for k in ('source_ignore','small_source','small_input')) else 'negative_people' if why else 'boxes'
        r[key].append(item)
    return np.asarray(canvas)[None],r

def main():
    OUT.mkdir(exist_ok=False);original=ROOT/'data/processed/person96_detection';archive=ROOT/'data/raw/MOT17.zip'
    if not archive.exists():
        candidate=archive.with_suffix('.zip.part')
        with zipfile.ZipFile(candidate) as z:assert z.namelist()
        candidate.rename(archive)
    mapping={'MOT17-02-FRCNN':'train','MOT17-04-FRCNN':'train','MOT17-09-FRCNN':'val','MOT17-10-FRCNN':'test'}
    xs={};rs={};counts={}
    for split in ('train','val','test'):
        with np.load(original/f'{split}.npz') as p:xs[split]=list(p['x'])
        rs[split]=[json.loads(l) for l in (original/f'{split}.jsonl').open()]
        for r in rs[split]:r['dataset']='crowdhuman'
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        for seq,split in mapping.items():
            base=next(n[:-len('gt/gt.txt')] for n in names if n.endswith(seq+'/gt/gt.txt'))
            gt=np.loadtxt(z.open(base+'gt/gt.txt'),delimiter=',')
            frames=sorted(n for n in names if n.startswith(base+'img1/') and n.endswith('.jpg'));counts[seq]=Counter()
            for name in frames[::30]:
                frame=int(Path(name).stem)
                with Image.open(z.open(name)) as source:
                    image=source.convert('L');W,H=image.size;side=round(min(W,H)*.6)
                    crops=[(0,0,W,H)]+[(int(x),int(y),side,side) for y in np.linspace(0,H-side,2) for x in np.linspace(0,W-side,3)]
                    for j,crop in enumerate(crops):
                        arr,r=make_view(image,gt[gt[:,0]==frame],seq,split,frame,crop,f'view{j}')
                        xs[split].append(arr);rs[split].append(r)
                        counts[seq].update(images=1,people=len(r['boxes']),positive_images=int(bool(r['boxes'])))
            print(seq,dict(counts[seq]),flush=True)
    for split in xs:
        np.savez_compressed(OUT/f'{split}.npz',x=np.stack(xs[split]),source_id=[r['source_id'] for r in rs[split]])
        (OUT/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rs[split]))
    groups={s:{r['group_id'] for r in rs[s]} for s in rs}
    for a,b in (('train','val'),('train','test'),('val','test')):assert not groups[a]&groups[b]
    config=json.loads((original/'config.json').read_text())
    config.update(extra_source='https://motchallenge.net/data/MOT17/',extra_sequences=mapping,frame_step=30,
                  extra_views='full-frame + six fixed grid tiles of side .6*min(W,H); views grouped by source video',
                  extra_visibility='max(0,official_visibility - (1-retained)) >= .8; conservative visible-area lower bound',
                  archive_sha256=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest())
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    stats=dict(mot17={k:dict(v) for k,v in counts.items()},splits={s:dict(images=len(rs[s]),people=sum(len(r['boxes']) for r in rs[s])) for s in rs})
    (OUT/'stats.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2),flush=True)
if __name__=='__main__':main()
