import copy, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'baseline/person_detector'))
from utils import load_split,ious,detection_metrics,presence_metrics,to_original


def augment(x,rows,rng):
    """Uniform scale/translation; update visibility, retained area and minimum-size eligibility."""
    images=[];records=[]
    from PIL import Image,ImageFilter
    for a,row in zip(x.numpy(),rows):
        scale=float(rng.uniform(.85,1.18));dx=int(rng.integers(-10,11));dy=int(rng.integers(-10,11));flip=rng.random()<.5
        if rng.random()<.20:scale,dx,dy=1.,0,0
        n=round(96*scale);scale=n/96.;ox=(96-n)//2+dx;oy=(96-n)//2+dy
        im=Image.new('L',(96,96),114)
        im.paste(Image.fromarray(a[0]).resize((n,n),Image.Resampling.BILINEAR),(ox,oy))
        if flip:im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if rng.random()<.12:im=im.filter(ImageFilter.GaussianBlur(float(rng.uniform(.1,.4))))
        r={**row,'boxes':[],'ignore_boxes':[],'negative_people':[]}
        for key in ('boxes','ignore_boxes','negative_people'):
            for b in row[key]:
                old=b['bbox_xyxy'];raw=[old[0]*scale+ox,old[1]*scale+oy,old[2]*scale+ox,old[3]*scale+oy]
                q=np.clip(raw,0,96).tolist()
                if q[2]<=q[0] or q[3]<=q[1]:continue
                frac=(q[2]-q[0])*(q[3]-q[1])/max((raw[2]-raw[0])*(raw[3]-raw[1]),1e-7)
                if flip:q=[96-q[2],q[1],96-q[0],q[3]]
                item={**b,'bbox_xyxy':q,'visible_ratio':max(0.,b['visible_ratio']-b['retained_ratio']*(1-frac)),'retained_ratio':b['retained_ratio']*frac}
                dest=key
                if key=='boxes':
                    if q[2]-q[0]<8 or q[3]-q[1]<24:dest='ignore_boxes'
                    elif min(item['visible_ratio'],item['retained_ratio'])<.8:dest='negative_people'
                r[dest].append(item)
        images.append(np.asarray(im).copy()[None]);records.append(r)
    batch=torch.from_numpy(np.stack(images)).float()/255
    count=len(batch);shape=(count,1,1,1)
    contrast=torch.from_numpy(rng.uniform(.7,1.3,shape)).float();bright=torch.from_numpy(rng.uniform(.7,1.3,shape)).float()
    mean=batch.mean((2,3),keepdim=True)
    batch=((batch-mean)*contrast+mean)*bright
    gamma=torch.from_numpy(rng.uniform(.8,1.2,shape)).float()
    batch=batch.clamp(0,1).pow(gamma)
    if rng.random()<.25:batch+=torch.from_numpy(rng.normal(0,.012,batch.shape)).float()
    return batch.clamp(0,1).contiguous(memory_format=torch.channels_last),records


def targets(rows):
    N=len(rows); heat=np.zeros((N,1,24,24),np.float32);valid=np.ones_like(heat)
    reg=np.zeros((N,4,24,24),np.float32);mask=np.zeros_like(heat);priority=np.full((N,24,24),np.inf)
    for i,row in enumerate(rows):
        for b in row['ignore_boxes']:
            x1,y1,x2,y2=b['bbox_xyxy']
            valid[i,0,max(0,int(y1//4)):min(24,int(np.ceil(y2/4))),max(0,int(x1//4)):min(24,int(np.ceil(x2/4)))]=0
        # Center cells always own their regression target before neighboring cells.
        assignments=[]
        for b in row['boxes']:
            x1,y1,x2,y2=b['bbox_xyxy'];w=x2-x1;h=y2-y1;cx=(x1+x2)/8;cy=(y1+y2)/8
            gx=min(23,int(cx));gy=min(23,int(cy));radius=min(2,max(0,int(min(w,h)/8)));sigma=(2*radius+1)/6
            for yy in range(max(0,gy-radius),min(24,gy+radius+1)):
                for xx in range(max(0,gx-radius),min(24,gx+radius+1)):
                    heat[i,0,yy,xx]=max(heat[i,0,yy,xx],np.exp(-((xx-gx)**2+(yy-gy)**2)/(2*sigma*sigma)))
            valid[i,0,gy,gx]=1
            for xx,yy in ((gx,gy),(gx-1,gy),(gx+1,gy),(gx,gy-1),(gx,gy+1)):
                if not(0<=xx<24 and 0<=yy<24):continue
                if not(-.5<cx-xx<1.5 and -.5<cy-yy<1.5):continue
                # Primary assignment dominates auxiliary neighbor assignment.
                p=0 if (xx,yy)==(gx,gy) else 1+(cx-xx-.5)**2+(cy-yy-.5)**2
                if p<priority[i,yy,xx]:
                    priority[i,yy,xx]=p;mask[i,0,yy,xx]=1;reg[i,:,yy,xx]=[cx-xx,cy-yy,w/96,h/96]
    return tuple(torch.from_numpy(v) for v in (heat,valid,reg,mask))


def representation(output):
    return torch.cat([2*output[:,1:3].sigmoid()-.5,output[:,3:5].sigmoid()],1)


def aligned_iou(a,b):
    inter=(torch.minimum(a[:,2:],b[:,2:])-torch.maximum(a[:,:2],b[:,:2])).clamp(min=0).prod(1)
    union=(a[:,2:]-a[:,:2]).clamp(min=0).prod(1)+(b[:,2:]-b[:,:2]).clamp(min=0).prod(1)-inter
    return inter/union.clamp(min=1e-7)


def loss_fn(out,t,teacher=None,feature=None,teacher_feature=None,adapter=None,kd_weight=0.):
    heat,valid,reg,mask=t;p=out[:,0:1].sigmoid().clamp(1e-5,1-1e-5)
    pos=(heat==1).float();neg=(heat<1).float()*valid
    focal=-(p.log()*(1-p)**2*pos+(1-p).log()*p**2*(1-heat)**4*neg).sum()/pos.sum().clamp(min=1)
    ix=mask[:,0].nonzero(as_tuple=True);qt=torch.zeros_like(heat)
    zero=out.sum()*0;box=offset=size=zero
    pred_all=representation(out)
    if len(ix[0]):
        pred=pred_all.permute(0,2,3,1)[ix];gt=reg.permute(0,2,3,1)[ix]
        offset=F.smooth_l1_loss(pred[:,:2],gt[:,:2],beta=.2)
        size=F.smooth_l1_loss(pred[:,2:],gt[:,2:],beta=.1)
        grid=torch.stack([ix[2],ix[1]],1).float();pc=(grid+pred[:,:2])/24;tc=(grid+gt[:,:2])/24
        pb=torch.cat([pc-pred[:,2:]/2,pc+pred[:,2:]/2],1);tb=torch.cat([tc-gt[:,2:]/2,tc+gt[:,2:]/2],1)
        iou=aligned_iou(pb,tb)
        outer=(torch.maximum(pb[:,2:],tb[:,2:])-torch.minimum(pb[:,:2],tb[:,:2])).square().sum(1).clamp(min=1e-7)
        # DIoU replaces GIoU; normalized center-distance penalty targets localization.
        box=(1-iou+(pc-tc).square().sum(1)/outer).mean()
        qt[:,0][ix]=iou.detach()
    qloss=F.binary_cross_entropy_with_logits(out[:,5:6],qt,reduction='none')
    quality=(qloss*mask).sum()/mask.sum().clamp(min=1)+.25*(qloss*(1-mask)*valid).sum()/((1-mask)*valid).sum().clamp(min=1)
    total=focal+.5*offset+5*size+3*box+.5*quality
    kd=zero
    if teacher is not None:
        region=F.max_pool2d(pos,5,1,2)*valid
        # No pseudo-positive supervision in unannotated or ineligible regions.
        tp=teacher[:,0:1].sigmoid().detach()
        kd_heat=((p-tp).square()*region).sum()/region.sum().clamp(min=1)
        reliable=mask*(teacher[:,5:6].sigmoid().detach()>.35)
        kd_box=((pred_all-representation(teacher).detach()).abs()*reliable).sum()/(4*reliable.sum()).clamp(min=1)
        sf=F.normalize(adapter(feature),dim=1);tf=F.normalize(teacher_feature.detach(),dim=1)
        kd_feature=((sf-tf).square()*region).sum()/(region.sum()*sf.shape[1]).clamp(min=1)
        kd=2*kd_heat+kd_box+2*kd_feature
        total=total+kd_weight*kd
    return total,dict(heat=float(focal.detach()),box=float(box.detach()),offset=float(offset.detach()),size=float(size.detach()),quality=float(quality.detach()),kd=float(kd.detach()))


@torch.inference_mode()
def decode(output,confidence=.01,nms_iou=.5,topk=50,min_width=8,min_height=24):
    heat=output[:,0:1].sigmoid()
    score=heat*output[:,5:6].sigmoid().sqrt()
    peaks=score*(heat==F.max_pool2d(heat,3,1,1))
    scores,idx=torch.topk(peaks.flatten(1),topk,dim=1)
    values=torch.gather(representation(output).flatten(2),2,idx[:,None].expand(-1,4,-1))
    cx=(idx%24+values[:,0])*4;cy=(idx//24+values[:,1])*4;w=values[:,2]*96;h=values[:,3]*96
    boxes=torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1).clamp(0,96).cpu().numpy();scores=scores.cpu().numpy()
    result=[]
    for b,s in zip(boxes,scores):
        good=(s>=confidence)&(b[:,2]-b[:,0]>=min_width)&(b[:,3]-b[:,1]>=min_height)
        b,s=b[good],s[good];keep=[]
        for j in range(len(s)):
            if not keep or np.all(ious(b[j],b[keep])<=nms_iou):keep.append(j)
        result.append(np.concatenate([b[keep],s[keep,None]],1).astype(np.float32).reshape(-1,5))
    return result
