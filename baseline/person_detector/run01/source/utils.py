import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def load_split(data, split):
    rows = [json.loads(line) for line in (Path(data)/f"{split}.jsonl").open()]
    with np.load(Path(data)/f"{split}.npz", allow_pickle=False) as pack:
        assert np.array_equal(pack["source_id"], [r["source_id"] for r in rows])
        x = torch.from_numpy(pack["x"].copy())
    return x, rows


def build_targets(rows, flip=False):
    heat = np.zeros((len(rows), 1, 24, 24), np.float32)
    valid = np.ones_like(heat)
    reg = np.zeros((len(rows), 4, 24, 24), np.float32)
    mask = np.zeros_like(heat)
    collisions = 0
    def mirror(b):
        return [96-b[2], b[1], 96-b[0], b[3]] if flip else b
    for i, row in enumerate(rows):
        for item in row["ignore_boxes"]:
            x1,y1,x2,y2 = mirror(item["bbox_xyxy"])
            valid[i,0,max(0,int(y1//4)):min(24,int(np.ceil(y2/4))),max(0,int(x1//4)):min(24,int(np.ceil(x2/4)))] = 0
        # Supervise centers of known large partial/occluded humans as negatives.
        for item in row["negative_people"]:
            x1,y1,x2,y2 = mirror(item["bbox_xyxy"])
            gx,gy = min(23,int((x1+x2)/8)), min(23,int((y1+y2)/8))
            valid[i,0,gy,gx] = 1
        # Largest target owns regression if multiple centers quantize to one cell.
        items = sorted(row["boxes"], key=lambda b: (b["bbox_xyxy"][2]-b["bbox_xyxy"][0])*(b["bbox_xyxy"][3]-b["bbox_xyxy"][1]), reverse=True)
        for item in items:
            x1,y1,x2,y2 = mirror(item["bbox_xyxy"])
            w,h = x2-x1, y2-y1
            cx,cy = (x1+x2)/8, (y1+y2)/8
            gx,gy = min(23,int(cx)), min(23,int(cy))
            radius = min(2,max(0,int(min(w,h)/8)))
            for yy in range(max(0,gy-radius),min(24,gy+radius+1)):
                for xx in range(max(0,gx-radius),min(24,gx+radius+1)):
                    sigma = (2*radius+1)/6
                    value = np.exp(-((xx-gx)**2+(yy-gy)**2)/(2*sigma*sigma))
                    heat[i,0,yy,xx] = max(heat[i,0,yy,xx],value)
            valid[i,0,gy,gx] = 1
            if mask[i,0,gy,gx]:
                collisions += 1
                continue
            mask[i,0,gy,gx] = 1
            reg[i,:,gy,gx] = [cx-gx,cy-gy,w/96,h/96]
    return tuple(torch.from_numpy(a) for a in (heat,valid,reg,mask)), collisions


def giou_loss(a,b):
    lt = torch.maximum(a[:,:2],b[:,:2]); rb = torch.minimum(a[:,2:],b[:,2:])
    inter = (rb-lt).clamp(min=0).prod(1)
    aa = (a[:,2:]-a[:,:2]).clamp(min=0).prod(1)
    bb = (b[:,2:]-b[:,:2]).clamp(min=0).prod(1)
    union = aa+bb-inter
    iou = inter/union.clamp(min=1e-7)
    outer = (torch.maximum(a[:,2:],b[:,2:])-torch.minimum(a[:,:2],b[:,:2])).clamp(min=0).prod(1)
    return 1-iou+(outer-union)/outer.clamp(min=1e-7)


def detection_loss(output, targets):
    heat,valid,reg,mask = targets
    p = output[:,0:1].sigmoid().clamp(1e-5,1-1e-5)
    positive = (heat==1).float()
    negative = (heat<1).float()*valid
    focal = -(p.log()*(1-p).pow(2)*positive + (1-p).log()*p.pow(2)*(1-heat).pow(4)*negative).sum()/positive.sum().clamp(min=1)
    indices = mask[:,0].nonzero(as_tuple=True)
    if len(indices[0]) == 0:
        return focal, {"heat": float(focal.detach()), "box": 0., "offset": 0., "size": 0.}
    pred = output[:,1:].permute(0,2,3,1)[indices].sigmoid()
    target = reg.permute(0,2,3,1)[indices]
    offset = nn.functional.l1_loss(pred[:,:2],target[:,:2])
    size = nn.functional.l1_loss(pred[:,2:],target[:,2:])
    grid = torch.stack([indices[2],indices[1]],1).float()
    pc = (grid+pred[:,:2])/24
    tc = (grid+target[:,:2])/24
    pb = torch.cat([pc-pred[:,2:]/2,pc+pred[:,2:]/2],1)
    tb = torch.cat([tc-target[:,2:]/2,tc+target[:,2:]/2],1)
    box = giou_loss(pb,tb).mean()
    total = focal + offset + 4*size + 2*box
    return total, {"heat": float(focal.detach()), "box": float(box.detach()),
                   "offset": float(offset.detach()), "size": float(size.detach())}


def ious(box, boxes, ioa=False):
    boxes = np.asarray(boxes,dtype=np.float32).reshape(-1,4)
    if len(boxes)==0:
        return np.zeros(0,np.float32)
    lt = np.maximum(box[:2], boxes[:,:2]); rb = np.minimum(box[2:],boxes[:,2:])
    inter = np.maximum(0,rb-lt).prod(1)
    area = max(0,box[2]-box[0])*max(0,box[3]-box[1])
    other = np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(1)
    return inter/np.maximum(area if ioa else area+other-inter,1e-7)


@torch.inference_mode()
def decode(output, confidence=.01, nms_iou=.5, topk=50):
    heat = output[:,0:1].sigmoid()
    peaks = heat*(heat==nn.functional.max_pool2d(heat,3,stride=1,padding=1))
    scores,indices = torch.topk(peaks.flatten(1),topk,dim=1)
    values = output[:,1:].sigmoid().flatten(2)
    values = torch.gather(values,2,indices[:,None].expand(-1,4,-1))
    cx = (indices%24 + values[:,0])*4
    cy = (indices//24 + values[:,1])*4
    w,h = values[:,2]*96,values[:,3]*96
    boxes = torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1).clamp(0,96).numpy()
    scores = scores.numpy()
    result = []
    for b,s in zip(boxes,scores):
        good = (s>=confidence)&((b[:,2]-b[:,0])>=8)&((b[:,3]-b[:,1])>=24)
        b,s = b[good],s[good]
        keep = []
        for j in range(len(s)):
            if not keep or np.all(ious(b[j],b[keep])<=nms_iou):
                keep.append(j)
        result.append(np.concatenate([b[keep],s[keep,None]],1).astype(np.float32).reshape(-1,5))
    return result


def detection_metrics(rows,predictions,iou=.5,confidence=.01):
    events, total_gt = [], sum(len(row["boxes"]) for row in rows)
    per_image = []
    for row,pred in zip(rows,predictions):
        gt = np.array([b["bbox_xyxy"] for b in row["boxes"]],np.float32).reshape(-1,4)
        ignored = np.array([b["bbox_xyxy"] for b in row["ignore_boxes"]],np.float32).reshape(-1,4)
        matched = np.zeros(len(gt),bool)
        counts = {"tp":0,"fp":0,"ignored":0}
        for det in pred:
            if det[4]<confidence:
                continue
            overlaps = ious(det[:4],gt)
            overlaps[matched] = -1
            j = int(np.argmax(overlaps)) if len(overlaps) else -1
            if j>=0 and overlaps[j]>=iou:
                matched[j] = True; event=1;counts["tp"]+=1
            elif len(ignored) and np.max(ious(det[:4],ignored,ioa=True))>=.5:
                counts["ignored"]+=1
                continue
            else:
                event=0;counts["fp"]+=1
            events.append((float(det[4]),event))
        counts["fn"] = int((~matched).sum())
        per_image.append(counts)
    events.sort(key=lambda e:e[0],reverse=True)
    tps = np.cumsum([e[1] for e in events]);fps=np.cumsum([1-e[1] for e in events])
    recalls = tps/max(total_gt,1);precisions=tps/np.maximum(tps+fps,1)
    ap = float(np.mean([precisions[recalls>=r].max() if np.any(recalls>=r) else 0. for r in np.linspace(0,1,101)]))
    tp,fp = sum(r["tp"] for r in per_image),sum(r["fp"] for r in per_image)
    precision=tp/max(tp+fp,1);recall=tp/max(total_gt,1)
    return {"ap":ap,"iou":iou,"confidence":confidence,"gt":total_gt,"tp":tp,"fp":fp,
            "fn":total_gt-tp,"ignored_detections":sum(r["ignored"] for r in per_image),
            "precision":precision,"recall":recall,"f1":2*precision*recall/max(precision+recall,1e-9)}


def presence_metrics(rows,predictions,confidence):
    # Runtime presence uses actual predicted boxes, without GT-based ignore filtering.
    gt=np.array([bool(r["boxes"]) for r in rows]);pred=np.array([np.any(p[:,4]>=confidence) for p in predictions])
    tp=int((gt&pred).sum());tn=int((~gt&~pred).sum());fp=int((~gt&pred).sum());fn=int((gt&~pred).sum())
    p=tp/max(tp+fp,1);r=tp/max(tp+fn,1)
    return {"accuracy":(tp+tn)/len(gt),"precision":p,"recall":r,"f1":2*p*r/max(p+r,1e-9),
            "balanced_accuracy":(r+tn/max(tn+fp,1))/2,"confusion_true_rows_pred_columns":[[tn,fp],[fn,tp]],
            "majority_baseline":float(max(gt.sum(),(~gt).sum())/len(gt))}


def to_original(boxes,row):
    b=np.array(boxes,dtype=np.float32).reshape(-1,5).copy()
    t=row["transform"]
    b[:,[0,2]]=(b[:,[0,2]]-t["pad_left"])/t["scale_x"]
    b[:,[1,3]]=(b[:,[1,3]]-t["pad_top"])/t["scale_y"]
    b[:,[0,2]]=b[:,[0,2]].clip(0,row["original_width"])
    b[:,[1,3]]=b[:,[1,3]].clip(0,row["original_height"])
    return b
