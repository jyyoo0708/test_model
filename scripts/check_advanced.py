"""Behavioral checks for label eligibility, new offsets/assignment, loss and teacher isolation."""
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'models'))
from advanced_model import AdvancedDetector,memory_profile
from advanced_utils import targets,loss_fn,decode,augment

def box(q):return dict(bbox_xyxy=q,visible_ratio=1.,retained_ratio=1.,reasons=[])
def row(boxes):return dict(boxes=boxes,ignore_boxes=[],negative_people=[])
def main():
    torch.set_num_threads(4);torch.manual_seed(13)
    r=row([box([27,12,43,60])]);tt=targets([r]);assert tt[3].sum()>=2
    ix=tt[3][:,0].nonzero(as_tuple=True);gt=tt[2].permute(0,2,3,1)[ix]
    cx=(ix[2]+gt[:,0])*4;cy=(ix[1]+gt[:,1])*4
    assert torch.allclose(cx,torch.full_like(cx,35)) and torch.allclose(cy,torch.full_like(cy,36))
    assert gt[:,:2].min()>=-.5 and gt[:,:2].max()<=1.5
    # Exact decode for a neighboring cell with negative offset.
    o=torch.full((1,6,24,24),-20.)
    x,y=9,9;o[0,0,y,x]=10;o[0,5,y,x]=10
    vals=torch.tensor([-.25,0.,16/96,48/96]);o[0,1:3,y,x]=torch.logit((vals[:2]+.5)/2)
    o[0,3:5,y,x]=torch.logit(vals[2:]);p=decode(o)[0]
    assert len(p)==1 and np.max(np.abs(p[0,:4]-[27,12,43,60]))<1e-4
    # Random crops and scales must never leave a disqualified box positive.
    rr=[row([box([0,0,18,48]),box([70,45,90,90])]) for _ in range(256)]
    xx=torch.zeros(256,1,96,96,dtype=torch.uint8)
    _,ar=augment(xx,rr,np.random.default_rng(17))
    rejected=0
    for r in ar:
        rejected+=len(r['negative_people'])+len(r['ignore_boxes'])
        for b in r['boxes']:
            q=b['bbox_xyxy'];assert min(b['visible_ratio'],b['retained_ratio'])>=.8 and q[2]-q[0]>=8 and q[3]-q[1]>=24
    assert rejected>0
    student=AdvancedDetector();teacher=AdvancedDetector(2).eval().requires_grad_(False);adapter=torch.nn.Conv2d(24,48,1,bias=False)
    inp=torch.rand(2,1,96,96);so,sf=student.features(inp)
    with torch.no_grad():to,tf=teacher.features(inp)
    loss,parts=loss_fn(so,targets([row([box([27,12,43,60])]),row([])]),to,sf,tf,adapter,1.)
    loss.backward();assert torch.isfinite(loss)
    assert all(p.grad is None for p in teacher.parameters())
    assert adapter.weight.grad is not None and student.quality.weight.grad is not None
    # A batch without positives must still train heatmap and quality backgrounds.
    out=student(inp);empty,_=loss_fn(out,targets([row([]),row([])]));assert torch.isfinite(empty);empty.backward()
    m=memory_profile(student.eval().fused());assert m['network_total_bytes']==155428
    print(json.dumps(dict(status='PASS',checks=['neighbor target geometry','negative offset decode','augmentation eligibility','finite supervised/KD gradients','teacher frozen','empty target batch','two-bank SRAM schedule'],sram=m['network_total_bytes'],augmentation_reclassified_boxes=rejected),indent=2))
if __name__=='__main__':main()
