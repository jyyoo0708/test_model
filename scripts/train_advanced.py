#!/usr/bin/env python3
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import argparse,hashlib,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'models'))
from advanced_model import AdvancedDetector,memory_profile
from advanced_utils import load_split,augment,targets,loss_fn,decode,detection_metrics,presence_metrics,to_original

def dump(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

@torch.inference_mode()
def predict(model,x,batch=128):
    model.eval();p=[]
    for start in range(0,len(x),batch):
        p.extend(decode(model((x[start:start+batch].float()/255).contiguous(memory_format=torch.channels_last))))
    return p


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['teacher','student'],required=True)
    ap.add_argument('--epochs',type=int,default=80);ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--batch-size',type=int,default=128);ap.add_argument('--seed',type=int,default=20261105)
    ap.add_argument('--data',type=Path,default=ROOT/'data/processed/advanced_detection')
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--teacher',type=Path)
    ap.add_argument('--lr',type=float,default=.002);ap.add_argument('--patience',type=int,default=20)
    ap.add_argument('--no-kd',action='store_true');args=ap.parse_args();started=time.time()
    args.output.mkdir(parents=True,exist_ok=False)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(args.seed);width=2 if args.stage=='teacher' else 1
    model=AdvancedDetector(width).to(memory_format=torch.channels_last)
    teacher=None;adapter=None
    if args.teacher:
        teacher=AdvancedDetector(2).eval().to(memory_format=torch.channels_last)
        teacher.load_state_dict(torch.load(args.teacher,map_location='cpu',weights_only=False)['state_dict'])
        teacher.requires_grad_(False);adapter=nn.Conv2d(24,48,1,bias=False)
    parameters=list(model.parameters())+(list(adapter.parameters()) if adapter is not None else [])
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    cfg.update(width=width,device='cpu',precision='float32',torch=torch.__version__,
               methods=[1,2,3,4,5]+([7] if teacher is not None and not args.no_kd else []),
               model_selection='highest validation AP50 on original CrowdHuman validation; MOT validation separately reported',
               loss='center focal + .5 smoothL1 offset + 5 smoothL1 size + 3 DIoU + .5 quality BCE + ramped KD',
               kd='GT-region heatmap MSE + quality-masked bbox L1 + normalized feature imitation; training-only 24->48 adapter',
               decode='offset=2*sigmoid-.5; size=sigmoid; score=heat*sqrt(quality); peak3; NMS=.5; top50; width8 height24',
               source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'models/advanced_model.py',ROOT/'models/advanced_utils.py']},
               data_config_sha256=hashlib.sha256((args.data/'config.json').read_bytes()).hexdigest())
    cfg['initial_student_weights_sha256']=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    cfg['initial_torch_rng_sha256']=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    if args.teacher:cfg['teacher_sha256']=hashlib.sha256(args.teacher.read_bytes()).hexdigest()
    dump(args.output/'config.json',cfg)
    if args.stage=='student':
        profile=memory_profile(model.fused());assert profile['network_total_bytes']<=175640
        dump(args.output/'memory.json',profile)
    x,rows=load_split(args.data,'train');vx,vrows=load_split(args.data,'val')
    assert not {r['group_id'] for r in rows}&{r['group_id'] for r in vrows}
    val_idx=[i for i,r in enumerate(vrows) if r['dataset']=='crowdhuman']
    for r in rows+vrows:
        for b in r['boxes']:
            q=b['bbox_xyxy'];assert min(b['visible_ratio'],b['retained_ratio'])>=.8 and q[2]-q[0]>=8-1e-5 and q[3]-q[1]>=24-1e-5
    optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=.00002)
    best=-1.;best_epoch=0;history=[]
    print(f'START {args.stage} train={len(rows)} val={len(vrows)} parameters={sum(p.numel() for p in model.parameters())}',flush=True)
    for epoch in range(1,args.epochs+1):
        tick=time.time();model.train();order=torch.randperm(len(rows));losses=[];parts=[]
        for start in range(0,len(order),args.batch_size):
            ids=order[start:start+args.batch_size]
            inputs,rr=augment(x[ids],[rows[i] for i in ids.tolist()],rng);tt=targets(rr)
            optimizer.zero_grad(set_to_none=True)
            out,feat=model.features(inputs)
            tout=tfeat=None
            if teacher is not None and not args.no_kd:
                with torch.no_grad():tout,tfeat=teacher.features(inputs)
            loss,values=loss_fn(out,tt,tout,feat,tfeat,adapter,min(1.,epoch/15))
            assert torch.isfinite(loss),values
            loss.backward();nn.utils.clip_grad_norm_(parameters,10);optimizer.step()
            losses.append(float(loss.detach()));parts.append(values)
        pp=predict(model,vx,args.batch_size)
        vm=detection_metrics([vrows[i] for i in val_idx],[pp[i] for i in val_idx])
        mot_idx=[i for i,r in enumerate(vrows) if r['dataset']=='mot17']
        mm=detection_metrics([vrows[i] for i in mot_idx],[pp[i] for i in mot_idx])
        row=dict(epoch=epoch,loss=float(np.mean(losses)),val_ap50=vm['ap'],mot_val_ap50=mm['ap'],
                 seconds=time.time()-tick,lr=optimizer.param_groups[0]['lr'],
                 **{k:float(np.mean([p[k] for p in parts])) for k in parts[0]})
        history.append(row);dump(args.output/'history.json',history);print(json.dumps(row),flush=True)
        if vm['ap']>best+1e-6:
            best=vm['ap'];best_epoch=epoch
            checkpoint=dict(state_dict=model.state_dict(),epoch=epoch,validation_ap50=best,config=cfg)
            if adapter is not None:checkpoint['adapter_state_dict']=adapter.state_dict()
            torch.save(checkpoint,args.output/'best.pt')
        scheduler.step()
        if epoch>=50 and epoch-best_epoch>=args.patience:break
    ck=torch.load(args.output/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict']);model.eval()
    result=dict(best_epoch=best_epoch,epochs_run=len(history),validation_ap50=best,train_seconds=time.time()-started)
    dump(args.output/'training_result.json',result)
    print('TRAINING COMPLETE '+json.dumps(result),flush=True)

if __name__=='__main__':main()
