#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"models/person_detector"))
from model import PersonDetector,memory_profile
from utils import load_split,build_targets,translated_view,detection_loss,decode,detection_metrics,presence_metrics,to_original


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")


@torch.inference_mode()
def predict(model,x,batch=128,min_width=8,min_height=24):
    model.eval();predictions=[]
    for start in range(0,len(x),batch):
        inputs=x[start:start+batch].float().div_(255).contiguous(memory_format=torch.channels_last)
        predictions.extend(decode(model(inputs),min_width=min_width,min_height=min_height))
    return predictions


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,default=ROOT/"data/processed/person96_detection")
    ap.add_argument("--output",type=Path,default=ROOT/"models/person_detector/run01")
    ap.add_argument("--epochs",type=int,default=80)
    ap.add_argument("--threads",type=int,default=8)
    ap.add_argument("--batch-size",type=int,default=128)
    ap.add_argument("--lr",type=float,default=.002)
    ap.add_argument("--seed",type=int,default=20261105)
    ap.add_argument("--patience",type=int,default=15)
    ap.add_argument('--spatial-augment',action='store_true')
    ap.add_argument('--supervise-known-small',action='store_true')
    ap.add_argument('--initialize-classifier',action='store_true')
    ap.add_argument('--min-pred-width',type=float,default=8)
    ap.add_argument('--min-pred-height',type=float,default=24)
    args=ap.parse_args();started=time.time()
    args.output.mkdir(parents=True,exist_ok=False)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    cfg.update(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device="cpu",precision="float32",
               selection="highest validation 101-point AP50, score floor 0.01, max50, NMS0.5",
               threshold_selection="maximum validation detection F1 at IoU0.5; ties presence balanced accuracy then higher score",
               heatmap_stride=4,loss="Center focal + offset L1 + 4*size L1 + 2*GIoU",
               augmentation="both horizontal orientations each epoch; brightness/contrast [0.85,1.15]; no crop/rotation",
               source_code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   (Path(__file__),ROOT/"models/person_detector/model.py",ROOT/"models/person_detector/utils.py")},
               data_config_sha256=hashlib.sha256((args.data/"config.json").read_bytes()).hexdigest())
    dump(args.output/"config.json",cfg)
    x,rows=load_split(args.data,"train");vx,vrows=load_split(args.data,"val")
    assert not {r['group_id'] for r in rows}&{r['group_id'] for r in vrows}
    shifts=[(0,0),(4,0),(-4,0),(0,4),(0,-4)] if args.spatial_augment else [(0,0)]
    variants=[];target_variants=[];collisions=[]
    for dx,dy in shifts:
        xx,rr=translated_view(x,rows,dx,dy)
        normal,c=build_targets(rr,supervise_known_small=args.supervise_known_small)
        flipped,fc=build_targets(rr,True,supervise_known_small=args.supervise_known_small)
        variants.extend([xx,xx.flip(-1)])
        target_variants.extend([normal,flipped]);collisions.append({'dx':dx,'dy':dy,'normal':c,'flipped':fc})
    x=torch.cat(variants,0)
    targets=tuple(torch.cat([v[j] for v in target_variants],0) for j in range(4))
    model=PersonDetector().to(memory_format=torch.channels_last)
    if args.initialize_classifier:
        initial=torch.load(ROOT/'models/person96/run01/best.pt',map_location='cpu',weights_only=False)['state_dict']
        state=model.state_dict()
        for old,new in (('blocks.0.','c1.'),('blocks.1.','c2.')):
            for key,value in initial.items():
                if key.startswith(old):
                    target_key=new+key[len(old):]
                    assert state[target_key].shape==value.shape
                    state[target_key]=value
        model.load_state_dict(state)
        cfg['initialization']='first two Conv/BN blocks from previous classifier trained on same train sources; all other layers random'
        cfg['classifier_checkpoint_sha256']=hashlib.sha256((ROOT/'models/person96/run01/best.pt').read_bytes()).hexdigest()
        dump(args.output/'config.json',cfg)
    profile=memory_profile(model.fused());assert profile['within_limit']
    dump(args.output/"memory.json",profile)
    dump(args.output/"target_stats.json",{"view_collisions":collisions,
                                        "train_original_people":sum(len(r['boxes']) for r in rows)})
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=.00002)
    best,best_epoch,history=-1.,0,[]
    print(f"PARAMETERS {sum(p.numel() for p in model.parameters())}; frames train={len(rows)} val={len(vrows)}; future SRAM {profile['total_bytes']}B",flush=True)
    for epoch in range(1,args.epochs+1):
        tick=time.time();model.train();order=torch.randperm(2*len(rows));losses=[];components=[]
        for step,start in enumerate(range(0,len(order),args.batch_size)):
            base=order[start:start+args.batch_size]
            view=torch.randint(len(shifts),(len(base),))*2+base//len(rows)
            indices=base%len(rows)+view*len(rows)
            inputs=x[indices].float().div_(255)
            mean=inputs.mean((2,3),keepdim=True)
            contrast=torch.empty(len(indices),1,1,1).uniform_(.85,1.15)
            brightness=torch.empty(len(indices),1,1,1).uniform_(.85,1.15)
            inputs=((inputs-mean)*contrast+mean).mul_(brightness).clamp_(0,1).contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            loss,parts=detection_loss(model(inputs),tuple(t[indices] for t in targets))
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step();losses.append(float(loss.detach()));components.append(parts)
            if (step+1)%25==0:
                print(f"epoch {epoch:03d} batch {step+1}/{(len(order)+args.batch_size-1)//args.batch_size} elapsed={time.time()-tick:.1f}s",flush=True)
        vp=predict(model,vx,args.batch_size,args.min_pred_width,args.min_pred_height)
        vm=detection_metrics(vrows,vp)
        row={"epoch":epoch,"train_loss":float(np.mean(losses)),"val_ap50":vm['ap'],
             "val_precision_at_0_01":vm['precision'],"val_recall_at_0_01":vm['recall'],
             "lr":optimizer.param_groups[0]['lr'],"seconds":time.time()-tick,
             **{"train_"+k:float(np.mean([c[k] for c in components])) for k in components[0]}}
        history.append(row);dump(args.output/"history.json",history);print(json.dumps(row),flush=True)
        if vm['ap']>best+1e-6:
            best,best_epoch=vm['ap'],epoch
            torch.save({"state_dict":model.state_dict(),"epoch":epoch,"validation_ap50":best,"config":cfg},args.output/"best.pt")
        scheduler.step()
        if epoch>=30 and epoch-best_epoch>=args.patience:
            print(f"EARLY STOP {epoch}; best {best_epoch}",flush=True);break
    checkpoint=torch.load(args.output/"best.pt",map_location="cpu",weights_only=False)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    vp=predict(model,vx,args.batch_size,args.min_pred_width,args.min_pred_height)
    candidates=[];threshold_metrics=[]
    for threshold in np.r_[.01,np.arange(.05,.851,.025)]:
        dm=detection_metrics(vrows,vp,confidence=float(threshold));pm=presence_metrics(vrows,vp,float(threshold))
        candidates.append((dm['f1'],pm['balanced_accuracy'],float(threshold)))
        threshold_metrics.append({"threshold":float(threshold),"detection_f1":dm['f1'],"presence_accuracy":pm['accuracy']})
    threshold=max(candidates)[2];dump(args.output/"threshold_sweep_validation.json",threshold_metrics)
    # Held-out test first loaded after architecture/checkpoint/threshold selection.
    tx,trows=load_split(args.data,"test")
    assert not {r['group_id'] for r in trows}&{r['group_id'] for r in rows+vrows}
    tp=predict(model,tx,args.batch_size,args.min_pred_width,args.min_pred_height)
    result={"best_epoch":best_epoch,"epochs_run":len(history),"threshold":threshold,"nms_iou":.5,"topk":50,"metrics":{}}
    # Train data stored twice for augmentation: final train evaluation uses original frames only.
    train_pred=predict(model,x[:len(rows)],args.batch_size,args.min_pred_width,args.min_pred_height)
    for split,rr,pp in (("train",rows,train_pred),("val",vrows,vp),("test",trows,tp)):
        aps=[detection_metrics(rr,pp,iou=float(t))['ap'] for t in np.arange(.5,.951,.05)]
        result['metrics'][split]={"ap50":aps[0],"ap75":aps[5],"map50_95":float(np.mean(aps)),
                                 "detection_at_selected_threshold":detection_metrics(rr,pp,confidence=threshold),
                                 "presence_at_selected_threshold":presence_metrics(rr,pp,threshold),
                                 "images":len(rr),"people":sum(len(r['boxes']) for r in rr)}
        with (args.output/f"{split}_predictions.jsonl").open('w') as f:
            for r,pred in zip(rr,pp):
                keep=pred[pred[:,4]>=threshold]
                f.write(json.dumps({"source_id":r['source_id'],"person_present":bool(len(keep)),
                    "boxes_xyxy_score_96":keep.tolist(),"boxes_xyxy_score_original":to_original(keep,r).tolist(),
                    "all_candidates_96":pred.tolist()})+'\n')
    fused=model.fused()
    example=tx[:64].float().div(255)
    with torch.inference_mode():
        a,b=model(example),fused(example)
        result['bn_fold_max_abs_error']=float((a-b).abs().max())
        assert torch.allclose(a,b,atol=3e-5,rtol=1e-4)
        script=torch.jit.freeze(torch.jit.trace(fused,example[:1]));script.save(str(args.output/"model_float32.ts"))
        loaded=torch.jit.load(str(args.output/"model_float32.ts")).eval()
        result['export_reload_max_abs_error']=float((loaded(example)-b).abs().max())
        assert torch.allclose(loaded(example),b,atol=3e-5,rtol=1e-4)
    torch.save({"state_dict":fused.state_dict(),"threshold":threshold,"config":cfg},args.output/"model_float32.pt")
    np.savez_compressed(args.output/"weights_float32.npz",**{k:v.detach().numpy() for k,v in fused.state_dict().items()})
    result['wall_seconds']=time.time()-started
    dump(args.output/"results.json",result);print('FINAL '+json.dumps(result),flush=True)


if __name__=='__main__':main()
