#!/usr/bin/env python3
"""Freeze threshold on original validation, then evaluate untouched tests and export student."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'models'))
from advanced_model import AdvancedDetector,memory_profile
from advanced_utils import load_split,decode,detection_metrics,presence_metrics,to_original,ious

def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
@torch.inference_mode()
def predict(m,x,batch=128,baseline=False):
    pp=[]
    if baseline:
        from utils import decode as baseline_decode
    for start in range(0,len(x),batch):
        o=m((x[start:start+batch].float()/255).contiguous(memory_format=torch.channels_last))
        pp.extend(baseline_decode(o,min_width=6,min_height=20) if baseline else decode(o))
    return pp

def metrics(rows,pred,threshold):
    aps=[detection_metrics(rows,pred,iou=float(t))['ap'] for t in np.arange(.5,.951,.05)]
    return dict(ap50=aps[0],ap75=aps[5],map50_95=float(np.mean(aps)),images=len(rows),people=sum(len(r['boxes']) for r in rows),
                detection=detection_metrics(rows,pred,confidence=threshold),presence=presence_metrics(rows,pred,threshold))

def error_slices(rows,pred,threshold):
    data={k:[0,0] for k in ('edge','center','single','multiple','height24_40','height40plus')}
    errors=[];dup=0
    for row,pp in zip(rows,pred):
        p=pp[pp[:,4]>=threshold];matched=set()
        for b in row['boxes']:
            q=np.array(b['bbox_xyxy']);overlap=ious(q,p[:,:4]);order=np.argsort(-overlap)
            ix=next((int(j) for j in order if j not in matched and overlap[j]>=.5),None)
            hit=ix is not None
            if hit:
                matched.add(ix);a=p[ix,:4]
                errors.append([abs((a[0]+a[2]-q[0]-q[2])/2),abs((a[1]+a[3]-q[1]-q[3])/2),abs((a[2]-a[0])-(q[2]-q[0])),abs((a[3]-a[1])-(q[3]-q[1]))])
            keys=['edge' if min(q[0],q[1],96-q[2],96-q[3])<8 else 'center',
                  'single' if len(row['boxes'])==1 else 'multiple','height24_40' if q[3]-q[1]<40 else 'height40plus']
            for k in keys:data[k][1]+=1;data[k][0]+=int(hit)
            dup+=max(0,int((overlap>=.5).sum())-1)
    return dict(recall_slices={k:dict(matched=v[0],gt=v[1],recall=v[0]/max(v[1],1)) for k,v in data.items()},
                matched_box_mean_abs_error_cx_cy_w_h_96=np.mean(errors,axis=0).tolist() if errors else None,
                excess_predictions_matching_same_gt=dup)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--baseline',action='store_true');args=ap.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    out=args.run.resolve();out.relative_to(ROOT);out.mkdir(exist_ok=True,parents=True)
    if args.baseline:
        m=torch.jit.load(str(ROOT/'baseline/person_detector/run02/model_float32.ts')).eval();threshold=.329
    else:
        ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);m=AdvancedDetector(ck['config']['width'])
        m.load_state_dict(ck['state_dict']);m.eval();threshold=None
    data=ROOT/'data/processed/advanced_detection'
    vx,vr=load_split(data,'val');vp=predict(m,vx,baseline=args.baseline)
    idx=[i for i,r in enumerate(vr) if r['dataset']=='crowdhuman'];cr=[vr[i] for i in idx];cp=[vp[i] for i in idx]
    if threshold is None:
        sweep=[]
        for t in np.r_[.01,np.arange(.05,.851,.025)]:
            dm=detection_metrics(cr,cp,confidence=float(t));pm=presence_metrics(cr,cp,float(t))
            sweep.append(dict(threshold=float(t),f1=dm['f1'],balanced_accuracy=pm['balanced_accuracy']))
        threshold=max(sweep,key=lambda r:(r['f1'],r['balanced_accuracy'],r['threshold']))['threshold']
        dump(out/'threshold_validation.json',sweep)
    result=dict(threshold=threshold,nms_iou=.5,peak_filter=3,topk=50,metrics={},slices={})
    # No test data used until checkpoint and threshold are fixed.
    for split in ('train','val','test'):
        x,rows=load_split(data,split);p=vp if split=='val' else predict(m,x,baseline=args.baseline)
        for dataset in ('crowdhuman','mot17'):
            ids=[i for i,r in enumerate(rows) if r['dataset']==dataset];rr=[rows[i] for i in ids];pp=[p[i] for i in ids]
            key=f'{dataset}_{split}';result['metrics'][key]=metrics(rr,pp,threshold)
            result['slices'][key]=error_slices(rr,pp,threshold)
        with (out/f'{split}_predictions.jsonl').open('w') as f:
            for r,pp in zip(rows,p):
                f.write(json.dumps(dict(source_id=r['source_id'],dataset=r['dataset'],person_present=bool(np.any(pp[:,4]>=threshold)),
                    boxes_xyxy_score_96=pp[pp[:,4]>=threshold].tolist(),all_candidates_96=pp.tolist()))+'\n')
    if not args.baseline:
        fused=m.fused();example=vx[:32].float()/255
        with torch.inference_mode():
            expected=m(example);actual=fused(example);err=float((expected-actual).abs().max())
            assert torch.allclose(expected,actual,atol=4e-5,rtol=1e-4)
            script=torch.jit.freeze(torch.jit.trace(fused,example[:1]));script.save(str(out/'model_float32.ts'))
            reload=torch.jit.load(str(out/'model_float32.ts'));loaded=reload(example)
            assert torch.allclose(actual,loaded,atol=4e-5,rtol=1e-4)
        result['export_validation']=dict(bn_fold_max_abs_error=err,reload_max_abs_error=float((loaded-actual).abs().max()))
        torch.save(dict(state_dict=fused.state_dict(),threshold=threshold,config=ck['config']),out/'model_float32.pt')
        if ck['config']['width']==1:
            mem=memory_profile(fused);dump(out/'memory.json',mem)
            # Actual concrete storage proposal for requantization (one int32 multiplier + int8 shift/output channel).
            channels=sum(r['biases'] for r in mem['layers'])
            result['sram_proposal']=dict(network_bytes=mem['network_total_bytes'],requant_multiplier_int32_and_shift_int8=channels*5,
                                         network_plus_requant_bytes=mem['network_total_bytes']+channels*5,
                                         note='Not an RTL implementation; bank schedule assumptions and unspecified alignment/postprocessing remain in memory.json')
            samples=vx[:1].float()/255
            with torch.inference_mode():
                for _ in range(10):reload(samples)
                tick=time.perf_counter()
                for _ in range(100):reload(samples)
            result['cpu_latency_ms_batch1_threads8']=(time.perf_counter()-tick)*10
            inference_dir=ROOT/'inference';inference_dir.mkdir(exist_ok=True)
            cfg=dict(model=str((out/'model_float32.ts').relative_to(ROOT)),threshold=threshold,input_size=96,padding=114,
                     channels=['heatmap','offset_x','offset_y','width','height','quality'],offset='2*sigmoid-.5',size='sigmoid*96',
                     score='sigmoid(heatmap)*sqrt(sigmoid(quality))',nms_iou=.5,topk=50,min_width=8,min_height=24)
            dump(inference_dir/'config.json',cfg)
    dump(out/'results.json',result)
    print(json.dumps({k:dict(ap50=v['ap50'],ap75=v['ap75'],accuracy=v['presence']['accuracy'],recall=v['detection']['recall']) for k,v in result['metrics'].items()},indent=2),flush=True)
if __name__=='__main__':main()
