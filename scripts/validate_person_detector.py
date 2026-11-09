#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'models/person_detector'))
from model import PersonDetector
from utils import load_split,build_targets,translated_view,decode,detection_metrics,presence_metrics,to_original


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-only',action='store_true')
    ap.add_argument('--run',type=Path,default=ROOT/'models/person_detector/run02');args=ap.parse_args()
    torch.set_num_threads(8)
    data=ROOT/'data/processed/person96_detection';run=args.run
    groups={};hashes={};counts={};records={}
    for split in ('train','val','test'):
        x,rows=load_split(data,split);records[split]=rows
        assert len(x)==len(rows) and x.dtype==torch.uint8 and tuple(x.shape[1:])==(1,96,96)
        people=0
        for i,row in enumerate(rows):
            assert row['split']==split
            assert groups.setdefault(row['group_id'],split)==split
            digest=hashlib.sha256(x[i].numpy().tobytes()).hexdigest()
            assert hashes.setdefault(digest,split)==split,'pixel duplicate crosses splits'
            with Image.open(data/row['image']) as image:
                assert image.mode=='L' and image.size==(96,96)
                assert np.array_equal(np.asarray(image),x[i,0].numpy())
            t=row['transform'];w,h=row['original_width'],row['original_height']
            assert round(w*t['scale_x'])==t['resized_width'] and round(h*t['scale_y'])==t['resized_height']
            for box in row['boxes']:
                x1,y1,x2,y2=box['bbox_xyxy'];sx,sy,sw,sh=box['source_fbox_xywh']
                expected=[max(0,sx)*t['scale_x']+t['pad_left'],max(0,sy)*t['scale_y']+t['pad_top'],
                          min(w,sx+sw)*t['scale_x']+t['pad_left'],min(h,sy+sh)*t['scale_y']+t['pad_top']]
                assert np.allclose(expected,[x1,y1,x2,y2])
                assert -1e-5<=x1<x2<=96+1e-5 and -1e-5<=y1<y2<=96+1e-5 and x2-x1>=8 and y2-y1>=24
                assert sw>=12 and sh>=48 and box['visible_ratio']>=.8 and box['retained_ratio']>=.8 and not box['reasons']
                original=to_original([[x1,y1,x2,y2,1]],row)[0,:4]
                assert np.allclose(original,[max(0,sx),max(0,sy),min(w,sx+sw),min(h,sy+sh)],atol=.001)
                people+=1
        counts[split]={'images':len(rows),'people':people}
    # Independently constructed perfect prediction checks decode/offset/size/IoU/AP.
    out=torch.full((1,5,24,24),-20.)
    out[0,0,9,6]=10.;out[0,1:3,9,6]=torch.logit(torch.tensor([1e-5,1e-5]))
    out[0,3:,9,6]=torch.logit(torch.tensor([.25,56/96]))
    pred=decode(out)
    assert len(pred[0])==1 and np.allclose(pred[0][0,:4],[12,8,36,64],atol=.001)
    row={'boxes':[{'bbox_xyxy':[12,8,36,64]}],'ignore_boxes':[],'negative_people':[]}
    target,collisions=build_targets([row]);flipped,_=build_targets([row],True)
    assert target[3].sum()==1 and collisions==0 and target[3][0,0,9,6]==1
    assert flipped[3][0,0,9,18]==1
    assert detection_metrics([row],pred)['ap']==1.
    assert len(decode(torch.full((1,5,24,24),-20.))[0])==0
    empty={'boxes':[],'ignore_boxes':[],'negative_people':[]}
    assert detection_metrics([empty],pred)['fp']==1
    ignored={'boxes':[],'ignore_boxes':[{'bbox_xyxy':[0,0,96,96]}],'negative_people':[]}
    assert detection_metrics([ignored],pred)['ignored_detections']==1
    marker=torch.zeros(1,1,96,96,dtype=torch.uint8);marker[0,0,8,12]=255
    shifted,shiftrows=translated_view(marker,[row],4,-4)
    assert shifted[0,0,4,16]==255 and shiftrows[0]['boxes'][0]['bbox_xyxy']==[16,4,40,60]
    translated_targets,_=build_targets(shiftrows)
    assert translated_targets[3][0,0,8,7]==1
    edge={'boxes':[{'bbox_xyxy':[0,0,20,40]}],'ignore_boxes':[],'negative_people':[]}
    rejected,rejectedrows=translated_view(marker,[edge],-4,0)
    assert torch.equal(rejected,marker) and rejectedrows[0]['boxes'][0]['bbox_xyxy']==[0,0,20,40]
    audit={'status':'PASS','data':counts,'cross_split_group_and_pixel_leakage':False,
           'png_npz_match':True,'bbox_transform_and_roundtrip':True,'synthetic_decode_flip_ap_ignore_tests':True,
           'spatial_augmentation_pixels_boxes_and_boundary_rejection':True}
    if not args.data_only:
        model=torch.jit.load(str(run/'model_float32.ts')).eval()
        result=json.loads((run/'results.json').read_text());x,rows=load_split(data,'test')
        cfg=json.loads((run/'config.json').read_text())
        saved=[json.loads(line) for line in (run/'test_predictions.jsonl').open()]
        checkpoint=torch.load(run/'model_float32.pt',map_location='cpu',weights_only=False)
        fused=PersonDetector().eval().fused();fused.load_state_dict(checkpoint['state_dict'])
        bn=PersonDetector().eval();bn.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=False)['state_dict'])
        prediction=[];error=0.
        with torch.inference_mode():
            for start in range(0,len(x),128):
                batch=x[start:start+128].float()/255
                exported=model(batch);original=bn(batch)
                error=max(error,float((exported-original).abs().max()))
                assert torch.allclose(exported,original,atol=4e-5,rtol=1e-4)
                prediction.extend(decode(exported,min_width=cfg.get('min_pred_width',8),min_height=cfg.get('min_pred_height',24)))
        max_box_error=0.
        for row,pred,sav in zip(rows,prediction,saved):
            assert row['source_id']==sav['source_id']
            expected=np.array(sav['all_candidates_96'],np.float32).reshape(-1,5)
            assert pred.shape==expected.shape
            if len(pred):max_box_error=max(max_box_error,float(np.max(np.abs(pred-expected))))
            assert np.allclose(pred,expected,atol=.002,rtol=1e-4)
        dm=detection_metrics(rows,prediction,confidence=result['threshold']);pm=presence_metrics(rows,prediction,result['threshold'])
        assert abs(pm['accuracy']-result['metrics']['test']['presence_at_selected_threshold']['accuracy'])<1e-9
        assert (dm['tp'],dm['fp'],dm['fn'])==tuple(result['metrics']['test']['detection_at_selected_threshold'][k] for k in ('tp','fp','fn'))
        with np.load(run/'weights_float32.npz',allow_pickle=False) as weights:
            for key,value in checkpoint['state_dict'].items():assert np.array_equal(weights[key],value.numpy())
        audit.update(export_all_test_frames=len(rows),max_abs_raw_output_error=error,max_abs_candidate_difference=max_box_error,
                     detection_and_presence_metrics_reproduced=True,float32_npz_equals_checkpoint=True)
    output=ROOT/'report/person_detector'/('data_validation.json' if args.data_only else 'model_validation.json')
    output.write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
