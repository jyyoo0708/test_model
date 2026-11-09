"""Inspect frozen model candidates for image 138 without changing inference settings."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'models'))
from advanced_utils import decode,representation

def inspect(model,x):
    with torch.inference_mode():
        out=model(x);heat=out[:,0:1].sigmoid();quality=out[:,5:6].sigmoid();score=heat*quality.sqrt()
        peaks=heat==torch.nn.functional.max_pool2d(heat,3,1,1)
        sv,ix=torch.topk((score*peaks).flatten(),20);v=representation(out)[0].reshape(4,-1)[:,ix]
        cx=(ix%24+v[0])*4;cy=(ix//24+v[1])*4;w=v[2]*96;h=v[3]*96
        boxes=torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],1).clamp(0,96)
        candidates=[]
        for j in range(len(ix)):
            q=boxes[j].tolist();cell=int(ix[j]);candidates.append(dict(score=float(sv[j]),heat=float(heat.flatten()[cell]),quality=float(quality.flatten()[cell]),
                box=q,width=q[2]-q[0],height=q[3]-q[1],passes_size=q[2]-q[0]>=8 and q[3]-q[1]>=24,cell_xy=[cell%24,cell//24]))
        filtered={str(t):decode(out,confidence=t)[0].tolist() for t in (.01,.05,.10,.15,.20,.25,.325)}
        return dict(max_heat=float(heat.max()),max_combined_any_cell=float(score.max()),candidates=candidates,threshold_predictions=filtered)

def main():
    torch.set_num_threads(8)
    path=ROOT/'result/frame_20261021_124417_138.jpg_input96.png'
    with Image.open(path) as im:im=im.copy();a=np.asarray(im).copy()
    x=torch.from_numpy(a)[None,None].float()/255
    result={}
    for name in ('student_kd','student_supervised'):
        model=torch.jit.load(str(ROOT/f'models/{name}/model_float32.ts')).eval();result[name]=inspect(model,x)
    from advanced_model import AdvancedDetector
    model=AdvancedDetector(2).eval();model.load_state_dict(torch.load(ROOT/'models/teacher/best.pt',map_location='cpu',weights_only=False)['state_dict'])
    result['teacher']=inspect(model,x)
    outdir=ROOT/'report/diagnosis_138';outdir.mkdir(exist_ok=True)
    (outdir/'candidates.json').write_text(json.dumps(result,indent=2))
    for t in (.1,.2,.325):
        canvas=im.copy();d=ImageDraw.Draw(canvas)
        for x1,y1,x2,y2,s in result['student_kd']['threshold_predictions'][str(t)]:
            d.rectangle((x1,y1,min(95,x2),min(95,y2)),outline=255,width=1)
        canvas.resize((576,576),Image.Resampling.NEAREST).save(outdir/f'threshold_{t}.png')
    for name,r in result.items():
        print(name,json.dumps(dict(max_heat=r['max_heat'],max_combined_any_cell=r['max_combined_any_cell'],top_candidates=r['candidates'][:6],counts={t:len(p) for t,p in r['threshold_predictions'].items()}),indent=2))
if __name__=='__main__':main()
