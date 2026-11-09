"""Final integration audit of selected weights, reproducibility metadata and local inference artifacts."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'models'))

def read(p):return json.loads((ROOT/p).read_text())
def main():
    torch.set_num_threads(4);checks=[]
    cfgs={}
    for stage in ('teacher','student_supervised','student_kd'):
        base=ROOT/'models'/stage;cfg=json.loads((base/'config.json').read_text());cfgs[stage]=cfg
        ck=torch.load(base/'best.pt',map_location='cpu',weights_only=False);h=json.loads((base/'history.json').read_text())
        assert abs(ck['validation_ap50']-max(r['val_ap50'] for r in h))<1e-6
        for p,sha in cfg['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha,p
        assert cfg['precision']=='float32'
        checks.append(f'{stage}: best checkpoint and source hashes valid')
    a,b=cfgs['student_supervised'],cfgs['student_kd']
    for k in ('initial_student_weights_sha256','initial_torch_rng_sha256','seed','epochs','data_config_sha256','teacher_sha256'):
        assert a[k]==b[k],k
    assert a['methods']==[1,2,3,4,5] and b['methods']==[1,2,3,4,5,7]
    checks.append('Student comparisons share initialization, RNG, data, teacher and epoch budget')
    mem=read('models/student_kd/memory.json')
    assert mem['network_total_bytes']==155428<175640<300000
    assert all(sum(s[k].values())<=mem['bank_bytes'] for s in mem['schedule'] for k in ('A','B'))
    checks.append('Two-bank network memory schedule within stated limits')
    inference_cfg=read('inference/config.json');assert 'student_kd' in inference_cfg['model']
    m=torch.jit.load(str(ROOT/inference_cfg['model'])).eval()
    with torch.inference_mode():raw=m(torch.zeros(1,1,96,96))
    assert raw.shape==(1,6,24,24) and raw.dtype==torch.float32 and torch.isfinite(raw).all()
    assert not any('adapter' in n for n,_ in m.named_parameters())
    checks.append('Deployment export is float32, six-channel stride4 student with no distillation adapter')
    summary=read('inference/results/summary.json');assert len(summary)==12
    for r in summary:
        name=Path(r['input']).stem
        with Image.open(ROOT/f'inference/results/{name}_input96.png') as im:assert im.size==(96,96) and im.mode=='L'
        with Image.open(ROOT/f'inference/results/{name}_bbox96.png') as im:assert im.size==(96,96)
        assert r['threshold']==inference_cfg['threshold'] and r['person_present']==bool(r['boxes_xyxy_score_96'])
        for q in r['boxes_xyxy_score_96']:
            assert 0<=q[0]<q[2]<=96 and 0<=q[1]<q[3]<=96 and q[4]>=r['threshold']
    checks.append('All 12 inference inputs and both bbox output sizes saved with consistent metadata')
    assert hashlib.sha256((ROOT/'baseline/person_detector/run02/model_float32.ts').read_bytes()).hexdigest()==hashlib.sha256((ROOT.parent/'models/person_detector/run02/model_float32.ts').read_bytes()).hexdigest()
    checks.append('Original baseline model and copied baseline still match')
    manifest={}
    for directory in ('models','scripts','inference','report'):
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    result=dict(status='PASS',checks=checks,inference_images=len(summary),sram_bytes=mem['network_total_bytes'],artifact_sha256=manifest)
    (ROOT/'logs/final_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},indent=2))
if __name__=='__main__':main()
