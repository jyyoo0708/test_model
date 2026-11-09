"""Audit array alignment, source separation, size/visibility policy and float-coordinate bounds."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    data=ROOT/'data/processed/advanced_detection';sets={};result={};eps=1e-8
    for s in ('train','val','test'):
        rows=[json.loads(x) for x in (data/f'{s}.jsonl').read_text().splitlines()];sets[s]={r['group_id'] for r in rows}
        with np.load(data/f'{s}.npz') as p:
            assert p['x'].shape==(len(rows),1,96,96) and p['x'].dtype==np.uint8
            assert np.array_equal(p['source_id'],[r['source_id'] for r in rows])
        min_vis=1.;min_ret=1.;n=0
        for r in rows:
            for b in r['boxes']:
                q=b['bbox_xyxy'];assert b['visible_ratio']>=.8 and b['retained_ratio']>=.8
                assert -eps<=q[0]<q[2]<=96+eps and -eps<=q[1]<q[3]<=96+eps
                assert q[2]-q[0]>=8-eps and q[3]-q[1]>=24-eps
                min_vis=min(min_vis,b['visible_ratio']);min_ret=min(min_ret,b['retained_ratio']);n+=1
        result[s]=dict(images=len(rows),people=n,min_visible=min_vis,min_retained=min_ret,sequence_counts=dict(Counter(r.get('sequence','crowdhuman') for r in rows)))
    for a,b in (('train','val'),('train','test'),('val','test')):assert not sets[a]&sets[b]
    result['status']='PASS';result['coordinate_tolerance']=eps
    (ROOT/'logs/data_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
