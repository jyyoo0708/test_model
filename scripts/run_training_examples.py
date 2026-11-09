#!/usr/bin/env python3
"""Save actual 96x96 grayscale inputs and 96x96 bbox inference results."""
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'models/person_detector'))
from utils import decode


def main():
    output=ROOT/'test'
    output.mkdir(exist_ok=False)
    for category in ('person','no_person'):
        (output/'original'/category).mkdir(parents=True)
        (output/'result'/category).mkdir(parents=True)
    selection=json.loads((ROOT/'models/person_detector/selected.json').read_text())
    run=ROOT/'models/person_detector'/selection['run']
    config=json.loads((run/'config.json').read_text())
    threshold=json.loads((run/'results.json').read_text())['threshold']
    train=[json.loads(line) for line in (ROOT/'data/processed/person96_detection/train.jsonl').open()]
    train_ids={r['source_id'] for r in train}
    rng=random.Random(20261105)
    positives=[r for r in train if r['boxes']]
    rng.shuffle(positives)
    chosen=[];groups=set();sources=set()
    for row in positives:
        if row['group_id'] in groups:continue
        chosen.append(('person',row));groups.add(row['group_id']);sources.add(row['source_id'])
        if len(chosen)==10:break
    negatives=[json.loads(line) for line in (ROOT/'data/processed/person96/train.jsonl').open()]
    negatives=[r for r in negatives if r.get('negative_kind')=='background' and r['source_id'] in train_ids]
    rng.shuffle(negatives)
    for row in negatives:
        if row['source_id'] in sources or row['group_id'] in groups:continue
        chosen.append(('no_person',row));groups.add(row['group_id']);sources.add(row['source_id'])
        if len(chosen)==20:break
    assert len(chosen)==20
    # Freeze selections before model inference; predictions never affect selection.
    (output/'selection.json').write_text(json.dumps([{'category':c,'source_id':r['source_id'],'sample_id':r.get('sample_id')} for c,r in chosen],indent=2)+'\n')
    torch.set_num_threads(1)
    model=torch.jit.load(str(run/'model_float32.ts'),map_location='cpu').eval()
    manifest=[];counts={'person':0,'no_person':0}
    previews={c:Image.new('RGB',(5*112,2*126),'#eeeeee') for c in counts}
    with torch.inference_mode():
        for category,row in chosen:
            counts[category]+=1
            number=counts[category]
            stem=f'{number:02d}_{row.get("sample_id",row["source_id"])}'
            original=output/'original'/category/(stem+'.png')
            result=output/'result'/category/(stem+'.png')
            if category=='person':
                source=ROOT/'data/processed/person96_detection'/row['image']
                shutil.copyfile(source,original)
                provenance={'dataset':'person96_detection','split':'train','sample_kind':'letterboxed_full_frame_96',
                            'eligible_gt_count':len(row['boxes']),'zip_member':row['zip_member'],
                            'training_image':str(source.relative_to(ROOT)),
                            'selection_label':1}
            else:
                source=ROOT/'data/processed/person96'/row['image']
                shutil.copyfile(source,original)
                assert hashlib.sha256(original.read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest()
                provenance={'dataset':'person96 (earlier classifier training data)','split':'train',
                            'sample_kind':'background_patch','selection_label':0,
                            'training_image':str(source.relative_to(ROOT)),'crop_xyxy':row['crop_xyxy'],
                            'note':'Background patch used by earlier classifier; source frame belongs to detector train. This exact patch was not an individual detector training example.'}
            assert original.read_bytes()==source.read_bytes()
            with Image.open(original) as saved:
                assert saved.mode=='L' and saved.size==(96,96)
                image=saved.copy()
            # Read the saved PNG directly: these exact pixels enter the network.
            inputs=torch.from_numpy(np.asarray(image,dtype=np.float32).copy()).view(1,1,96,96)/255.
            candidates=decode(model(inputs),confidence=threshold,
                min_width=config.get('min_pred_width',8),min_height=config.get('min_pred_height',24))[0]
            boxes=candidates
            annotated=image.convert('RGB');box_draw=ImageDraw.Draw(annotated)
            for b in boxes:
                x1,y1,x2,y2=b[:4]
                # PIL rectangles include the last pixel; stored xyxy coordinates do not.
                raster=[max(0,int(np.floor(x1))),max(0,int(np.floor(y1))),
                        min(95,int(np.ceil(x2))-1),min(95,int(np.ceil(y2))-1)]
                box_draw.rectangle(raster,outline=(255,0,0),width=1)
            annotated.save(result)
            record={'category':category,'number':number,'source_id':row['source_id'],
                    'group_id':row['group_id'],'original':str(original.relative_to(ROOT)),
                    'result':str(result.relative_to(ROOT)),'width':image.width,'height':image.height,
                    'coordinate_format':'xyxy on the 96x96 model input; right/bottom exclusive',
                    'model':str((run/'model_float32.ts').relative_to(ROOT)),'threshold':threshold,
                    'person_present':bool(len(boxes)),'count':len(boxes),'provenance':provenance,
                    'detections':[{'bbox_xyxy':b[:4].tolist(),'score':float(b[4]),'bbox_xyxy_96':p[:4].tolist()} for b,p in zip(boxes,candidates)]}
            result.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
            manifest.append(record)
            tile=annotated
            xx=((number-1)%5)*112;yy=((number-1)//5)*126
            previews[category].paste(tile,(xx+8,yy))
            draw=ImageDraw.Draw(previews[category]);font=ImageFont.load_default(size=10)
            draw.text((xx+2,yy+98),f'{number:02d}: pred={int(bool(len(boxes)))} n={len(boxes)}',fill='black',font=font)
            draw.text((xx+2,yy+112),'96x96',fill='black',font=font)
    for category,preview in previews.items():preview.save(output/f'preview_{category}.jpg',quality=95)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (output/'summary.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['category','number','source_id','expected_presence','predicted_presence','bbox_count','original','result'])
        for r in manifest:writer.writerow([r['category'],r['number'],r['source_id'],r['provenance']['selection_label'],int(r['person_present']),r['count'],r['original'],r['result']])
    summary={'model':str(run.relative_to(ROOT)),'threshold':threshold,'seed':20261105,
             'input_shape':[1,1,96,96],'original_image_mode':'L','result_image_size':[96,96],
             'counts':{c:{'images':sum(r['category']==c for r in manifest),
                          'predicted_person_present':sum(r['category']==c and r['person_present'] for r in manifest)} for c in counts},
             'scope':'20 inference demonstrations; mixed detector-train full frames and classifier-train background patches, not a held-out performance estimate.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'README.md').write_text('''# 학습 데이터 20장 추론 예시

`original/person/`: 검출 train의 실제 96×96 grayscale 입력 10장.
`original/no_person/`: 기존 분류 train의 실제 96×96 grayscale 배경 입력 10장.
`result/person/`, `result/no_person/`: 96×96 입력 위에 빨간 bbox를 그린 PNG와 좌표·점수 JSON.

original의 PNG를 그대로 읽어 float32로 바꾸고 255로 나눈 뒤 네트워크에 넣었다.
original은 모두 96×96 PNG mode L이고, result도 모두 96×96이다.
결과의 배경은 같은 grayscale 픽셀이며 빨간 박스 표현을 위해 RGB로 저장했다.
JSON의 bbox_xyxy와 bbox_xyxy_96는 모두 이 96×96 입력 좌표다.

검출 원본 CrowdHuman은 모두 사람을 포함하므로, 적격 bbox가 0인 전체 영상을
“실제 사람이 없는 사진”으로 표시하지 않았다. no_person 10장은 기존 분류 학습에
사용한 배경 패치를 원본 픽셀 그대로 복사했으며, 해당 패치의 source도 검출 train에 속한다.
**이 배경 패치 자체는 검출기의 개별 학습 입력이 아니었다.**

선택은 seed 20261105로 모델 추론 전에 고정했다. 사람이 있는 10장은 모델의
크기/가시성 정답 기준을 만족한다. 결과에는 실제 예측 bbox만 빨간색으로 표시하며,
bbox가 없으면 결과 영상에 사각형이 없다. 96×96 배경 패치의 좌표는 그 패치 기준이다.

`preview_person.jpg`, `preview_no_person.jpg`로 각 10장 결과를 한눈에 볼 수 있다.
`summary.csv`, `summary.json`, `manifest.json`에는 결과 요약, 원본 경로와 출처가 있다.
원본과 결과 번호는 1:1로 대응한다. 혼합된 학습 출처의 추론 예시이며 독립 평가 정확도가 아니다.
CrowdHuman 이미지의 교육/비상업 연구 및 재배포 제한을 따른다.
''')
    assert sum(1 for p in (output/'original').rglob('*') if p.is_file())==20
    assert len(list((output/'result').rglob('*.png')))==20
    assert len(list((output/'result').rglob('*.json')))==20
    for r in manifest:
        with Image.open(ROOT/r['original']) as a,Image.open(ROOT/r['result']) as b:
            assert a.mode=='L' and a.size==b.size==(96,96)
        for detection in r['detections']:
            x1,y1,x2,y2=detection['bbox_xyxy']
            assert 0<=x1<x2<=96 and 0<=y1<y2<=96
    (output/'validation.json').write_text(json.dumps({'status':'PASS','original_images':20,
        'result_images':20,'all_originals_96x96_grayscale':True,'all_results_96x96':True,
        'saved_input_bytes_equal_training_images':True,'inference_directly_from_saved_png':True,
        'bbox_coordinates_within_96x96':True},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
