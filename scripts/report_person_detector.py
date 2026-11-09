#!/usr/bin/env python3
"""Generate detector report, structural diagram, bbox examples and figures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import zipfile
os.environ.setdefault('MPLCONFIGDIR','/tmp/person96-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'models/person_detector'))
from utils import detection_metrics
from infer_person_detector import draw_boxes


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=ROOT/'models/person_detector/run02');args=ap.parse_args()
    run=args.run;assets=ROOT/'report/person_detector';assets.mkdir(exist_ok=True,parents=True)
    read=lambda name:json.loads((run/name).read_text())
    results=read('results.json');cfg=read('config.json');memory=read('memory.json');history=read('history.json')
    audit_path=assets/'model_validation.json'
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else None
    data=ROOT/'data/processed/person96_detection'
    stats=json.loads((data/'stats.json').read_text())
    rows=[json.loads(line) for line in (data/'test.jsonl').open()]
    saved=[json.loads(line) for line in (run/'test_predictions.jsonl').open()]
    byid={r['source_id']:r for r in rows};predbyid={r['source_id']:r for r in saved}
    pct=lambda x:f'{x:.2%}'
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
    epochs=[r['epoch'] for r in history]
    axes[0].plot(epochs,[r['train_loss'] for r in history]);axes[0].set(xlabel='Epoch',ylabel='Mean training loss',title='Detector training loss')
    axes[1].plot(epochs,[r['val_ap50'] for r in history],label='Validation AP50')
    axes[1].axvline(results['best_epoch'],color='gray',linestyle=':',label='Selected checkpoint')
    axes[1].set(xlabel='Epoch',ylabel='101-point interpolated AP',ylim=(0,1),title='Full-frame detection validation')
    for ax in axes:ax.grid(alpha=.2)
    axes[1].legend();fig.savefig(assets/'learning_curves.png',dpi=200);fig.savefig(assets/'learning_curves.svg');plt.close(fig)
    # Examples are label-stratified random samples, not selected for good predictions.
    rng=random.Random(20261105)
    positive=rng.sample([r for r in rows if r['boxes']],6)
    negative=rng.sample([r for r in rows if not r['boxes']],6)
    examples=positive+negative
    examples_dir=ROOT/'output/person_detector/examples';examples_dir.mkdir(exist_ok=True,parents=True)
    canvas=Image.new('RGB',(3*360,4*300),'#eeeeee');d=ImageDraw.Draw(canvas)
    font=ImageFont.load_default(size=13)
    index=[]
    with zipfile.ZipFile(ROOT/'data/raw/crowdhuman/CrowdHuman_val.zip') as archive:
        for i,row in enumerate(examples):
            predicted=predbyid[row['source_id']]
            with archive.open(row['zip_member']) as stream,Image.open(stream) as image:original=image.convert('RGB')
            gt=[]
            for b in row['boxes']:
                x,y,w,h=b['source_fbox_xywh'];gt.append([max(0,x),max(0,y),min(original.width,x+w),min(original.height,y+h)])
            annotated=draw_boxes(original,gt,(30,255,60),'GT')
            annotated=draw_boxes(annotated,predicted['boxes_xyxy_score_original'],(255,50,30),'pred')
            path=examples_dir/(row['source_id']+'_bbox.jpg');annotated.save(path,quality=92)
            small=annotated.copy();small.thumbnail((354,266))
            x,y=(i%3)*360,(i//3)*300
            canvas.paste(small,(x+(360-small.width)//2,y+(270-small.height)//2))
            d.text((x+4,y+272),f'{i}: GT={len(gt)} pred={len(predicted["boxes_xyxy_score_original"])}',fill='black',font=font)
            d.text((x+4,y+286),row['source_id'][:36],fill='black',font=font)
            index.append({'index':i,'source_id':row['source_id'],'gt_count':len(gt),'predicted_count':len(predicted['boxes_xyxy_score_original']),'annotated_image':str(path.relative_to(ROOT))})
        # Standalone inference demo; fixed first random positive, without GT overlay.
        demo=positive[0]
        demo_input=ROOT/'output/person_detector/demo_input.jpg'
        with archive.open(demo['zip_member']) as stream:demo_input.write_bytes(stream.read())
    canvas.save(assets/'bbox_examples.jpg',quality=95)
    (assets/'bbox_examples_index.json').write_text(json.dumps(index,indent=2)+'\n')
    (assets/'demo_source.json').write_text(json.dumps({'source_id':demo['source_id'],'input':str(demo_input.relative_to(ROOT))},indent=2)+'\n')
    metrics=results['metrics'];test=metrics['test'];presence=test['presence_at_selected_threshold'];det=test['detection_at_selected_threshold']
    baseline=json.loads((ROOT/'models/person_detector/run01/results.json').read_text())
    bv=baseline['metrics']['val'];final_val=metrics['val']
    threshold=results['threshold']
    report=['# 사람 위치 검출 모델 재구성 및 학습 보고서','',
        '작성일: 2026-11-05 (Asia/Seoul)','',
        f"기존 패치 분류 모델을 별도 보존하고, **전체 영상 → 96×96 grayscale → 여러 사람 bbox와 점수**를 출력하는 검출 모델을 구현·학습했다. 최종 선택 모델은 `{run.relative_to(ROOT)}`의 epoch **{results['best_epoch']}**, score threshold **{threshold:.3f}**다.",'',
        f"최종 test 원본 {test['images']}장에서 **bbox AP50 {pct(test['ap50'])}**, AP75 {pct(test['ap75'])}, AP50:95 평균 {pct(test['map50_95'])}를 얻었다. 선택 임계값의 검출 precision/recall은 **{pct(det['precision'])}/{pct(det['recall'])}**, 영상 단위 사람 존재 여부 정확도는 **{pct(presence['accuracy'])}**다.",'',
        '**존재 여부 정확도와 bbox 검출 성능은 다른 지표다.** 이전 분류기의 96.70%는 잘라낸 패치 분류 성능이며 이번 전체 화면 검출 AP와 직접 비교할 수 없다. 좌표·그리기 기능과 실제 학습은 완료했지만, 아래 성능 범위를 넘는 검출 품질이나 실제 배포 성능을 보장하지 않는다.','',
        '## 입력·출력과 bbox 그리기','',
        '- 입력 영상은 종횡비를 유지해 긴 변을 96으로 resize하고, 남는 부분을 grayscale 114로 letterbox 처리한다. float32 `[B,1,96,96]`, 값 범위 `[0,1]`을 사용한다.',
        '- 모델 원시 출력은 `[B,5,24,24]`이며 채널은 중심 heatmap logit 1개, 중심 cell 내부 offset x/y 2개, 너비/높이 2개다.',
        '- sigmoid → heatmap 3×3 local maximum → top 50 → bbox 복원 → NMS IoU 0.5 → score threshold 순서로 최종 bbox 목록을 만든다.',
        f"- 예측 bbox의 최소 너비/높이는 96 입력에서 {cfg.get('min_pred_width',8):g}/{cfg.get('min_pred_height',24):g}px다. 정답 기준 8/24px보다 작은 여유를 허용해 경계 크기 대상의 회귀 오차로 인한 즉시 탈락을 줄인다. 정답 최소 크기 자체는 바꾸지 않았다.",
        '- bbox 좌표는 `[x1,y1,x2,y2]`이며 우측/하단 exclusive다. padding과 resize 비율을 역변환해 **원본 영상 픽셀 좌표**도 저장한다.',
        '- confidence 기준을 넘는 유효 bbox가 하나 이상 있으면 `person_present=true`, 없으면 `false`다. 좌표는 입력 영상 크기 안으로 제한한다.',
        '- raw TorchScript 모델에는 sigmoid·후처리·그리기가 포함되지 않는다. `scripts/infer_person_detector.py`가 후처리를 수행하고 bbox 이미지와 JSON을 함께 저장한다.','',
        '```text',
        'cx = (grid_x + sigmoid(offset_x)) × 4',
        'cy = (grid_y + sigmoid(offset_y)) × 4',
        'w  = sigmoid(size_w) × 96',
        'h  = sigmoid(size_h) × 96',
        'bbox96 = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]',
        'x_original = (x96 - pad_left) / scale_x',
        'y_original = (y96 - pad_top) / scale_y','```','',
        '## 실제 bbox 결과','',
        '**초록색은 적격 GT, 빨간색은 예측 bbox와 점수**다. test에서 seed 20261105로 적격 사람 존재/부재 영상 각각 6장을 무작위 추출했다. 성능이 좋은 사례만 고른 그림이 아니다. 사람 부재는 “조건을 충족하는 사람 GT가 없음”을 뜻하며 작은 사람이나 가려진 사람이 영상에 있을 수 있다.','',
        '![GT and predicted bounding boxes](person_detector/bbox_examples.jpg)','',
        '원본 크기의 개별 결과는 `output/person_detector/examples/`, 대응 source ID는 `report/person_detector/bbox_examples_index.json`에 있다. 추론 전용 예시는 `output/person_detector/demo_bbox.jpg`와 `demo_bbox.json`이다.','',
        '그림 7번은 사람을 실제로 포함하지만, 96 입력에서 bbox 높이가 약 20~24px이고 일부는 너비도 8px보다 작아, 최소 높이 24px 또는 너비 8px 조건에 미달해 GT=0이다. 이 영상에서 나온 예측 5개는 영상 단위 존재 여부 평가의 오검출로 집계된다. 이처럼 실제 사람의 존재와 이번 크기·가시성 조건을 충족하는 사람의 존재를 구분해야 한다.','',
        '## 데이터 재구성','',
        '기존 분류용 crop을 검출 학습에 그대로 사용하지 않고, 원본 이미지 4,370장을 전체 화면 기준으로 다시 변환했다. 기존 source/dHash 그룹 split을 유지해 train/val/test 사이 같은 원본이나 가까운 dHash 그룹이 섞이지 않게 했다.','',
        '| Split | 원본 전체 영상 | 적격 사람 bbox | 사람 존재 영상 | 조건 충족 사람 없는 영상 |','|---|---:|---:|---:|---:|']
    for split in ('train','val','test'):
        s=stats['splits'][split]
        report.append(f"| {split} | {s['images']:,} | {s['people']:,} | {s['positive_images']:,} | {s['negative_images']:,} |")
    report += ['',
        '정답은 `tag=person`, source ignore 아님, 원본 높이/너비 ≥48/12px, 전신 유지 비율 ≥0.8, 가시 영역 bbox 면적 비율 ≥0.8, 실제 96 입력에서 bbox 높이/너비 ≥24/8px를 모두 만족한다. 정수 resize 크기를 사용해 scale_x/scale_y를 따로 계산하므로 초기 근사 집계와 몇 명 차이가 있다.','',
        '80%는 `area(vbox ∩ fbox ∩ 원본 영상) / area(fbox)`의 bbox 기반 근사치다. 실제 신체 부위나 segmentation의 80%를 측정한 값이 아니다. 알려진 큰 사람 중 가시성·전신 유지 비율 미달 대상은 적격 GT에서 제외하고 음성 중심으로 학습한다. source ignore/mask는 loss에서 무시한다. 최종 run에서는 알려진 작은 사람 영역도 배경/음성 heatmap supervision을 적용해 사람 존재 여부의 오검출을 줄였다.','',
        '평가에서는 작은 사람·source ignore 영역에 대한 unmatched 예측의 prediction-area overlap(IoA)이 0.5 이상이면 해당 예측을 검출 TP/FP에서 제외한다. 알려진 큰 사람의 가시성 미달은 무시 영역이 아니므로 오검출로 집계된다. 영상 단위 존재 여부 평가는 GT를 이용해 예측을 지우지 않고 런타임 bbox 목록 그대로 판단한다.','',
        '출력 격자의 같은 cell에 중심이 겹친 GT는 큰 bbox가 회귀 정답을 차지한다. 모든 GT는 평가에 남긴다. 격자 충돌과 인접 중심의 local maximum 억제는 밀집 장면의 recall 한계다. 원본 방향의 train 중심 충돌은 77/7,972개다.','',
        '## 네트워크 구조','',
        '중심점과 크기를 회귀하는 [Objects as Points](https://arxiv.org/abs/1904.07850)의 접근을 참고한 자체 소형 모델이다. 원 논문의 backbone이나 성능을 재현한 모델은 아니다. ConvBlock은 Conv3×3 → BatchNorm → ReLU이며, Conv bias는 학습 시 없고 추론 export 시 BN을 Conv로 fold한다.','',
        '```mermaid','flowchart TD',
        '    I["1×96×96"] --> A["Conv s2: 8×48×48"]',
        '    A --> B["Conv s2: 16×24×24"]',
        '    B --> C["Conv s2: 24×12×12"]',
        '    C --> D["Conv s2: 32×6×6"]',
        '    D --> E["Context Conv: 32×6×6"]',
        '    E --> U1["Nearest upsample ×2"]',
        '    U1 --> J1["Concat: 56×12×12"]',
        '    C --> J1',
        '    J1 --> F1["Conv: 24×12×12"]',
        '    F1 --> U2["Nearest upsample ×2"]',
        '    U2 --> J2["Concat: 40×24×24"]',
        '    B --> J2',
        '    J2 --> F2["Conv: 16×24×24"]',
        '    F2 --> H["Head Conv: 16×24×24"]',
        '    H --> P["1×1 Conv: heatmap 1ch"]',
        '    H --> O["1×1 Conv: offset 2ch"]',
        '    H --> S["1×1 Conv: size 2ch"]',
        '    P --> R["Concat raw output: 5×24×24"]',
        '    O --> R','    S --> R','```','',
        '| 계층 | 출력 (batch 제외) | Weight | Bias (fold 후) | MACs |','|---|---|---:|---:|---:|']
    for layer in memory['layers']:
        report.append(f"| {layer['name']} | {'×'.join(map(str,layer['shape'][1:]))} | {layer['weight_count']:,} | {layer['bias_count']:,} | {layer['macs']:,} |")
    report += ['',f"BN fold 후 파라미터 **{memory['parameters']:,}개**, 입력당 **{memory['macs']:,} MACs**다. 마지막 FC 대신 공간별 head를 사용해 여러 bbox를 출력한다.",'',
        '## 학습과 선택 과정','',
        f"최종 run은 float32 CPU {cfg['threads']} threads, batch {cfg['batch_size']}, seed {cfg['seed']}, AdamW(weight decay 0.0001), 초기 LR {cfg['lr']}로 학습했다. CosineAnnealingLR의 최저 LR은 0.00002, 최대 {cfg['epochs']} epochs이며, 최소 30 epoch 이후 validation AP50 개선이 15 epoch 없으면 early stop한다. 실제 {results['epochs_run']} epochs, 로딩·최종 평가·export 포함 {results['wall_seconds']/60:.2f}분을 실행했다.",'',
        'Loss는 `중심 heatmap focal + offset L1 + 4×size L1 + 2×GIoU`다. 회귀는 GT 중심 cell에서만 수행한다. train augmentation은 양쪽 좌우 방향, 밝기/대비 0.85~1.15, 최대 ±4px 평행 이동이다. 이동으로 적격 GT가 잘리는 영상은 그 이동을 취소한다. bbox·ignore 영역도 픽셀과 함께 이동한다. 5개 이동 후보 × 좌우 2방향을 미리 만들고, epoch당 각 원본의 두 방향에서 이동 후보를 무작위 선택한다. epoch당 학습 영상 수는 6,992개다.','',
        '최종 run의 처음 두 Conv/BN block은 이전 분류기의 동일 train source로 학습한 checkpoint에서 초기화했다. 나머지 계층은 무작위 초기화했다. 분류기의 FC를 붙여 bbox를 만드는 방식은 아니며, 전체 검출 네트워크를 재학습했다.','',
        '| 실험 | Validation AP50 | Validation 검출 F1 | Validation 존재 여부 정확도 |','|---|---:|---:|---:|',
        f"| run01: 기본 중심점 검출 | {pct(bv['ap50'])} | {pct(bv['detection_at_selected_threshold']['f1'])} | {pct(bv['presence_at_selected_threshold']['accuracy'])} |",
        f"| {run.name}: 음성 supervision·이동 증강·초기화 보완 | {pct(final_val['ap50'])} | {pct(final_val['detection_at_selected_threshold']['f1'])} | {pct(final_val['presence_at_selected_threshold']['accuracy'])} |",'',
        '보완 방향과 모델 선택은 validation 지표에 근거했다. run01도 자동 학습 스크립트에서 test 평가 파일을 생성했으므로 test를 단 한 번만 사용한 연구 설계라고 주장하지 않는다. run02의 학습·checkpoint·score 선택에는 test 지표를 사용하지 않았다. 여러 변경을 함께 적용했으므로 개별 변경의 인과 효과를 분리한 ablation 실험도 아니다.','',
        '증강 구현 중 빈 배치가 반복되는 오류를 발견한 중간 실행은 중단하고 루프를 수정해 run02를 처음부터 다시 학습했다. 중단 실행은 `run02_interrupted`에 보존했으며 최종 모델·실험 비교에서 제외했다.','',
        '![Detection learning curves](person_detector/learning_curves.png)','',
        'Checkpoint는 validation AP50 최대값으로 선택했다. score threshold는 validation의 IoU0.5 검출 F1 최대값으로 선택했으며 동점이면 존재 여부 balanced accuracy, 더 높은 score 순으로 정했다.','',
        '## 최종 평가','',
        'AP는 단일 person class, confidence floor 0.01, 최대 50개, NMS 0.5, greedy 1:1 IoU 매칭, recall 101점 precision interpolation을 사용한 자체 evaluator 수치다. AP50:95는 IoU 0.50~0.95를 0.05 간격으로 계산한 AP 평균이다. 공식 CrowdHuman/COCO evaluator의 전체 프로토콜과 동일하지 않다.','',
        '| Split | AP50 | AP75 | AP50:95 | 검출 precision | 검출 recall | 검출 F1 | 존재 여부 accuracy |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('train','val','test'):
        m=metrics[split];dd=m['detection_at_selected_threshold'];pp=m['presence_at_selected_threshold']
        report.append(f"| {split} | {pct(m['ap50'])} | {pct(m['ap75'])} | {pct(m['map50_95'])} | {pct(dd['precision'])} | {pct(dd['recall'])} | {pct(dd['f1'])} | {pct(pp['accuracy'])} |")
    report += ['',f"Test bbox TP/FP/FN은 **{det['tp']}/{det['fp']}/{det['fn']}**이며, 무시 영역과 겹쳐 평가에서 제외한 예측은 {det['ignored_detections']}개다. 영상 단위 존재 여부 confusion matrix(행 실제, 열 예측; 0/1 순서)는 `{presence['confusion_true_rows_pred_columns']}`다. 다수 클래스 기준선은 {pct(presence['majority_baseline'])}다.",'',
        f"목표 70~80%에 대한 해석: 영상 단위 존재 여부 정확도는 {pct(presence['accuracy'])}이며, bbox AP50은 {pct(test['ap50'])}다. AP를 분류 정확도로 표현하거나 bbox 품질이 70~80%에 도달했다고 해석하면 안 된다.",'',
        '## SRAM과 export 검증','',
        '학습·배포 파일은 float32이며, 아래는 향후 int8 weight/activation, int32 bias/accumulator, batch=1, im2col 없는 직접 convolution을 가정한 SRAM 추정이다. upsample·concat·skip feature를 포함해 모든 출력이 남아 있다고 계산하고 후처리·기타 작업 공간 16KiB를 더했다.','',
        '| 항목 | Bytes |','|---|---:|']
    for key,value in memory['parts'].items():report.append(f'| {key} | {value:,} |')
    report += [f"| **합계** | **{memory['total_bytes']:,}** |",'',
        f"약 **{memory['total_bytes']/1000:.2f}KB**로 300,000 bytes 이내이며 약 {(300000-memory['total_bytes'])/1000:.2f}KB 여유다. 실제 int8 모델·정수 정확도·RTL/SRAM 배치는 구현·측정하지 않았다. 정렬, 포트, requantization, 후처리 구현에 따라 재검증해야 한다.",'',
        f"BN fold 전후 원시 출력의 최대 차이는 `{results['bn_fold_max_abs_error']:.8g}`, TorchScript 저장/재로딩 차이는 `{results['export_reload_max_abs_error']:.8g}`다. 전체 test export 재현 검사는 `report/person_detector/model_validation.json`, 데이터 전수/좌표 역변환/synthetic 검사는 `data_validation.json`에 기록한다.",'',
        (f"저장 모델·데이터 검사 결과는 **{audit['status']}**다. test {audit['export_all_test_frames']}장 전체에서 원래 모델과 저장 모델의 bbox·존재 여부 평가 수치가 재현됐고, 원시 출력 최대 차이는 `{audit['max_abs_raw_output_error']:.8g}`, 후보 bbox/score 최대 차이는 `{audit['max_abs_candidate_difference']:.8g}`였다. PNG/NPZ 일치, 원본 좌표 왕복, 이동 증강의 픽셀/GT 일치와 경계 이동 취소, synthetic decode/AP/ignore 및 float32 배열 일치도 확인했다." if audit else '검증 스크립트 실행 후 model_validation.json에서 전체 결과를 확인할 수 있다.'),'',
        '원본 JPEG를 직접 넣는 CLI에서도 예제의 bbox 4개가 저장된 평가 좌표와 일치함을 확인했다. 선택된 test bbox의 원본 좌표에 너비/높이 0 이하인 경우는 없었고, 학습 소스 코드 SHA-256도 config와 일치했다. 기록은 `report/person_detector/cli_validation.json`에 있다.','',
        '## 결과물과 실행 방법','',
        f"- 모델 코드: `models/person_detector/model.py`, 후처리·좌표·평가: `utils.py`.",
        f"- 최종 모델: `{run.relative_to(ROOT)}/model_float32.ts`.",
        f"- 학습 checkpoint·fold state dict·배열: 같은 폴더의 `best.pt`, `model_float32.pt`, `weights_float32.npz`.",
        '- 검출용 입력·annotation: `data/processed/person96_detection/{train,val,test}.npz`, `{train,val,test}.jsonl`.',
        f"- 모든 영상의 원본/96 bbox·score·존재 여부: `{run.relative_to(ROOT)}/{{train,val,test}}_predictions.jsonl`.",
        '- bbox 그림: `output/person_detector/examples/`, 추론 이미지/JSON: `output/person_detector/demo_bbox.jpg`, `demo_bbox.json`.',
        '- 설정·epoch 기록·수치: run 폴더의 `config.json`, `history.json`, `results.json`, `memory.json`.',
        '- 데이터/학습 로그: `report/person_detector/prepare.log`, `train.log`(run01), `train_run02.log`.', '',
        '이 시스템은 Cadence `LD_LIBRARY_PATH`와 PyTorch 라이브러리가 충돌하므로 실행 프로세스에서만 해당 변수를 제거한다. Python 환경은 이전 학습에서 설치한 `/tmp/person96-train-venv`와 `requirements-train.txt`를 사용했다.','',
        '```bash',
        '# 임의의 원본 이미지에서 bbox 그림과 원본 좌표 JSON 생성',
        'env -u LD_LIBRARY_PATH /tmp/person96-train-venv/bin/python scripts/infer_person_detector.py output/person_detector/demo_input.jpg --output output/person_detector/demo_bbox.jpg',
        '# 전체 데이터 및 저장 모델 검사',
        f'env -u LD_LIBRARY_PATH /tmp/person96-train-venv/bin/python scripts/validate_person_detector.py --run {run.relative_to(ROOT)}',
        '# 재학습: 기존 폴더를 덮어쓰지 않도록 새 output 사용',
        'env -u LD_LIBRARY_PATH /tmp/person96-train-venv/bin/python scripts/train_person_detector.py --output models/person_detector/reproduce01 --epochs 80 --lr 0.001 --spatial-augment --supervise-known-small --initialize-classifier --min-pred-width 6 --min-pred-height 20',
        '```','',
        '## 성능 한계와 다음 개선점','',
        '96×96 전체 화면에서는 많은 사람이 정보 손실로 작아진다. 정답에서 제외된 작은 사람 수가 많고, 군중의 인접 중심·한 cell 중복·가림 때문에 모든 사람을 찾지 못한다. 80% 가시성의 bbox 근사와 실제 신체 가시성 차이도 남아 있다.','',
        '비사람 영상은 원래 CrowdHuman의 사람이 포함된 장면에서 적격 bbox가 없는 경우다. 실제 사람 없는 배경의 광범위한 다양성을 대표하지 않는다. bbox 결과를 확인한 뒤 실제 카메라의 빈 장면·부분 신체·조명 변화 데이터를 추가하고, 필요 시 더 높은 입력 해상도나 다중 영역 탐색을 비교해야 한다. 해상도를 바꾸면 SRAM 제약도 다시 계산해야 한다.','',
        'CrowdHuman 원본 및 파생 이미지의 비상업 연구·교육 목적/재배포 금지 조건은 [공식 이용 조건](https://www.crowdhuman.org/download.html)과 기존 데이터 준비 보고서를 따른다.','']
    (ROOT/'report/person_detector_training.md').write_text('\n'.join(report))
    # Keep final inference metadata discoverable without editing the old classifier.
    (ROOT/'models/person_detector/selected.json').write_text(json.dumps({'run':run.name,'threshold':threshold,'selection':'validation AP50'},indent=2)+'\n')
    files=[run/n for n in ('best.pt','model_float32.pt','model_float32.ts','weights_float32.npz','results.json')]
    (run/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in files))
    print(ROOT/'report/person_detector_training.md')


if __name__=='__main__':main()
