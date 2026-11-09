"""Generate local experiment report, learning curves and inference contact sheet."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads((ROOT/p).read_text())
def pc(v):return f'{100*v:.2f}%'
def main():
    out=ROOT/'report';out.mkdir(exist_ok=True)
    baseline=read('baseline/evaluation/results.json');supervised=read('models/student_supervised/results.json');kd=read('models/student_kd/results.json')
    stats=read('data/processed/advanced_detection/stats.json');memory=read('models/student_kd/memory.json')
    table=['| 모델 | 존재 정확도 | AP50 | AP75 | 검출 precision | 검출 recall |','|---|---:|---:|---:|---:|---:|']
    for label,result in [('기존 run02',baseline),('개선 학생 증류 없음',supervised),('개선 학생 증류 적용',kd)]:
        m=result['metrics']['crowdhuman_test'];table.append('| '+' | '.join([label,pc(m['presence']['accuracy']),pc(m['ap50']),pc(m['ap75']),pc(m['detection']['precision']),pc(m['detection']['recall'])])+' |')
    extra=['| 모델 | MOT17 10 존재 정확도 | AP50 | AP75 | 검출 recall |','|---|---:|---:|---:|---:|']
    for label,result in [('기존 run02',baseline),('개선 학생 증류 없음',supervised),('개선 학생 증류 적용',kd)]:
        m=result['metrics']['mot17_test'];extra.append('| '+' | '.join([label,pc(m['presence']['accuracy']),pc(m['ap50']),pc(m['ap75']),pc(m['detection']['recall'])])+' |')
    hist={name:read(f'models/{name}/history.json') for name in ('teacher','student_supervised','student_kd')}
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,h in hist.items():
        axes[0].plot([r['epoch'] for r in h],[r['val_ap50'] for r in h],label=name)
        axes[1].plot([r['epoch'] for r in h],[r['loss'] for r in h],label=name)
    axes[0].set(xlabel='Epoch',ylabel='Validation AP50',ylim=(0,1));axes[1].set(xlabel='Epoch',ylabel='Training loss')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)
    images=sorted((ROOT/'inference/results').glob('*_bbox.jpg'))
    sheet=Image.new('RGB',(4*320,((len(images)+3)//4)*230),'white');draw=ImageDraw.Draw(sheet)
    for i,p in enumerate(images):
        with Image.open(p) as im:thumb=ImageOps.contain(im,(320,205));sheet.paste(thumb,((i%4)*320,(i//4)*230))
        draw.text(((i%4)*320+4,(i//4)*230+208),p.stem[:44],fill='black')
    sheet.save(out/'inference_contact_sheet.jpg')
    bests='\n'.join(f'- `{name}`: best epoch {read(f"models/{name}/training_result.json")["best_epoch"]}, {len(h)} epochs, {sum(r["seconds"] for r in h):.1f}s.' for name,h in hist.items())
    old_test=baseline['metrics']['crowdhuman_test'];new_test=kd['metrics']['crowdhuman_test']
    conclusion=f"기존 모델 대비 독립 테스트 AP50은 {pc(old_test['ap50'])}→{pc(new_test['ap50'])}, AP75는 {pc(old_test['ap75'])}→{pc(new_test['ap75'])}로 높아졌다. 존재 여부 정확도는 {pc(old_test['presence']['accuracy'])}→{pc(new_test['presence']['accuracy'])}로 낮아졌다. 박스 검출 지표와 존재 판정 지표를 구분해서 사용해야 한다."
    text=f'''# 사람 검출 모델 개선 학습 결과

문서의 1, 2, 3, 4, 5, 7번을 적용해 교사와 학생을 floating point로 학습하고 추론했다. 모든 신규 데이터·코드·모델·로그·결과는 `advance_network` 안에 있다. 기존 자료는 복사본을 사용했다.

{conclusion}

## 기존 독립 테스트 결과

원래 CrowdHuman test 437장을 유지했다. 체크포인트와 score threshold는 원래 CrowdHuman validation에서만 선택했고 test로 조정하지 않았다. AP는 score floor 0.01에서 계산했다. 존재 정확도는 threshold를 넘는 예측 bbox가 하나 이상 있는지로 계산하며 정답 ignore 영역으로 예측을 지우지 않는다.

{chr(10).join(table)}

기존 모델은 복사한 TorchScript를 다시 추론했다. 기존 추론 기준인 최소 예측 폭 6·높이 20px, threshold 0.329를 유지했다. 새 학생은 최소 예측 폭 8·높이 24px, threshold {kd['threshold']:.3f}를 사용한다. 정답 최소 크기는 모든 평가에서 폭 8·높이 24px다. 구조·학습·예측 크기 필터 차이가 함께 반영된 비교다.

## 추가 영상 평가

{chr(10).join(extra)}

추가 데이터는 [MOT17 공식 원본](https://motchallenge.net/data/MOT17/)에서 받았다. MOT17 02·04는 train, 09는 validation, 10은 test로 영상 전체를 분리했다. FRCNN 버전의 원본 이미지와 수작업 정답만 사용했으며 FRCNN 검출 결과는 학습에 사용하지 않았다. 프레임은 30장 간격으로 뽑고, 전체 화면과 고정 격자 타일 6개를 만들었다. 추가 train 385개 뷰는 원본 55프레임에서 생성했으며, 추가 validation은 원본 18프레임의 126개 뷰, test는 원본 22프레임의 154개 뷰다. 이번에 새로 다운로드한 데이터셋은 MOT17 한 종류다. 같은 프레임·영상에서 나온 타일은 같은 split에만 둔다. MOT17 test는 한 영상의 파생 이미지이므로 다양한 카메라에 대한 일반화 지표로 확대 해석하지 않는다.

```json
{json.dumps(stats,indent=2,ensure_ascii=False)}
```

추가 데이터의 가시율은 `max(0, official_visibility - (1-retained))`를 써서 보수적으로 판정한다. 기존 CrowdHuman은 원래 bbox 기반 가시율 정책을 유지한다. 실제 사용자 CCTV 10장과 test1·test2는 정답이 없는 정성 추론용이며 학습과 성능 지표에 사용하지 않았다. 정성 결과에서는 test2에 전신에 가까운 박스 한 개가 남았지만, 회의실 의자 주변 오검출과 일부 CCTV 사람 미검출도 남아 있다. 정성 결과 모음에서 이 오류들을 직접 확인할 수 있다.

## 적용 내용

1. 데이터 확대: MOT17 실제 감시·거리 영상, 전체 화면·타일의 사람 및 배경을 추가했다. 원래 CrowdHuman train/val/test 그룹은 유지했다.
2. 증강: scale 0.85~1.18, 이동 ±10px, 좌우 반전, 밝기·대비 0.7~1.3, gamma 0.8~1.2, 약한 blur/noise. 변환 후 가시율·유지율 80% 및 최소 크기를 다시 판정한다. 잘린 가시율은 제거 면적 전체가 가시 영역이었다고 보는 하한으로 계산한다.
3. 채널 확대: 12→24→32→48 backbone, 32→24 decoder. 입력 96×96 grayscale, 출력 24×24를 유지한다.
4. Head: shared 분류 경로와 별도 24→24 회귀 tower, 1채널 IoU quality head. 품질 정답은 gradient를 끊은 예측/GT IoU이며, 배경 품질 BCE도 학습한다. 최종 점수는 `sigmoid(heat)*sqrt(sigmoid(quality))`다.
5. 박스 손실·할당: smooth L1 offset/size와 DIoU를 사용한다. 기본 중심 셀 외에 표현 가능한 인접 셀에 회귀 정답을 주며 중심 셀 우선으로 충돌을 해결한다. Offset은 `2*sigmoid-0.5`, decode도 같은 표현을 쓴다. 분류 heatmap의 정점은 원래 중심 셀 하나에 둔다.
7. 증류: 학생의 모든 채널을 2배로 늘린 교사를 동일한 적격 정답으로 학습했다. 적격 GT 주변 heatmap·feature와 교사 quality>0.35인 할당 셀의 bbox를 모방한다. 교사는 동결하며 feature adapter 24→48은 학습에만 사용한다. Annotation 없는 영역에 pseudo-positive를 추가하지 않는다.

증류 없는 학생과 증류 학생은 같은 초기 학생 가중치·seed·증강 난수·batch 순서·epoch 상한·모델 선택 규칙을 사용했다. 단일 seed 비교이므로 seed에 따른 편차는 이 실험에서 측정하지 않았다. 6번 출력 격자 확대와 8번 후처리 공동 탐색은 적용하지 않았다. Score threshold 한 개의 validation 선택은 기존 학습 절차를 따른다.

## 학습 및 검증

{bests}

- CPU, float32. 학생 MACs: {memory['macs']:,}, threads8 배치1 TorchScript 시간: {kd['cpu_latency_ms_batch1_threads8']:.3f}ms.
- 새 좌표 표현·인접 할당·증강 정답 재판정·빈 정답 batch·유한 gradient·교사 동결 검사를 통과했다. `logs/behavior_checks.log` 참조.
- BN folding 최대 차이 {kd['export_validation']['bn_fold_max_abs_error']:.8g}, TorchScript 저장·재로드 최대 차이 {kd['export_validation']['reload_max_abs_error']:.8g}.
- AP75, edge/center·단일/다중·크기별 recall, bbox 중심/폭/높이 오차, 같은 GT의 중복 예측 수는 각 모델의 `results.json`에 있다. 중복 수는 단순 진단 집계다.

## SRAM 산정

Activation/weight int8, bias int32, 배치1, output stationary PE 레지스터 누산을 가정한다. 실제 int8 추론이나 RTL을 구현·검증한 결과는 아니다.

| 항목 | Bytes |
|---|---:|
| Activation bank A | {memory['bank_bytes']:,} |
| Activation bank B | {memory['bank_bytes']:,} |
| Weight int8 | {memory['weights_int8']:,} |
| Bias int32 | {memory['bias_int32']:,} |
| 네트워크 합계 | {memory['network_total_bytes']:,} |
| 출력 채널별 int32 multiplier + int8 shift 제안 | {kd['sram_proposal']['requant_multiplier_int32_and_shift_int8']:,} |
| 위 계수 저장 제안까지 포함 | {kd['sram_proposal']['network_plus_requant_bytes']:,} |

두 bank의 레이어별 생존 텐서와 복사·재사용 가정은 `models/student_kd/memory.json`에 기록했다. Upsample은 concat 목적지에 바로 기록하고 사용이 끝난 tensor는 회수한다. 별도 int32 누산 SRAM은 없다. 정렬·실제 재양자화 형식·후처리 저장 구조는 미확정이므로 해당 비용까지 검증한 하드웨어 전체 SRAM이라고 주장하지 않는다. 위 합계는 문서 예산 175,640B와 상위 제약 300,000B보다 작다. 교사와 adapter는 배포에서 제외한다.

## 파일과 재실행

- 배포 학생: `models/student_kd/model_float32.ts`, `inference/config.json`
- 학습 checkpoint: `models/teacher/best.pt`, `models/student_supervised/best.pt`, `models/student_kd/best.pt`
- 정량 결과: 각 모델 `results.json`, `history.json`, `*_predictions.jsonl`
- 정성 추론: `inference/results/`, `report/inference_contact_sheet.jpg`
- 학습 곡선: `report/learning_curves.png`
- 증강 전후 정답: `report/augmentation_preview.jpg` (초록: 적격, 노랑: ignore, 빨강: 가시율/유지율 미달).
- 실행 로그: `logs/`, 실행 환경: `.venv/`, 임시·캐시: `tmp/`, `.cache/`

```bash
cd advance_network
./run_env.sh infer.py --input inference_input --output inference/results
./run_env.sh scripts/check_advanced.py
```

학습 재현 시 `scripts/train_advanced.py`를 쓰고 `--output`을 새 디렉토리로 지정한다. `config.json`에 seed·손실·source hash·데이터 설정 hash·교사 hash가 기록되어 있다. 학습된 TorchScript는 floating point이며 int8 양자화 정확도는 별도 평가 대상이다.
'''
    (out/'TRAINING_RESULTS.md').write_text(text)
    (ROOT/'README.md').write_text('''# Advanced person detector

96×96 grayscale 사람 검출 모델 개선 실험. 데이터·환경·학습·추론 파일은 이 디렉토리에 둔다.

학습 결과와 비교: [report/TRAINING_RESULTS.md](report/TRAINING_RESULTS.md)

```bash
./run_env.sh infer.py --input inference_input --output inference/results
./run_env.sh scripts/check_advanced.py
```

모델은 `models/student_kd/model_float32.ts`, 설정은 `inference/config.json`, 결과 이미지는 `inference/results`에 있다. `baseline`은 기존 모델·코드 복사본이다.
''')
    print(text[:1800])
if __name__=='__main__':main()
