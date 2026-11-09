#!/usr/bin/env python3
"""Build the Korean Markdown training report and standalone figures from logs."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/person96-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=ROOT / "models/person96/run01")
    args = ap.parse_args()
    run = args.run
    cfg = json.loads((run / "config.json").read_text())
    result = json.loads((run / "results.json").read_text())
    audit_path = ROOT / "report/person96_training/model_validation.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    memory = json.loads((run / "memory.json").read_text())
    history = json.loads((run / "history.json").read_text())
    stats = json.loads((ROOT / "data/processed/person96/stats.json").read_text())
    assets = ROOT / "report/person96_training"
    assets.mkdir(exist_ok=True, parents=True)
    epoch = [r["epoch"] for r in history]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1), constrained_layout=True)
    axes[0].plot(epoch, [r["train_augmented_accuracy"] for r in history], label="Train (augmented)")
    axes[0].plot(epoch, [r["val_accuracy"] for r in history], label="Validation accuracy")
    axes[0].plot(epoch, [r["val_balanced_accuracy"] for r in history], "--", label="Validation balanced accuracy")
    axes[0].axvline(result["best_epoch"], color="gray", linestyle=":", label="Selected epoch")
    axes[0].axhline(.8, color="green", alpha=.4, linestyle=":")
    axes[0].set(xlabel="Epoch", ylabel="Score at threshold 0.5", ylim=(.45, 1.0), title="Learning curves")
    axes[1].plot(epoch, [r["train_weighted_loss"] for r in history], label="Train BCE (partial weight 3)")
    axes[1].plot(epoch, [r["val_loss"] for r in history], label="Validation BCE (unweighted)")
    axes[1].set(xlabel="Epoch", ylabel="BCE loss", title="Losses (different weighting)")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(assets / "learning_curves.png", dpi=200)
    fig.savefig(assets / "learning_curves.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.7), constrained_layout=True)
    for ax, split in zip(axes, ("val", "test")):
        cm = np.array(result["metrics"][split]["confusion_matrix_true_rows_pred_columns"])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", color="white" if cm[i,j] > cm.max() * .5 else "black", fontsize=14)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-person", "Person"],
               yticklabels=["Non-person", "Person"], xlabel="Predicted", ylabel="True", title=split.upper())
    fig.savefig(assets / "confusion_matrices.png", dpi=200)
    fig.savefig(assets / "confusion_matrices.svg")
    plt.close(fig)
    with (run / "test_predictions.csv").open() as f:
        predictions = list(csv.DictReader(f))
    sources = {row["ID"]: (row["width"], row["height"])
               for row in map(json.loads, (ROOT / "data/processed/person96/source_annotations.jsonl").open())}
    crop_rows = {row["sample_id"]: row for row in map(json.loads, (ROOT / "data/processed/person96/test.jsonl").open())}
    padding = {"padded_person": [0, 0], "unpadded_person": [0, 0], "padded_non_person": [0, 0]}
    for row in predictions:
        crop = crop_rows[row["sample_id"]]["crop_xyxy"]
        width, height = sources[row["source_id"]]
        padded = crop[0] < 0 or crop[1] < 0 or crop[2] > width or crop[3] > height
        key = ("padded_person" if padded else "unpadded_person") if row["label"] == "1" else ("padded_non_person" if padded else None)
        if key:
            padding[key][0] += 1
            padding[key][1] += row["label"] == row["prediction"]
    (assets / "padding_audit.json").write_text(json.dumps(padding, indent=2) + "\n")
    errors = {}
    for kind in ("false_positive", "false_negative"):
        selected = [r for r in predictions if (r["label"], r["prediction"]) == (("0", "1") if kind == "false_positive" else ("1", "0"))]
        selected.sort(key=lambda r: float(r["probability_person"]), reverse=kind == "false_positive")
        errors[kind] = len(selected)
        chosen = selected[:36]
        canvas = Image.new("RGB", (6 * 116, 6 * 126), "white")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(chosen):
            x, y = (i % 6) * 116, (i // 6) * 126
            with Image.open(ROOT / "data/processed/person96" / row["image"]) as patch:
                canvas.paste(patch.convert("RGB"), (x, y))
            draw.text((x, y + 97), f"{i}: p={float(row['probability_person']):.3f}", fill="black")
            draw.text((x, y + 110), row["negative_kind"] or "person", fill="black")
        canvas.save(assets / f"{kind}.jpg", quality=95)
        (assets / f"{kind}_index.json").write_text(json.dumps(chosen, indent=2) + "\n")
    pct = lambda x: f"{x:.2%}"
    m = result["metrics"]; default = result["metrics_at_0_5"]
    ci = result["test_group_bootstrap_accuracy_ci"]
    report = ["# 사람 인식 CNN 학습 결과", "", "작성일: 2026-11-05 (Asia/Seoul)", "",
        f"96×96 grayscale 입력의 소형 CNN을 float32로 학습했다. 선택한 epoch는 **{result['best_epoch']}**, validation에서 선택한 확률 임계값은 **{result['threshold']:.3f}**다. 최종 held-out test {m['test']['n']:,}개에서 정확도 **{pct(m['test']['accuracy'])}**, balanced accuracy **{pct(m['test']['balanced_accuracy'])}**, 사람 recall **{pct(m['test']['person_recall'])}**를 얻었다.", "",
        f"`AGENTS.md`의 목표 정확도 70~80% {'범위 이상을 달성했다' if m['test']['accuracy'] >= .8 else ('하한 70%를 달성했다' if m['test']['accuracy'] >= .7 else '하한 70%에 도달하지 못했다')}. 이 결과는 준비한 패치 데이터의 자체 split 기준이며, 실제 카메라 전체 화면 성능이나 공식 CrowdHuman benchmark 성능을 뜻하지 않는다.", "",
        "## 데이터와 평가 절차", "",
        "데이터 준비 기준은 [데이터 보고서](data_preparation.md)에 기록했다. 공식 CrowdHuman validation 원본 4,370장을 자체 train/val/test로 분할한 데이터다. 같은 원본과 dHash 유사 그룹은 split을 공유한다. 전신/가시 bbox 면적 기준 ≥80%와 원본/입력 최소 크기를 만족하는 사람만 label 1이다. label 0은 안전한 배경 또는 전신의 50% 이하만 남긴 부분 신체다. 80%는 실제 신체 segmentation 비율이 아닌 bbox 기반 근사치다.", "",
        "| Split | 원본 이미지 | 입력 패치 | 사람 | 배경 | 부분 신체 |", "|---|---:|---:|---:|---:|---:|"]
    for split in ("train", "val", "test"):
        d = stats["splits"][split]
        report.append(f"| {split} | {stats['grouping']['source_counts'][split]:,} | {d['total']:,} | {d['person']:,} | {d['background']:,} | {d['partial_person']:,} |")
    report += ["", "학습 중 train/validation만 읽었다. validation의 threshold 0.5 balanced accuracy가 가장 높은 checkpoint를 선택했고, 선택된 checkpoint의 validation 확률로 threshold 0.05~0.95를 0.005 간격으로 탐색했다. 동점이면 0.5에 가까운 값을 선택했다. 이후 처음 test를 읽어 평가했으며 test 결과로 재학습·모델/임계값 선택을 하지 않았다. train 최종 수치는 augmentation과 dropout을 끈 별도 평가 결과다.", "",
        "## 네트워크", "",
        "입력: float32 `[B,1,96,96]`, PNG uint8 값을 255로 나눠 `[0,1]`로 정규화한다. Conv는 모두 kernel 3×3, stride 2, padding 1이다. 각 Conv 뒤에 BatchNorm과 ReLU가 있고, 추론 export 시 BatchNorm을 Conv에 접어 넣었다. Flatten 후 FC 32 → ReLU → Dropout 0.25 → FC 1로 사람 logit을 출력한다. 판정은 sigmoid(logit) ≥ 선택 threshold다.", "",
        "| 계층 | 출력 (batch 제외) | Weight 수 | Bias 수 (BN fold 후) | MACs |", "|---|---|---:|---:|---:|"]
    for row in memory["layers"]:
        shape = "×".join(map(str, row["output_shape"][1:]))
        report.append(f"| {row['name']} ({row['type']}) | {shape} | {row['weights']:,} | {row['biases']:,} | {row['macs']:,} |")
    report += ["", f"학습 파라미터 75,225개(BN affine 포함), BN fold 후 {memory['float32_fused_parameters']:,}개다. 입력당 {memory['macs_per_input']:,} MACs이며 사전 학습 가중치를 사용하지 않고 처음부터 학습했다.", "",
        "## 학습 설정", "",
        f"- 환경: Python {cfg['python']}, PyTorch {cfg['torch']}, NumPy {cfg['numpy']}, CPU 8 threads. 이 실행에서 GPU를 사용하지 않았다.",
        f"- seed `{cfg['seed']}`, float32, batch {cfg['batch_size']}, AdamW, 초기 learning rate {cfg['lr']}, weight decay 0.0001.",
        f"- CosineAnnealingLR: 최대 {cfg['epochs']} epochs, 최저 learning rate 0.00005. 10 epoch 이후 validation 개선이 {cfg['patience']} epoch 동안 없으면 early stopping.",
        f"- 실제 {result['epochs_run']} epochs 실행, 데이터 로딩·최종 평가·export 포함 {result['wall_seconds']/60:.2f}분.",
        "- BCEWithLogitsLoss. 부분 신체 음성의 sample weight는 3, 나머지는 1. 배치 내 weight 합으로 정규화했다.",
        "- train에만 좌우 반전 확률 0.5, contrast/brightness 각각 0.85~1.15. 신체 비율을 바꾸는 random crop·회전·erasing은 사용하지 않았다.",
        "- deterministic algorithms를 활성화했다. 같은 환경/버전/seed 재현을 위한 소스·데이터 설정 SHA-256을 run config에 기록했다.", "",
        "![Learning curves](person96_training/learning_curves.png)", "",
        "곡선의 train accuracy는 증강/dropout이 적용된 학습 중 수치이며 아래 표의 train accuracy는 최종 선택 모델의 원본 입력 평가다. train loss는 부분 신체 가중 BCE, validation loss는 비가중 BCE이므로 절대값을 직접 비교하지 않는다.", "",
        "## 최종 성능", "",
        "| Split | Accuracy | Balanced accuracy | Person precision | Person recall | Person F1 | 다수 클래스 기준선 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("train", "val", "test"):
        row = m[split]
        report.append("| " + split + " | " + " | ".join(pct(row[k]) for k in ("accuracy", "balanced_accuracy", "person_precision", "person_recall", "person_f1", "majority_baseline")) + " |")
    report += ["", f"기본 threshold 0.5의 validation/test accuracy는 각각 {pct(default['val']['accuracy'])}/{pct(default['test']['accuracy'])}이며, 선택 threshold {result['threshold']:.3f}의 validation/test accuracy는 {pct(m['val']['accuracy'])}/{pct(m['test']['accuracy'])}다.", "",
        f"test accuracy의 원본 유사 그룹 단위 bootstrap 95% 구간은 **{pct(ci['lower_95'])}~{pct(ci['upper_95'])}**다. test의 {ci['groups']}개 그룹을 2,000회 재표집했다. 패치 간 상관을 일부 반영하지만 신원/장면의 완전한 독립성이나 실제 환경 변화를 보장하지는 않는다.", "",
        "![Confusion matrices](person96_training/confusion_matrices.png)", "",
        "행은 실제 label, 열은 예측 label이며 순서는 non-person / person이다.", "",
        "| 음성 종류 | Test 수량 | 사람으로 잘못 예측한 비율 |", "|---|---:|---:|",
        f"| 배경 | {m['test']['background_count']:,} | {pct(m['test']['background_false_positive_rate'])} |",
        f"| 부분 신체 (전신 ≤50%) | {m['test']['partial_person_count']:,} | {pct(m['test']['partial_person_false_positive_rate'])} |", "",
        f"오검출(false positive) {errors['false_positive']:,}개, 사람 미검출(false negative) {errors['false_negative']:,}개다. 아래 이미지는 오답 중 가장 자신 있게 틀린 최대 36개씩이며, 전체 분포를 대표하는 무작위 표본은 아니다. 각 이미지의 sample ID·원본 ID는 같은 이름의 `_index.json`에서 추적할 수 있다.", "",
        "![False positives](person96_training/false_positive.jpg)", "", "![False negatives](person96_training/false_negative.jpg)", "",
        "상위 오답을 시각 확인하면 배경의 건물 창틀·문자·나뭇가지처럼 사람과 혼동되는 패턴, 부분 신체 오검출이 보인다. 사람 미검출에는 blur가 심하거나 작게 보이는 대상, 앉거나 누운 자세, 혼잡한 장면 등이 포함된다. 이는 그림의 표본에 대한 관찰이며 모든 오답의 원인을 전수 분류한 결과는 아니다.", "",
        "## SRAM 제약 검토", "",
        "아래는 **향후** activation/weight int8, bias/accumulator int32, batch=1, im2col 없는 직접 convolution을 가정한 계산이다. 모든 계층 출력을 동시에 보존하고 가장 큰 계층의 전체 int32 accumulator를 별도로 두는 방식으로 계산했다. 16KiB의 추가 제어/작업 공간을 예약했다. 추론용 BN은 이미 Conv에 fold했다.", "",
        "| 항목 | Bytes |", "|---|---:|"]
    labels = {"int8_weights_bytes": "int8 weights", "int32_biases_bytes": "int32 biases",
              "input_and_all_layer_int8_outputs_bytes": "입력 및 전체 계층 int8 출력",
              "largest_full_int32_accumulator_bytes": "최대 전체 계층 int32 accumulator",
              "per_output_channel_and_activation_scales_float32_bytes": "채널별/activation scale (float32)",
              "reserved_control_and_other_workspace_bytes": "추가 제어/작업 공간 예약"}
    for key, value in memory["parts"].items():
        report.append(f"| {labels[key]} | {value:,} |")
    report += [f"| **합계** | **{memory['estimated_total_bytes']:,}** |", "",
        f"총 **{memory['estimated_total_bytes']/1000:.2f}KB ({memory['estimated_total_bytes']/1024:.2f}KiB)**로, 더 엄격한 300,000 bytes 기준으로도 약 {(300000-memory['estimated_total_bytes'])/1000:.2f}KB 여유가 있다. 실제 RTL의 bank 정렬·포트 구성·DMA·requantization·구체적인 작업 공간을 설계한 뒤 다시 확인해야 한다. 이 수치는 SRAM 실측이나 구현 보증이 아니다.", "",
        f"현재 제공 모델은 float32이며 BN fold 후 파라미터만 {memory['float32_fused_parameter_bytes']:,} bytes다. 위 300KB 판정은 float32 모델이 아닌 미래 정수 추론 배치를 가정한다. int8 변환·정수 정확도 측정·RTL 구현은 수행하지 않았다.", "",
        "## Export와 검증", "",
        f"- Conv/BN fold 전후 64개 test 입력의 최대 logit 차이: `{result['bn_fold_max_abs_logit_error']:.8g}`.",
        f"- TorchScript 저장 후 재로딩 최대 logit 차이: `{result['torchscript_reload_max_abs_logit_error']:.8g}`.",
        f"- 이 64개 입력의 원본 모델과 export 모델 판정 일치: `{result['exported_predictions_match_original']}`.",
        (f"- 별도 저장 모델 검사 **{audit['status']}**: test {audit['test_examples']:,}개 전체의 판정이 저장된 원본 예측과 일치했고 최대 확률 차이는 `{audit['max_abs_export_probability_error']:.8g}`다. float32 NPZ 배열과 checkpoint 가중치 일치, fused state dict 재구성, PNG 양성/음성 직접 추론도 확인했다. 기록: `report/person96_training/model_validation.json`." if audit else "- 별도 전체 export 검사는 `scripts/validate_person96_model.py`로 실행할 수 있다."),
        f"- CPU 1 thread, batch=1, warm-up 20회 뒤 200회 forward 중앙값 `{result['cpu_batch1_forward_latency_ms']['median']:.3f} ms`, p95 `{result['cpu_batch1_forward_latency_ms']['p95']:.3f} ms`. 파일 읽기·전처리 제외이며 TinyAI/RTL 속도를 뜻하지 않는다.", "",
        "## 결과물 및 재현", "",
        "- 모델 코드: `models/person96/model.py`",
        "- 선택 checkpoint(BN 포함): `models/person96/run01/best.pt`",
        "- BN fold float32 state dict: `models/person96/run01/model_float32.pt`",
        "- 바로 추론 가능한 TorchScript: `models/person96/run01/model_float32.ts`",
        "- float32 배열: `models/person96/run01/weights_float32.npz`",
        "- 설정/학습 기록/최종 수치: 같은 run 폴더의 `config.json`, `history.json`, `results.json`, `memory.json`",
        "- 전체 split별 예측: 같은 run 폴더의 `{train,val,test}_predictions.csv`",
        "- 실행 로그·설치 버전: `report/person96_training/train.log`, `environment.txt`",
        "- 학습/추론/보고서 스크립트: `scripts/train_person96.py`, `scripts/infer_person96.py`, `scripts/report_person96.py`", "",
        "이 시스템의 Cadence 라이브러리 경로가 PyTorch의 `libshm.so`와 충돌해, 아래처럼 실행 프로세스에서만 `LD_LIBRARY_PATH`를 제거했다. 시스템 설정이나 Cadence 설치는 수정하지 않았다.", "",
        "```bash", "python3.11 -m venv /tmp/person96-train-venv",
        "/tmp/person96-train-venv/bin/pip install -r requirements-train.txt",
        "# 기존 결과를 덮어쓰지 않도록 재학습은 새 output 폴더 사용",
        "env -u LD_LIBRARY_PATH /tmp/person96-train-venv/bin/python scripts/train_person96.py --threads 8 --epochs 24 --batch-size 256 --output models/person96/reproduce01",
        "# 보고서 재생성 (지정 run의 결과로 report/person96_training.md를 갱신)",
        "env -u LD_LIBRARY_PATH /tmp/person96-train-venv/bin/python scripts/report_person96.py --run models/person96/run01",
        "# 기존 96×96 정사각형 PNG 패치 추론",
        "env -u LD_LIBRARY_PATH /tmp/person96-train-venv/bin/python scripts/infer_person96.py <patch.png>", "```", "",
        "저장 모델 전체 test 재현 검사는 `env -u LD_LIBRARY_PATH /tmp/person96-train-venv/bin/python scripts/validate_person96_model.py`로 실행한다. 모델 파일 체크섬은 `models/person96/run01/SHA256SUMS`에 기록했다.", "",
        "## 결과 해석과 한계", "",
        "현재 결과는 잘라낸 영역의 사람 존재 판별 성능이다. 카메라 전체 화면을 96×96으로 줄여 넣는 검출 시스템의 성능과 다르며, 실제 적용에는 동일한 영역 추출 방식이나 별도 검출 설계가 필요하다. 원본 배경의 일부와 부분 신체가 음성이어서 독립적인 사람 없는 장면·물체·야간 환경 등의 다양성도 제한적이다.", "",
        "부분 신체 음성은 test 172개로 적고 50~80% 경계 구간은 학습 정답에서 제외됐다. 따라서 전체 정확도가 높아도 **실제 신체 80% 기준을 엄밀히 학습·보장했다고 해석하면 안 된다**. 해당 조건을 엄격하게 적용하려면 경계 부근의 정밀 annotation과 실제 사용 환경에서의 추가 검증이 필요하다. 96×96 입력에서 작은 대상, 가려짐, 강한 blur, grayscale만으로 구분하기 어려운 부분 신체는 실패 유형으로 따로 살펴봐야 한다.", "",
        f"추가로 crop의 원본 밖 검은 padding은 양성에만 생기는 데이터 편향이 있다. test 양성 중 padding 포함 {padding['padded_person'][0]:,}개, 미포함 {padding['unpadded_person'][0]:,}개이며, 음성의 padding 포함 수는 {padding['padded_non_person'][0]}개다. 각 양성 그룹의 recall은 {pct(padding['padded_person'][1]/max(padding['padded_person'][0],1))}/{pct(padding['unpadded_person'][1]/max(padding['unpadded_person'][0],1))}다. 모델이 padding을 단서로 이용했을 가능성을 배제할 수 없다. 향후 padding 정책을 양성/음성에 맞추고 실제 입력 영역으로 평가해야 한다. 이 항목은 사후 진단이며 이 test를 보고 모델을 다시 선택하지 않았다.", "",
        "교육·비상업 연구 목적의 CrowdHuman 이용 조건과 이미지 재배포 제한은 데이터 준비 보고서와 동일하게 적용된다. 학습 모델의 실제 환경 성능과 int8 정수 성능은 아직 검증하지 않았다.", "",
        "참고: [PyTorch CPU 설치 안내](https://pytorch.org/get-started/locally/), [CrowdHuman annotation 및 이용 조건](https://www.crowdhuman.org/download.html).", ""]
    (ROOT / "report/person96_training.md").write_text("\n".join(report))
    print(ROOT / "report/person96_training.md")


if __name__ == "__main__":
    main()
