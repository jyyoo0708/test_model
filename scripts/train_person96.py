#!/usr/bin/env python3
"""Train with train/val only; load test after checkpoint and threshold are fixed."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models/person96"))
from model import Person96CNN, memory_profile


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def load_split(root, split):
    rows = [json.loads(line) for line in (root / f"{split}.jsonl").open()]
    with np.load(root / f"{split}.npz", allow_pickle=False) as pack:
        x = torch.from_numpy(pack["x"].copy())
        y = torch.from_numpy(pack["y"].copy()).float()
        assert np.array_equal(pack["sample_id"], [row["sample_id"] for row in rows])
    assert x.shape == (len(rows), 1, 96, 96) and x.dtype == torch.uint8
    assert all(row["split"] == split for row in rows)
    return x, y, rows


def metrics(y, probability, rows, threshold=.5):
    y = np.asarray(y, dtype=np.int64)
    pred = probability >= threshold
    tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred)); tp = int(np.sum((y == 1) & pred))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result = {"n": len(y), "threshold": float(threshold), "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (recall + tn / max(tn + fp, 1)) / 2,
        "person_precision": precision, "person_recall": recall,
        "person_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "confusion_matrix_true_rows_pred_columns": [[tn, fp], [fn, tp]],
        "majority_baseline": float(max(np.sum(y == 0), np.sum(y == 1)) / len(y))}
    for kind in ("background", "partial_person"):
        mask = np.array([r.get("negative_kind") == kind for r in rows])
        result[kind + "_count"] = int(mask.sum())
        result[kind + "_false_positive_rate"] = float(pred[mask].mean()) if mask.any() else None
    return result


@torch.inference_mode()
def evaluate(model, x, y, batch):
    model.eval()
    probabilities, total_loss = [], 0.
    for start in range(0, len(y), batch):
        inputs = x[start:start + batch].float().div_(255).contiguous(memory_format=torch.channels_last)
        logits = model(inputs)
        total_loss += nn.functional.binary_cross_entropy_with_logits(logits, y[start:start + batch], reduction="sum").item()
        probabilities.append(logits.sigmoid().numpy())
    return np.concatenate(probabilities), total_loss / len(y)


def augment(x):
    # No crop/rotation/erasing: body visibility labels must remain valid.
    flip = torch.rand(len(x)) < .5
    x[flip] = x[flip].flip(-1)
    mean = x.mean((2, 3), keepdim=True)
    contrast = torch.empty(len(x), 1, 1, 1).uniform_(.85, 1.15)
    brightness = torch.empty(len(x), 1, 1, 1).uniform_(.85, 1.15)
    return ((x - mean) * contrast + mean).mul_(brightness).clamp_(0, 1)


def cluster_accuracy_ci(rows, y, p, threshold, seed):
    groups = sorted(set(r["group_id"] for r in rows))
    index = {g: i for i, g in enumerate(groups)}
    sizes, correct = np.zeros(len(groups)), np.zeros(len(groups))
    for row, target, prob in zip(rows, y, p):
        i = index[row["group_id"]]
        sizes[i] += 1
        correct[i] += (prob >= threshold) == target
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(groups), size=(2000, len(groups)))
    values = correct[samples].sum(1) / sizes[samples].sum(1)
    return {"method": "2000 bootstrap resamples of source similarity groups", "groups": len(groups),
            "lower_95": float(np.quantile(values, .025)), "upper_95": float(np.quantile(values, .975))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/person96")
    ap.add_argument("--output", type=Path, default=ROOT / "models/person96/run01")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=.002)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--partial-weight", type=float, default=3.)
    args = ap.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "device": "cpu", "precision": "float32", "optimizer": "AdamW", "weight_decay": .0001,
        "checkpoint_criterion": "validation balanced accuracy at threshold 0.5",
        "threshold_criterion": "maximum validation balanced accuracy; ties closest to 0.5",
        "source_code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (Path(__file__), ROOT / "models/person96/model.py")},
        "data_config_sha256": hashlib.sha256((args.data / "config.json").read_bytes()).hexdigest()})
    dump(args.output / "config.json", config)
    train_x, train_y, train_rows = load_split(args.data, "train")
    val_x, val_y, val_rows = load_split(args.data, "val")
    assert not {r["group_id"] for r in train_rows} & {r["group_id"] for r in val_rows}
    sample_weights = torch.tensor([args.partial_weight if r.get("negative_kind") == "partial_person" else 1.
                                   for r in train_rows])
    model = Person96CNN().to(memory_format=torch.channels_last)
    print(f"TRAIN: {len(train_y)}; VAL: {len(val_y)}; parameters: {sum(p.numel() for p in model.parameters())}", flush=True)
    dump(args.output / "memory.json", memory_profile(model.fused()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=.00005)
    best, best_epoch, history = -1., 0, []
    for epoch in range(1, args.epochs + 1):
        tick = time.time()
        model.train()
        order = torch.randperm(len(train_y))
        loss_sum, weight_sum, correct = 0., 0., 0
        for batch_no, start in enumerate(range(0, len(order), args.batch_size)):
            indices = order[start:start + args.batch_size]
            x = augment(train_x[indices].float().div_(255)).contiguous(memory_format=torch.channels_last)
            y, w = train_y[indices], sample_weights[indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            losses = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
            loss = (losses * w).sum() / w.sum()
            loss.backward()
            optimizer.step()
            loss_sum += (losses.detach() * w).sum().item(); weight_sum += w.sum().item()
            correct += ((logits.detach() >= 0) == y.bool()).sum().item()
            if (batch_no + 1) % 55 == 0:
                print(f"epoch {epoch:02d} batch {batch_no + 1}/166 elapsed={time.time()-tick:.1f}s", flush=True)
        val_p, val_loss = evaluate(model, val_x, val_y, args.batch_size)
        vm = metrics(val_y.numpy(), val_p, val_rows)
        row = {"epoch": epoch, "train_weighted_loss": loss_sum / weight_sum,
               "train_augmented_accuracy": correct / len(train_y), "val_loss": val_loss,
               "val_accuracy": vm["accuracy"], "val_balanced_accuracy": vm["balanced_accuracy"],
               "val_person_recall": vm["person_recall"],
               "val_partial_false_positive_rate": vm["partial_person_false_positive_rate"],
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - tick}
        history.append(row)
        dump(args.output / "history.json", history)
        print(json.dumps(row), flush=True)
        if vm["balanced_accuracy"] > best + 1e-6:
            best, best_epoch = vm["balanced_accuracy"], epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "validation": vm, "config": config}, args.output / "best.pt")
        scheduler.step()
        if epoch >= 10 and epoch - best_epoch >= args.patience:
            print(f"Early stop at {epoch}; best epoch {best_epoch}", flush=True)
            break
    checkpoint = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    val_p, _ = evaluate(model, val_x, val_y, args.batch_size)
    choices = []
    for threshold in np.linspace(.05, .95, 181):
        vm = metrics(val_y.numpy(), val_p, val_rows, threshold)
        choices.append((vm["balanced_accuracy"], -abs(threshold - .5), float(threshold)))
    threshold = max(choices)[2]
    # First access to held-out test occurs only after both selections are final.
    test_x, test_y, test_rows = load_split(args.data, "test")
    assert not {r["group_id"] for r in test_rows} & {r["group_id"] for r in train_rows + val_rows}
    train_p, _ = evaluate(model, train_x, train_y, args.batch_size)
    test_p, test_loss = evaluate(model, test_x, test_y, args.batch_size)
    result = {"best_epoch": best_epoch, "epochs_run": len(history), "threshold": threshold,
              "test_loss_unweighted": test_loss, "metrics": {}, "metrics_at_0_5": {}}
    for split, y, p, rows in (("train", train_y, train_p, train_rows), ("val", val_y, val_p, val_rows),
                              ("test", test_y, test_p, test_rows)):
        result["metrics"][split] = metrics(y.numpy(), p, rows, threshold)
        result["metrics_at_0_5"][split] = metrics(y.numpy(), p, rows, .5)
        with (args.output / f"{split}_predictions.csv").open("w") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "source_id", "group_id", "image", "label", "negative_kind", "probability_person", "prediction"])
            for row, target, prob in zip(rows, y.numpy(), p):
                writer.writerow([row["sample_id"], row["source_id"], row["group_id"], row["image"], int(target),
                                 row.get("negative_kind", ""), float(prob), int(prob >= threshold)])
    result["test_group_bootstrap_accuracy_ci"] = cluster_accuracy_ci(test_rows, test_y.numpy(), test_p, threshold, args.seed)
    fused = model.fused()
    torch.save({"state_dict": fused.state_dict(), "threshold": threshold, "config": config}, args.output / "model_float32.pt")
    np.savez_compressed(args.output / "weights_float32.npz", **{k: v.detach().numpy() for k, v in fused.state_dict().items()})
    example = test_x[:64].float().div(255).contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        original_logits, fused_logits = model(example), fused(example)
        assert torch.allclose(original_logits, fused_logits, atol=2e-5, rtol=1e-5)
        result["bn_fold_max_abs_logit_error"] = float((original_logits - fused_logits).abs().max())
        exported = torch.jit.freeze(torch.jit.trace(fused, example[:1]))
        exported.save(str(args.output / "model_float32.ts"))
        loaded = torch.jit.load(str(args.output / "model_float32.ts")).eval()
        loaded_logits = loaded(example)
        assert torch.allclose(fused_logits, loaded_logits, atol=2e-5, rtol=1e-5)
        result["torchscript_reload_max_abs_logit_error"] = float((fused_logits - loaded_logits).abs().max())
        result["exported_predictions_match_original"] = bool(torch.equal(original_logits.sigmoid() >= threshold, loaded_logits.sigmoid() >= threshold))
        assert result["exported_predictions_match_original"]
        # Time only model forward, not file I/O; single CPU thread, batch=1.
        torch.set_num_threads(1)
        for _ in range(20):
            loaded(example[:1])
        timings = []
        for _ in range(200):
            tick = time.perf_counter(); loaded(example[:1]); timings.append((time.perf_counter() - tick) * 1000)
    result["cpu_batch1_forward_latency_ms"] = {"threads": 1, "runs": 200,
        "median": float(np.median(timings)), "p95": float(np.quantile(timings, .95))}
    result["wall_seconds"] = time.time() - started
    dump(args.output / "results.json", result)
    print("FINAL " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
