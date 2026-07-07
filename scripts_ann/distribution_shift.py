"""
distribution_shift.py
Feedback-free response-frequency (distribution-shift) metacognitive module
(manuscript Supp 6.3).

Unlike the learned heads in ``train_metacognitive.py``, this module uses NO
correctness labels and NO gradient training. It mirrors the feedback structure of
the human task: from a balanced calibration set it estimates which response
categories the frozen classifier OVER-selects, then at test time it subtracts a
per-category evidence penalty S_k that down-weights confidence for over-selected
categories.

Pipeline (per frozen base instance)
-----------------------------------
1. Calibrate on a balanced MNIST-TRAIN subset (seed 12345): for each chosen
   category k, collect the z-scored top-two logit margin (the evidence signal),
   and count how often k is chosen.
2. For every over-selected category (response rate > 1/K), set
       S_k = percentile(evidence_k, (1 - (1/K)/p_actual) * 100)
   i.e. the evidence quantile that would bring k's response rate down to 1/K.
   Categories chosen at or below chance get S_k = 0.
3. Evaluate on a balanced MNIST-TEST subset (seed 12346): the corrected
   confidence is  conf_meta = conf_z_top2diff - S_{chosen category}.

Output columns (per test image):
    instance, image_idx, true_label, pred_label, correct,
    conf_meta, conf_top2diff, max_softmax, entropy, noise_level, mode,
    conf_z_top2diff, shift_applied, bias_metric

Usage
-----
python scripts_ann/distribution_shift.py \
    --model alexnet --noise_level 1.05 \
    --base_dir ./weights/alexnet/ \
    --output_dir ./data/model_data/distribution_shift/
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from core.meta_modules import build_backbone, load_base_into_backbone
from core.datasets import get_balanced_loader, resize_and_noise, \
    BALANCED_CALIB_SEED, BALANCED_TEST_SEED

import warnings
warnings.filterwarnings("ignore")

DEFAULT_NOISE = {'alexnet': 1.05, 'resnet18': 0.24, 'vgg19': 0.19}
DEFAULT_BATCH = {'alexnet': 256, 'resnet18': 256, 'vgg19': 128}
NUM_CLASSES = 10
MODE = 'distribution_shift'

CSV_COLUMNS = [
    'instance', 'image_idx', 'true_label', 'pred_label', 'correct',
    'conf_meta', 'conf_top2diff', 'max_softmax', 'entropy', 'noise_level', 'mode',
    'conf_z_top2diff', 'shift_applied', 'bias_metric',
]


def _z_top2_margin(logits):
    """Z-score logits across classes, then return the top-two margin (evidence)."""
    z = (logits - logits.mean(1, keepdim=True)) / (logits.std(1, keepdim=True) + 1e-8)
    z_top2 = torch.topk(z, 2, dim=1).values
    return (z_top2[:, 0] - z_top2[:, 1])


def get_distribution_stats(backbone, loader, device, noise_level, num_classes=NUM_CLASSES):
    """Calibration pass: per-chosen-category evidence samples + choice counts."""
    evidence_stats = {k: [] for k in range(num_classes)}
    counts = np.zeros(num_classes)
    total = 0
    with torch.no_grad():
        for x, _ in loader:
            x = resize_and_noise(x.to(device), noise_level)
            logits = backbone(x)
            ev = _z_top2_margin(logits).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            for p, e in zip(preds, ev):
                counts[p] += 1
                evidence_stats[p].append(e)
            total += x.size(0)
    return counts, evidence_stats, total


def calculate_shifts(counts, evidence_stats, total, num_classes=NUM_CLASSES):
    """Per-category evidence penalty S_k (0 for categories not over-selected)."""
    shifts = {}
    target_prob = 1.0 / num_classes
    for k in range(num_classes):
        count = counts[k]
        if count < 10:
            shifts[k] = 0.0
            continue
        p_actual = count / total
        if p_actual > target_prob:
            keep_ratio = target_prob / p_actual
            pct = max(0.0, min(100.0, (1 - keep_ratio) * 100))
            shifts[k] = float(np.percentile(np.array(evidence_stats[k]), pct))
        else:
            shifts[k] = 0.0
    return shifts


def eval_instance(inst, backbone, calib_loader, test_loader, device, noise_level):
    """Calibrate then evaluate one frozen base instance; return per-image rows."""
    # 1. calibration (reproducible per-instance noise)
    torch.manual_seed(inst)
    counts, evidence_stats, total = get_distribution_stats(backbone, calib_loader, device, noise_level)
    shifts = calculate_shifts(counts, evidence_stats, total)

    # 2. test
    torch.manual_seed(inst)
    rows = []
    img_idx = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = resize_and_noise(x.to(device), noise_level)
            logits = backbone(x)
            conf_z = _z_top2_margin(logits).cpu().numpy()

            probs = F.softmax(logits, dim=1)
            sm_top2 = torch.topk(probs, 2, dim=1).values
            conf_top2diff = (sm_top2[:, 0] - sm_top2[:, 1]).cpu().numpy()
            max_softmax = probs.max(1).values.cpu().numpy()
            entropy = (-probs * probs.clamp_min(1e-12).log()).sum(1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            labels = y.numpy()

            for i in range(len(labels)):
                p = int(preds[i])
                s = float(shifts[p])
                cz = float(conf_z[i])
                rows.append({
                    'instance': inst,
                    'image_idx': img_idx + i,
                    'true_label': int(labels[i]),
                    'pred_label': p,
                    'correct': int(preds[i] == labels[i]),
                    'conf_meta': cz - s,                     # corrected confidence
                    'conf_top2diff': float(conf_top2diff[i]),
                    'max_softmax': float(max_softmax[i]),
                    'entropy': float(entropy[i]),
                    'noise_level': noise_level,
                    'mode': MODE,
                    'conf_z_top2diff': cz,
                    'shift_applied': s,
                    'bias_metric': float(counts[p] / total),
                })
            img_idx += len(labels)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Feedback-free distribution-shift metacognitive module.")
    parser.add_argument('--model', choices=['alexnet', 'resnet18', 'vgg19'], required=True)
    parser.add_argument('--noise_level', type=float, default=None)
    parser.add_argument('--base_dir', type=str, required=True, help='Frozen base weights dir.')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_instances', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()

    noise_level = args.noise_level if args.noise_level is not None else DEFAULT_NOISE[args.model]
    batch_size = args.batch_size if args.batch_size is not None else DEFAULT_BATCH[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    calib_loader = get_balanced_loader(train=True, seed=BALANCED_CALIB_SEED, batch_size=batch_size)
    test_loader = get_balanced_loader(train=False, seed=BALANCED_TEST_SEED, batch_size=batch_size)
    out_csv = os.path.join(args.output_dir, f'{args.model}_distribution_shift.csv')
    print(f"Distribution-shift module for {args.model} (noise={noise_level}) -> {out_csv}")

    with open(out_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for inst in tqdm(range(args.num_instances), desc="Instances"):
            base_ckpt = os.path.join(args.base_dir, f"{args.model}-224-{inst}-final.pt")
            if not os.path.exists(base_ckpt):
                print(f"[skip] missing {base_ckpt}")
                continue
            backbone = build_backbone(args.model, num_classes=NUM_CLASSES)
            load_base_into_backbone(backbone, base_ckpt)
            backbone.to(device).eval()
            writer.writerows(eval_instance(inst, backbone, calib_loader, test_loader, device, noise_level))
            fh.flush()
            del backbone
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"Done. Wrote {out_csv}")


if __name__ == "__main__":
    main()
