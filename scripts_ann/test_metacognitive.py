"""
test_metacognitive.py
Evaluate a trained learned metacognitive head (fixed_base / chen / dual) on the
balanced MNIST test set and write one per-image CSV per architecture x mode.

The frozen base classifier produces the digit response; the learned head produces
the meta-confidence for that response. Because the backbone is frozen, the
per-image responses, digit-level FAR, and digit-level accuracy are identical
across all metacognitive modes — only the confidence readout differs. The same
CSV therefore also carries ``conf_top2diff`` (the standard softmax logit-margin
confidence), which reproduces the standard-ANN analysis (Fig 6) from the same run.

Output columns (one row per test image):
    instance, image_idx, true_label, pred_label, correct,
    conf_meta, conf_top2diff, max_softmax, entropy, noise_level, mode
    [+ p_FA_0 ... p_FA_9 for mode='dual']

These match the schema consumed by the analysis notebook / scripts_analysis.

Usage
-----
python scripts_ann/test_metacognitive.py \
    --model alexnet --mode fixed_base --noise_level 1.05 \
    --base_dir ./weights/alexnet/ \
    --meta_dir ./weights_meta/alexnet/ \
    --output_dir ./data/model_data/meta/
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from core.meta_modules import LitMetaModel, VALID_MODES
from core.datasets import get_balanced_loader, BALANCED_TEST_SEED

import warnings
warnings.filterwarnings("ignore")

DEFAULT_NOISE = {'alexnet': 1.05, 'resnet18': 0.24, 'vgg19': 0.19}
DEFAULT_BATCH = {'alexnet': 256, 'resnet18': 256, 'vgg19': 128}
NUM_CLASSES = 10


def base_columns():
    return ['instance', 'image_idx', 'true_label', 'pred_label', 'correct',
            'conf_meta', 'conf_top2diff', 'max_softmax', 'entropy', 'noise_level', 'mode']


def eval_instance(inst, model, test_loader, device, noise_level, mode):
    """Return per-image row dicts for one trained instance."""
    model.to(device)
    model.eval()
    torch.manual_seed(inst)   # reproducible per-instance test noise
    rows = []
    img_idx = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, conf, aux, _, entropy = model(x)

            probs = F.softmax(logits, dim=-1)
            top2 = torch.topk(probs, 2, dim=-1).values
            conf_top2diff = (top2[:, 0] - top2[:, 1]).cpu().numpy()
            max_softmax = probs.max(1).values.cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            labels = y.cpu().numpy()
            conf_np = conf.cpu().numpy()
            entropy_np = entropy.cpu().numpy()
            p_fa = aux.cpu().numpy() if (mode == 'dual' and aux is not None) else None

            for i in range(len(labels)):
                row = {
                    'instance': inst,
                    'image_idx': img_idx + i,
                    'true_label': int(labels[i]),
                    'pred_label': int(preds[i]),
                    'correct': int(preds[i] == labels[i]),
                    'conf_meta': float(conf_np[i]),
                    'conf_top2diff': float(conf_top2diff[i]),
                    'max_softmax': float(max_softmax[i]),
                    'entropy': float(entropy_np[i]),
                    'noise_level': noise_level,
                    'mode': mode,
                }
                if p_fa is not None:
                    for k in range(NUM_CLASSES):
                        row[f'p_FA_{k}'] = float(p_fa[i, k])
                rows.append(row)
            img_idx += len(labels)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained metacognitive head -> per-image CSV.")
    parser.add_argument('--model', choices=['alexnet', 'resnet18', 'vgg19'], required=True)
    parser.add_argument('--mode', choices=list(VALID_MODES), default='fixed_base')
    parser.add_argument('--noise_level', type=float, default=None)
    parser.add_argument('--base_dir', type=str, required=True, help='Frozen base weights dir.')
    parser.add_argument('--meta_dir', type=str, required=True, help='Trained metacognitive head dir.')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_instances', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()

    noise_level = args.noise_level if args.noise_level is not None else DEFAULT_NOISE[args.model]
    batch_size = args.batch_size if args.batch_size is not None else DEFAULT_BATCH[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    columns = base_columns()
    if args.mode == 'dual':
        columns += [f'p_FA_{k}' for k in range(NUM_CLASSES)]

    test_loader = get_balanced_loader(train=False, seed=BALANCED_TEST_SEED, batch_size=batch_size)
    out_csv = os.path.join(args.output_dir, f'{args.model}_{args.mode}.csv')
    print(f"Evaluating {args.model} '{args.mode}' (noise={noise_level}) -> {out_csv}")

    with open(out_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for inst in tqdm(range(args.num_instances), desc="Instances"):
            base_ckpt = os.path.join(args.base_dir, f"{args.model}-224-{inst}-final.pt")
            meta_ckpt = os.path.join(args.meta_dir, f"{args.mode}-{args.model}-{inst}-final.pt")
            if not (os.path.exists(base_ckpt) and os.path.exists(meta_ckpt)):
                print(f"[skip] missing weights for instance {inst}")
                continue
            model = LitMetaModel(arch=args.model, mode=args.mode, noise_level=noise_level)
            model.load_base_weights(base_ckpt)
            model.load_head_state_dict(torch.load(meta_ckpt, map_location='cpu'))
            writer.writerows(eval_instance(inst, model, test_loader, device, noise_level, args.mode))
            fh.flush()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"Done. Wrote {out_csv}")


if __name__ == "__main__":
    main()
