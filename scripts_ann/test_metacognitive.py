import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.models import MODEL_DICT
from core.datasets import NoisyMNISTDataset

# Ensure test_baseline is importable regardless of how this script is invoked
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_baseline import load_model_weights

import warnings
warnings.filterwarnings("ignore")

def get_distribution_stats(model, loader, device, num_classes=10):
    """Calculates bias via Z-Score Normalized Logits without accuracy feedback."""
    model.eval()
    evidence_stats = {i: [] for i in range(num_classes)}
    chosen_counts = np.zeros(num_classes)
    total_samples = 0

    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device))
            norm_logits = (logits - logits.mean(dim=1, keepdim=True)) / (logits.std(dim=1, keepdim=True) + 1e-8)

            top2 = torch.topk(norm_logits, k=2, dim=1).values
            evidence = (top2[:, 0] - top2[:, 1]).cpu().numpy()
            preds = torch.argmax(norm_logits, dim=1).cpu().numpy()

            for p, ev in zip(preds, evidence):
                chosen_counts[p] += 1
                evidence_stats[p].append(ev)
            total_samples += len(images)

    return chosen_counts, evidence_stats, total_samples

def calculate_shifts(chosen_counts, evidence_stats, total_samples, num_classes=10):
    """Generates the S_k penalty per manuscript formula."""
    shifts = {}
    target_prob = 1.0 / num_classes

    for i in range(num_classes):
        count = chosen_counts[i]
        if count < 10:
            shifts[i] = 0.0
            continue

        p_actual = count / total_samples
        if p_actual > target_prob:
            keep_ratio = target_prob / p_actual
            percentile_to_cut = max(0, min(100, (1 - keep_ratio) * 100))
            shifts[i] = np.percentile(np.array(evidence_stats[i]), percentile_to_cut)
        else:
            shifts[i] = 0.0

    return shifts

def process_instance(instance_num, args, ModelClass, device):
    model_path = os.path.join(args.model_dir, f"{args.model}-224-{instance_num}-final.pt")
    if not os.path.exists(model_path): return

    model = ModelClass(num_classes=10)
    load_model_weights(model, model_path, device)

    # 1. Metacognitive Calibration (Unsupervised parameter learning on Train set)
    calib_loader = DataLoader(NoisyMNISTDataset('./data', train=True, noise_level=args.noise_level), batch_size=args.batch_size, shuffle=False, num_workers=2)
    counts, ev_stats, total = get_distribution_stats(model, calib_loader, device)
    shifts = calculate_shifts(counts, ev_stats, total)

    # 2. Metacognitive Testing (Apply S_k penalty on Test Set)
    test_loader = DataLoader(NoisyMNISTDataset('./data', train=False, noise_level=args.noise_level), batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    data_rows = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            logits = model(images.to(device))
            norm_logits = (logits - logits.mean(dim=1, keepdim=True)) / (logits.std(dim=1, keepdim=True) + 1e-8)

            top2 = torch.topk(norm_logits, k=2, dim=1).values
            conf_naive = (top2[:, 0] - top2[:, 1]).cpu().numpy()
            preds = torch.argmax(norm_logits, dim=1).cpu().numpy()
            labels_np, probs_np = labels.numpy(), F.softmax(logits, dim=1).cpu().numpy()

            for i in range(len(labels_np)):
                p = preds[i]
                c_naive = conf_naive[i]
                s = shifts[p]

                data_rows.append({
                    'instance': instance_num,
                    'image_index': batch_idx * args.batch_size + i,
                    'true_label': labels_np[i],
                    'response': p,
                    'accuracy': 1 if p == labels_np[i] else 0,
                    'conf_naive': c_naive,
                    'conf_final': c_naive - s,      # Self-monitoring penalty applied!
                    'shift_applied': s,
                    'bias_metric': counts[p] / total,
                    'conf_softmax': np.max(probs_np[i])
                })

    df = pd.DataFrame(data_rows)
    df.to_csv(os.path.join(args.output_dir, f'bias_correction_{args.model}_inst{instance_num}.csv'), index=False)

def main():
    parser = argparse.ArgumentParser(description='Apply Metacognitive Module (Bias Shift) to ANNs')
    parser.add_argument('--model', type=str, choices=['alexnet', 'resnet18', 'vgg19'], required=True)
    parser.add_argument('--model_dir', type=str, required=True, help='Directory with trained models')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for bias-corrected CSVs')
    parser.add_argument('--noise_level', type=float, required=True, help='Noise level from test_baseline.py (e.g., 1.19)')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = MODEL_DICT[args.model]

    print(f"Starting Metacognitive Module for {args.model} at Noise {args.noise_level}...")
    for i in tqdm(range(60), desc="Processing instances"):
        process_instance(i, args, ModelClass, device)

if __name__ == "__main__":
    main()