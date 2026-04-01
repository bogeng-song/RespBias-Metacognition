import argparse
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.models import MODEL_DICT
from core.datasets import NoisyMNISTDataset

import warnings
warnings.filterwarnings("ignore")

def load_model_weights(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return True
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return False

def find_optimal_noise(model_folder, model_name, noise_range, target_acc, ModelClass, device, num_calib=5):
    print(f"\n=== PHASE 1: Calibration (Target Accuracy: {target_acc*100:.1f}%) ===")
    calib_instances = [i for i in range(60) if os.path.exists(os.path.join(model_folder, f"{model_name}-224-{i}-final.pt"))][:num_calib]

    avg_accs = []
    for noise in noise_range:
        test_loader = DataLoader(NoisyMNISTDataset(root='./data', train=False, noise_level=noise), batch_size=512, shuffle=False, num_workers=2)
        current_noise_accs = []
        for inst_idx in calib_instances:
            path = os.path.join(model_folder, f"{model_name}-224-{inst_idx}-final.pt")
            model = ModelClass(num_classes=10)
            if not load_model_weights(model, path, device): continue

            correct = 0; total = 0
            with torch.no_grad():
                for imgs, lbls in test_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    preds = model(imgs).argmax(dim=1)
                    correct += (preds == lbls).sum().item()
                    total += lbls.size(0)

            current_noise_accs.append(correct/total)

        mean_acc = np.mean(current_noise_accs)
        avg_accs.append(mean_acc)
        print(f"Noise {noise:.2f} -> Mean Acc: {mean_acc*100:.2f}%")

    best_idx = np.argmin(np.abs(np.array(avg_accs) - target_acc))
    best_noise = noise_range[best_idx]
    print(f">>> Optimal Noise Level Found: {best_noise:.2f} (Acc: {avg_accs[best_idx]*100:.2f}%)")
    return best_noise

def generate_behavior_data(model_folder, output_folder, model_name, noise_level, ModelClass, device):
    print(f"\n=== PHASE 2: Generating Data at Noise {noise_level:.2f} ===")
    os.makedirs(output_folder, exist_ok=True)
    test_loader = DataLoader(NoisyMNISTDataset(root='./data', train=False, noise_level=noise_level), batch_size=256, shuffle=False, num_workers=2)

    for inst_idx in tqdm(range(60), desc="Evaluating Base Behavior"):
        ckpt_path = os.path.join(model_folder, f"{model_name}-224-{inst_idx}-final.pt")
        if not os.path.exists(ckpt_path): continue

        model = ModelClass(num_classes=10)
        load_model_weights(model, ckpt_path, device)

        data_rows = []
        with torch.no_grad():
            for batch_i, (imgs, lbls) in enumerate(test_loader):
                imgs = imgs.to(device)
                logits = model(imgs)
                norm_logits = (logits - logits.mean(dim=1, keepdim=True)) / (logits.std(dim=1, keepdim=True) + 1e-8)
                
                top2_vals = torch.topk(logits, k=2, dim=1).values
                conf_top2_batch = (top2_vals[:, 0] - top2_vals[:, 1]).cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                lbls_np, logits_np, norm_logits_np = lbls.numpy(), logits.cpu().numpy(), norm_logits.cpu().numpy()

                for i in range(len(lbls_np)):
                    row = {
                        'instance_id': inst_idx, 'image_index': batch_i * 256 + i,
                        'true_label': lbls_np[i], 'noise_level': noise_level,
                        'prediction': preds[i], 'correct': 1 if preds[i] == lbls_np[i] else 0,
                        'conf_top2': conf_top2_batch[i] 
                    }
                    for d in range(10):
                        row[f'logit_{d}'] = logits_np[i, d]
                        row[f'norm_logit_{d}'] = norm_logits_np[i, d]
                    data_rows.append(row)

        df = pd.DataFrame(data_rows)
        df.to_csv(os.path.join(output_folder, f"behavior_{model_name}_inst{inst_idx}_noise{noise_level:.2f}.csv"), index=False)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Base ANNs (No Metacognition)')
    parser.add_argument('--model', type=str, choices=['alexnet', 'resnet18', 'vgg19'], required=True)
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing trained weights')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--target_acc', type=float, default=0.64, help='Target accuracy for noise calibration')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = MODEL_DICT[args.model]

    noise_range = np.arange(1.0, 1.4, 0.02)
    best_noise = find_optimal_noise(args.model_dir, args.model, noise_range, args.target_acc, ModelClass, device)
    generate_behavior_data(args.model_dir, args.output_dir, args.model, best_noise, ModelClass, device)

if __name__ == "__main__":
    main()