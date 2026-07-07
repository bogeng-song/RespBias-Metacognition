"""
train_metacognitive.py
Train a learned metacognitive head on top of a FROZEN base classifier.

Three learned modules are available (``--mode``); all freeze the base classifier
and train only the metacognitive readout:

    fixed_base   MAIN model (Fig 7): correctness-monitoring head on the frozen
                 penultimate features + top-2 margin + entropy.
    chen         Chen-style white-box correctness model (Supp 6.1): linear probes
                 on intermediate layers + an MLP meta-model.
    dual         Dual-output correctness + auxiliary false-response model (Supp 6.2).

A fourth, feedback-free module (no training, no labels) is in
``scripts_ann/distribution_shift.py``.

Base weights (from ``scripts_ann/train.py``) are expected at
``<base_dir>/<model>-224-<instance>-final.pt``. The trained head is saved to
``<save_dir>/<mode>-<model>-<instance>-final.pt`` (only the head is stored, not
the frozen backbone).

Usage
-----
python scripts_ann/train_metacognitive.py \
    --model alexnet --instance 0 --mode fixed_base \
    --noise_level 1.05 \
    --base_dir ./weights/alexnet/ --save_dir ./weights_meta/alexnet/

# Repeat for --mode chen and --mode dual, and for resnet18 / vgg19 using their
# respective noise levels (see DEFAULT_NOISE below).
"""

import argparse
import os
import random

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from core.meta_modules import LitMetaModel, VALID_MODES
from core.datasets import (base_transform, balanced_train_val_indices,
                           BALANCED_CALIB_SEED)

import warnings
warnings.filterwarnings("ignore")

# Architecture-specific defaults (noise calibrated to ~64% base accuracy; batch
# size follows the training notebooks).
DEFAULT_NOISE = {'alexnet': 1.05, 'resnet18': 0.24, 'vgg19': 0.19}
DEFAULT_BATCH = {'alexnet': 128, 'resnet18': 128, 'vgg19': 64}


def set_random_seed(inst):
    """Deterministic per-instance seed (same scheme as scripts_ann/train.py)."""
    np.random.seed(42)
    inst_seed = int(np.random.choice(range(1, 10000), size=100, replace=False)[inst])
    print(f"Using seed: {inst_seed} for instance {inst}")
    np.random.seed(inst_seed)
    random.seed(inst_seed)
    torch.manual_seed(inst_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(inst_seed)
    pl.seed_everything(inst_seed, workers=True)


def get_train_val_loaders(batch_size, root='./data', seed=BALANCED_CALIB_SEED,
                          num_workers=4):
    """Balanced train/val split from the MNIST TRAIN set (28x28 normalized
    tensors; resize+noise are applied on the GPU inside the model)."""
    ds = datasets.MNIST(root=root, train=True, transform=base_transform(), download=True)
    targets = ds.targets.numpy()
    train_idx, val_idx = balanced_train_val_indices(targets, seed=seed)
    train_loader = DataLoader(Subset(ds, train_idx.tolist()), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(Subset(ds, val_idx.tolist()), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train a learned metacognitive head on a frozen classifier.")
    parser.add_argument('--model', choices=['alexnet', 'resnet18', 'vgg19'], required=True)
    parser.add_argument('--instance', type=int, required=True, help='Instance number (0-59).')
    parser.add_argument('--mode', choices=list(VALID_MODES), default='fixed_base')
    parser.add_argument('--noise_level', type=float, default=None,
                        help='Gaussian noise SD (defaults to the architecture-specific value).')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Directory with the frozen base weights.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save the trained metacognitive head.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--bias_lambda', type=float, default=1.0, help='Dual-output aux loss weight.')
    parser.add_argument('--lambda_probe', type=float, default=0.3, help='Chen-style probe CE weight.')
    args = parser.parse_args()

    noise_level = args.noise_level if args.noise_level is not None else DEFAULT_NOISE[args.model]
    batch_size = args.batch_size if args.batch_size is not None else DEFAULT_BATCH[args.model]

    set_random_seed(args.instance)
    os.makedirs(args.save_dir, exist_ok=True)

    base_ckpt = os.path.join(args.base_dir, f"{args.model}-224-{args.instance}-final.pt")
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt}")

    model = LitMetaModel(arch=args.model, mode=args.mode, noise_level=noise_level,
                         bias_lambda=args.bias_lambda, lambda_probe=args.lambda_probe)
    model.load_base_weights(base_ckpt)
    print(f"Loaded frozen base classifier from {base_ckpt}")

    train_loader, val_loader = get_train_val_loaders(batch_size)

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1,
                        dirpath=args.save_dir,
                        filename=f'{args.mode}-{args.model}-{args.instance}-best'),
    ]
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='auto', devices=1,
                         callbacks=callbacks, log_every_n_steps=50)

    print(f"Training '{args.mode}' metacognitive head for {args.model} "
          f"| instance {args.instance} | noise {noise_level}")
    trainer.fit(model, train_loader, val_loader)

    save_path = os.path.join(args.save_dir, f'{args.mode}-{args.model}-{args.instance}-final.pt')
    payload = model.head_state_dict()
    payload.update({'mode': args.mode, 'arch': args.model,
                    'noise_level': noise_level, 'instance': args.instance})
    torch.save(payload, save_path)
    print(f"Training complete. Saved metacognitive head to {save_path}")


if __name__ == "__main__":
    main()
