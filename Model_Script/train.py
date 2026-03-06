import argparse
import os
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from core.models import MODEL_DICT
from core.datasets import get_clean_dataloaders

import warnings
warnings.filterwarnings("ignore")

def set_random_seed(inst):
    np.random.seed(42)
    random_seeds = np.random.choice(range(1, 10000), size=100, replace=False)
    inst_seed = int(random_seeds[inst])
    print(f"Using seed: {inst_seed} for Instance {inst}")

    np.random.seed(inst_seed)
    random.seed(inst_seed)
    torch.manual_seed(inst_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(inst_seed)
    pl.seed_everything(inst_seed, workers=True)

def main():
    parser = argparse.ArgumentParser(description='Train ANNs on MNIST (224x224)')
    parser.add_argument('--model', type=str, choices=['alexnet', 'resnet18', 'vgg19'], required=True)
    parser.add_argument('--instance', type=int, required=True, help='Instance number (0-59)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_dir', type=str, default='./weights')
    args = parser.parse_args()

    set_random_seed(args.instance)
    os.makedirs(args.save_dir, exist_ok=True)

    wandb_logger = WandbLogger(project=f'{args.model}_MNIST_224', name=f"{args.model}_inst_{args.instance}")
    
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=5, verbose=True, mode='min'),
        ModelCheckpoint(monitor='val_acc', dirpath=args.save_dir, filename=f'{args.model}-224-{args.instance}', save_top_k=1, mode='max')
    ]

    ModelClass = MODEL_DICT[args.model]
    model = ModelClass(num_classes=10)

    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator="auto", devices=1,
        callbacks=callbacks, logger=wandb_logger, log_every_n_steps=50
    )

    train_loader, test_loader = get_clean_dataloaders(batch_size=args.batch_size)
    print(f"Starting training for {args.model.upper()} | Instance {args.instance}...")
    trainer.fit(model, train_loader, test_loader)

    final_model_path = os.path.join(args.save_dir, f'{args.model}-224-{args.instance}-final.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"Training Complete. Saved to {final_model_path}")

if __name__ == "__main__":
    main()