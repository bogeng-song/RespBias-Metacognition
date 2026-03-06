"""
PyTorch Lightning model definitions for multi-alternative digit classification.
Includes AlexNet, ResNet18, and VGG19 adapted for 1-channel, 224x224 grayscale images.
"""
import torch
from torch import nn, optim
import pytorch_lightning as pl
from torchvision import models
from torchmetrics import Accuracy

class BaseLitModel(pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.accuracy = Accuracy(num_classes=num_classes, task='multiclass')

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        acc = self.accuracy(outputs.softmax(dim=-1), labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        acc = self.accuracy(outputs.softmax(dim=-1), labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

class LitAlexNet(BaseLitModel):
    def __init__(self, num_classes=10, lr=1e-4):
        super().__init__(num_classes, lr)
        self.model = models.alexnet(weights=None)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)
    def forward(self, x): return self.model(x)

class LitResNet18(BaseLitModel):
    def __init__(self, num_classes=10, lr=1e-4):
        super().__init__(num_classes, lr)
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x): return self.model(x)

class LitVGG19(BaseLitModel):
    def __init__(self, num_classes=10, lr=1e-4):
        super().__init__(num_classes, lr)
        self.model = models.vgg19_bn(weights=None)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)
    def forward(self, x): return self.model(x)

MODEL_DICT = {
    'alexnet': LitAlexNet,
    'resnet18': LitResNet18,
    'vgg19': LitVGG19
}