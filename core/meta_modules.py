"""
meta_modules.py
Learned metacognitive "second-order" readouts attached to a FROZEN base
classifier. These modules change only the confidence assigned to each response,
never the perceptual decision itself: the backbone and its classification head
are frozen, so the perceptual responses, the digit-level false-alarm rate (FAR),
and the digit-level accuracy are byte-identical across modules. Only the
confidence readout differs.

Two modules are provided, matching the manuscript:

    'logit_only'  MAIN model (Figure 6). A correctness-monitoring head that sees
                  ONLY the frozen classifier's C raw final-layer logits, and is
                  trained with a binary objective to predict whether the frozen
                  classifier's response was correct.

    'pen_only'    SUPPLEMENTARY model. Identical in every respect except that the
                  head sees ONLY the penultimate-layer representation (with the
                  same Dropout(0.5) applied), never the logits.

The two share everything else -- frozen backbone, noisy training images at the
architecture's calibrated noise level, BCE-on-correctness plus a small balance
regulariser, Adam(1e-4, wd 1e-3), and ReduceLROnPlateau. They differ ONLY in
what feeds the head, which is what makes them a clean comparison: any difference
in the bias-confidence relationship is attributable to the input representation
rather than to the training procedure.

Note on dropout: 'pen_only' applies Dropout(0.5) to the penultimate feature.
'logit_only' does not, because at C = 10 a p = 0.5 mask would destroy the signal
and there is no high-dimensional representation to regularise.

The module is architecture-general across AlexNet, ResNet18 and VGG19-BN; only
the backbone surgery and penultimate-feature extraction differ (see ARCH_CONFIG).
"""

import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from torchmetrics import Accuracy


# Per-architecture configuration:
#   feat_dim : width of the penultimate representation (the 'pen_only' head input)
ARCH_CONFIG = {
    'alexnet':  {'feat_dim': 4096},
    'resnet18': {'feat_dim': 512},
    'vgg19':    {'feat_dim': 4096},
}

VALID_MODES = ('logit_only', 'pen_only')

# Gaussian noise SD (added at 224x224, after normalisation) calibrated so that
# each architecture's base accuracy lands near the human level of ~64%.
CALIBRATED_NOISE = {'alexnet': 1.05, 'resnet18': 0.24, 'vgg19': 0.19}


def meta_in_dim(mode, feat_dim, num_classes):
    """Input width of the metacognitive head for each mode."""
    if mode == 'logit_only':
        return num_classes
    if mode == 'pen_only':
        return feat_dim
    raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Backbone construction / loading
# ──────────────────────────────────────────────────────────────────────────────
def build_backbone(arch, num_classes=10):
    """Build a bare (non-Lightning) base classifier with the 1-channel / 10-class
    surgery, matching the base models trained by ``scripts_ann/train.py``."""
    if arch == 'alexnet':
        base = models.alexnet(weights=None)
        base.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        base.classifier[6] = nn.Linear(base.classifier[6].in_features, num_classes)
    elif arch == 'resnet18':
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
    elif arch == 'vgg19':
        base = models.vgg19_bn(weights=None)
        base.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        base.classifier[6] = nn.Linear(base.classifier[6].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return base


def load_base_into_backbone(backbone, ckpt_path):
    """Load base-classifier weights saved by ``train.py`` (a LightningModule whose
    only submodule was ``self.model = models.<arch>(...)``). Strip the ``model.``
    prefix and ignore unrelated buffers (accuracy metric state, etc.)."""
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    fixed = {}
    for k, v in sd.items():
        nk = k[len('model.'):] if k.startswith('model.') else k
        fixed[nk] = v
    backbone.load_state_dict(fixed, strict=False)
    return backbone


# ──────────────────────────────────────────────────────────────────────────────
# Metacognitive LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class LitMetaModel(pl.LightningModule):
    """Learned metacognitive readout on a frozen base classifier.

    Parameters
    ----------
    arch : {'alexnet', 'resnet18', 'vgg19'}
    mode : {'logit_only', 'pen_only'}
    noise_level : Gaussian noise SD added at 224x224 (architecture-specific;
        see CALIBRATED_NOISE, calibrated to ~64% base accuracy).
    """

    def __init__(self, arch='alexnet', mode='logit_only', noise_level=None,
                 num_classes=10, lr=1e-4, weight_decay=1e-3):
        super().__init__()
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        if arch not in ARCH_CONFIG:
            raise ValueError(f"arch must be one of {tuple(ARCH_CONFIG)}, got {arch!r}")
        if noise_level is None:
            noise_level = CALIBRATED_NOISE[arch]
        self.save_hyperparameters()

        self.feat_dim = ARCH_CONFIG[arch]['feat_dim']
        self.backbone = build_backbone(arch, num_classes=num_classes)
        self.accuracy = Accuracy(num_classes=num_classes, task='multiclass')

        # Applied to the penultimate feature in 'pen_only' only; constructed
        # unconditionally so a checkpoint saved by either mode loads cleanly.
        self.dropout = nn.Dropout(p=0.5)
        self.meta_head = nn.Sequential(
            nn.Linear(meta_in_dim(mode, self.feat_dim, num_classes), 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        self.freeze_backbone()

    # -- freezing --------------------------------------------------------------
    def freeze_backbone(self):
        """Freeze every backbone parameter. Called at construction."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def load_base_weights(self, ckpt_path):
        """Load a trained base classifier and re-freeze it."""
        load_base_into_backbone(self.backbone, ckpt_path)
        self.freeze_backbone()
        return self

    def train(self, mode=True):
        """Keep the frozen backbone in eval mode so its own dropout/BN stay off."""
        super().train(mode)
        self.backbone.eval()
        return self

    # -- backbone feature extraction ------------------------------------------
    def _backbone_forward(self, x):
        """Return (penultimate_feature, logits) for the configured architecture."""
        arch = self.hparams.arch
        if arch in ('alexnet', 'vgg19'):
            f = self.backbone.features(x)
            f = self.backbone.avgpool(f)
            f = torch.flatten(f, 1)
            for layer in list(self.backbone.classifier)[:-1]:
                f = layer(f)
            logits = self.backbone.classifier[-1](f)
        else:  # resnet18
            b = self.backbone
            f = b.conv1(x); f = b.bn1(f); f = b.relu(f); f = b.maxpool(f)
            f = b.layer1(f); f = b.layer2(f); f = b.layer3(f); f = b.layer4(f)
            f = b.avgpool(f); f = torch.flatten(f, 1)
            logits = b.fc(f)
        return f, logits

    @staticmethod
    def uncertainty_features(logits):
        """Top-2 softmax margin and softmax entropy (each shape [B, 1]).

        These are the deterministic standard-ANN confidence baselines. They are
        never fed to the metacognitive head in either mode; they are computed so
        the evaluation CSV can carry them alongside the learned readout, which is
        what lets one CSV reproduce both the standard-ANN and the metacognitive
        result for the same instance.
        """
        probs = F.softmax(logits, dim=-1)
        top2 = torch.topk(probs, 2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
        entropy = (-probs * probs.clamp_min(1e-12).log()).sum(-1, keepdim=True)
        return margin, entropy

    def _meta_forward(self, feat, logits):
        """Build the head input. The two modes differ ONLY here."""
        if self.hparams.mode == 'logit_only':
            meta_in = logits                    # the C raw logits, no dropout
        else:                                   # pen_only
            meta_in = self.dropout(feat)        # penultimate representation only
        return self.meta_head(meta_in).squeeze(-1)

    # -- forward ---------------------------------------------------------------
    def forward(self, x):
        """Run resize+noise, the frozen backbone, and the metacognitive head.

        Returns
        -------
        logits  : [B, C] base-classifier logits (frozen).
        conf    : [B]    scalar meta-confidence in (0, 1).
        margin  : [B]    top-2 softmax margin of the base logits.
        entropy : [B]    softmax entropy of the base logits.
        """
        from core.datasets import resize_and_noise
        x = resize_and_noise(x, self.hparams.noise_level)
        with torch.no_grad():
            feat, logits = self._backbone_forward(x)
        feat, logits = feat.detach(), logits.detach()
        conf = self._meta_forward(feat, logits)
        margin, entropy = self.uncertainty_features(logits)
        return logits, conf, margin.squeeze(-1), entropy.squeeze(-1)

    # -- losses ----------------------------------------------------------------
    def _compute_loss(self, batch):
        x, y = batch
        logits, conf, _, _ = self(x)
        preds = logits.argmax(1)
        correct = (preds == y).float()          # the correctness target
        loss_meta = F.binary_cross_entropy(conf, correct)
        # Small balance regulariser: discourages the head from collapsing onto
        # the base rate when accuracy is far from 50%.
        reg = 1e-4 * (conf.mean() - 0.5).abs()
        acc_frozen = self.accuracy(logits.softmax(-1), y)
        return loss_meta + reg, loss_meta, acc_frozen

    def training_step(self, batch, batch_idx):
        loss, loss_meta, acc = self._compute_loss(batch)
        self.log_dict({'train_loss': loss, 'train_meta_loss': loss_meta,
                       'train_acc_frozen': acc},
                      prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_meta, acc = self._compute_loss(batch)
        # Checkpoint on the pure correctness-monitoring BCE.
        self.log_dict({'val_loss': loss_meta, 'val_total_loss': loss,
                       'val_acc_frozen': acc},
                      prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        params = [p for p in self.meta_head.parameters() if p.requires_grad]
        opt = optim.Adam(params, lr=self.hparams.lr,
                         weight_decay=self.hparams.weight_decay)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                     factor=0.5, patience=3)
        return {'optimizer': opt,
                'lr_scheduler': {'scheduler': sched, 'monitor': 'val_loss'}}

    # -- checkpoint helpers (save only the trainable head, not the backbone) ---
    def head_state_dict(self):
        return {'meta_head': self.meta_head.state_dict()}

    def load_head_state_dict(self, state):
        self.meta_head.load_state_dict(state['meta_head'])
        return self
