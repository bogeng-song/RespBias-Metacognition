"""
meta_modules.py
Learned metacognitive "second-order" readouts attached to a FROZEN base
classifier. These modules change only the confidence assigned to each response,
never the perceptual decision itself (the backbone and classification head are
frozen), isolating the effect of a metacognitive confidence readout.

Three learned modules are provided (selected via ``mode``); a fourth,
feedback-free module lives in ``scripts_ann/distribution_shift.py``:

    'fixed_base'  MAIN model (manuscript Fig 7). A correctness-monitoring head
                  that receives the frozen backbone's penultimate representation
                  plus two uncertainty features (top-2 softmax margin, softmax
                  entropy) and is trained with a binary objective to predict
                  whether the frozen classifier's response was correct.

    'chen'        Chen-style white-box correctness-monitoring model
                  (Chen et al., 2019; manuscript Supp 6.1). Linear classifier
                  probes are attached to intermediate layers of the frozen
                  network; their softmax outputs plus the base softmax feed an
                  MLP meta-model that predicts trial-level correctness. Probes
                  are trained with cross-entropy on the true label; the meta-head
                  with BCE on correctness.

    'dual'        Dual-output correctness + auxiliary false-response model
                  (manuscript Supp 6.2). A shared trunk feeds two heads: a
                  correctness head (used as confidence) and an auxiliary
                  false-response head that predicts class-specific error events,
                  encouraging the shared representation to encode response bias.

The module is architecture-general across AlexNet, ResNet18, and VGG19-BN; only
the backbone surgery, penultimate-feature extraction, and probe hook points
differ (see ARCH_CONFIG).
"""

import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from torchmetrics import Accuracy


# Per-architecture configuration:
#   feat_dim       : width of the penultimate representation fed to the meta head
#   probe_channels : channel counts at the Chen-style probe hook points
ARCH_CONFIG = {
    'alexnet':  {'feat_dim': 4096, 'probe_channels': [64, 192, 256]},
    'resnet18': {'feat_dim': 512,  'probe_channels': [64, 128, 256, 512]},
    'vgg19':    {'feat_dim': 4096, 'probe_channels': [64, 128, 256, 512, 512]},
}

VALID_MODES = ('fixed_base', 'chen', 'dual')


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
    mode : {'fixed_base', 'chen', 'dual'}
    noise_level : Gaussian noise SD added at 224x224 (architecture-specific,
        calibrated to ~64% base accuracy).
    """

    def __init__(self, arch='alexnet', mode='fixed_base', noise_level=1.05,
                 num_classes=10, lr=1e-4, weight_decay=1e-3,
                 bias_lambda=1.0, lambda_probe=0.3):
        super().__init__()
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        if arch not in ARCH_CONFIG:
            raise ValueError(f"arch must be one of {tuple(ARCH_CONFIG)}, got {arch!r}")
        self.save_hyperparameters()

        cfg = ARCH_CONFIG[arch]
        self.feat_dim = cfg['feat_dim']
        self.probe_channels = cfg['probe_channels']

        self.backbone = build_backbone(arch, num_classes=num_classes)
        self.accuracy = Accuracy(num_classes=num_classes, task='multiclass')

        if mode in ('fixed_base', 'dual'):
            self.dropout = nn.Dropout(p=0.5)
        if mode == 'fixed_base':
            self.meta_head = nn.Sequential(
                nn.Linear(self.feat_dim + 2, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid(),
            )
        elif mode == 'dual':
            self.meta_trunk = nn.Sequential(
                nn.Linear(self.feat_dim + 2, 512),
                nn.ReLU(),
            )
            self.conf_head = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
            self.bias_head = nn.Sequential(nn.Linear(512, num_classes), nn.Sigmoid())
        elif mode == 'chen':
            self.probes = nn.ModuleList([
                nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                              nn.Linear(c, num_classes))
                for c in self.probe_channels
            ])
            meta_in_dim = (len(self.probes) + 1) * num_classes
            self.meta_head = nn.Sequential(
                nn.Linear(meta_in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Sigmoid(),
            )
            self._probe_inputs = [None] * len(self.probes)
            self._hook_handles = []
            self._attach_hooks()

        self.freeze_backbone()

    # -- backbone freezing -----------------------------------------------------
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode=True):
        """Keep the frozen backbone in eval mode even when Lightning switches the
        module to train (so backbone dropout / BN stay fixed)."""
        super().train(mode)
        self.backbone.eval()
        return self

    def load_base_weights(self, ckpt_path):
        load_base_into_backbone(self.backbone, ckpt_path)
        self.freeze_backbone()
        return self

    # -- Chen probe hooks ------------------------------------------------------
    def _attach_hooks(self):
        if self.hparams.arch in ('alexnet', 'vgg19'):
            targets = [m for m in self.backbone.features if isinstance(m, nn.MaxPool2d)]
        else:  # resnet18
            targets = [self.backbone.layer1, self.backbone.layer2,
                       self.backbone.layer3, self.backbone.layer4]
        assert len(targets) == len(self.probe_channels), (
            f"{self.hparams.arch}: expected {len(self.probe_channels)} probe sites, "
            f"found {len(targets)}")
        for i, mod in enumerate(targets):
            def make_hook(idx):
                def hook(module, inp, out):
                    self._probe_inputs[idx] = out
                return hook
            self._hook_handles.append(mod.register_forward_hook(make_hook(i)))

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
    def _uncertainty_features(logits):
        """Top-2 softmax margin and softmax entropy (each shape [B, 1])."""
        probs = F.softmax(logits, dim=-1)
        top2 = torch.topk(probs, 2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
        entropy = (-probs * probs.clamp_min(1e-12).log()).sum(-1, keepdim=True)
        return margin, entropy

    # -- forward ---------------------------------------------------------------
    def forward(self, x):
        """Run resize+noise, the frozen backbone, and the metacognitive head.

        Returns
        -------
        logits : [B, C] base-classifier logits (frozen).
        conf   : [B]    scalar meta-confidence in (0, 1).
        aux    : mode-dependent extra output
                 - 'fixed_base' : None
                 - 'chen'       : list of per-probe logit tensors ([B, C] each)
                 - 'dual'       : p_fa [B, C] auxiliary false-response probabilities
        margin, entropy : [B] uncertainty summaries of the base logits.
        """
        from core.datasets import resize_and_noise
        x = resize_and_noise(x, self.hparams.noise_level)

        if self.hparams.mode == 'chen':
            self._probe_inputs = [None] * len(self.probes)
            with torch.no_grad():
                _, logits = self._backbone_forward(x)
            logits = logits.detach()
            probe_logits = [probe(inp.detach()) for probe, inp in
                            zip(self.probes, self._probe_inputs)]
            meta_in = torch.cat([F.softmax(pl, dim=-1) for pl in probe_logits]
                                + [F.softmax(logits, dim=-1)], dim=1)
            conf = self.meta_head(meta_in).squeeze(-1)
            margin, entropy = self._uncertainty_features(logits)
            return logits, conf, probe_logits, margin.squeeze(-1), entropy.squeeze(-1)

        with torch.no_grad():
            feat, logits = self._backbone_forward(x)
        feat, logits = feat.detach(), logits.detach()
        margin, entropy = self._uncertainty_features(logits)
        # "soft-detach": let 90% of the gradient flow from the scalar features.
        meta_in = torch.cat([
            self.dropout(feat),
            margin + 0.1 * margin.detach(),
            entropy + 0.1 * entropy.detach(),
        ], dim=1)

        if self.hparams.mode == 'fixed_base':
            conf = self.meta_head(meta_in).squeeze(-1)
            aux = None
        else:  # dual
            h = self.meta_trunk(meta_in)
            conf = self.conf_head(h).squeeze(-1)
            aux = self.bias_head(h)
        return logits, conf, aux, margin.squeeze(-1), entropy.squeeze(-1)

    # -- losses ----------------------------------------------------------------
    def _compute_loss(self, batch):
        x, y = batch
        logits, conf, aux, _, _ = self(x)
        preds = logits.argmax(1)
        correct = (preds == y).float()
        loss_meta = F.binary_cross_entropy(conf, correct)
        reg = 1e-4 * (conf.mean() - 0.5).abs()
        loss = loss_meta + reg

        if self.hparams.mode == 'dual':
            # auxiliary false-response target: one-hot at the (wrongly) chosen
            # class on incorrect trials, all-zeros on correct trials.
            fa_target = torch.zeros_like(aux)
            wrong = preds != y
            if wrong.any():
                fa_target[wrong, preds[wrong]] = 1.0
            loss_bias = F.binary_cross_entropy(aux, fa_target)
            loss = loss + self.hparams.bias_lambda * loss_bias
        elif self.hparams.mode == 'chen':
            probe_ce = torch.stack([F.cross_entropy(pl, y) for pl in aux]).mean()
            loss = loss + self.hparams.lambda_probe * probe_ce

        acc_frozen = self.accuracy(logits.softmax(-1), y)
        return loss, loss_meta, acc_frozen

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
        params = [p for p in self.parameters() if p.requires_grad]
        opt = optim.Adam(params, lr=self.hparams.lr,
                         weight_decay=self.hparams.weight_decay)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                     factor=0.5, patience=3)
        return {'optimizer': opt,
                'lr_scheduler': {'scheduler': sched, 'monitor': 'val_loss'}}

    # -- checkpoint helpers (save only the trainable head, not the backbone) ---
    def head_state_dict(self):
        mode = self.hparams.mode
        if mode == 'fixed_base':
            return {'meta_head': self.meta_head.state_dict()}
        if mode == 'dual':
            return {'meta_trunk': self.meta_trunk.state_dict(),
                    'conf_head': self.conf_head.state_dict(),
                    'bias_head': self.bias_head.state_dict()}
        # chen
        return {'probes': self.probes.state_dict(),
                'meta_head': self.meta_head.state_dict()}

    def load_head_state_dict(self, state):
        mode = self.hparams.mode
        if mode == 'fixed_base':
            self.meta_head.load_state_dict(state['meta_head'])
        elif mode == 'dual':
            self.meta_trunk.load_state_dict(state['meta_trunk'])
            self.conf_head.load_state_dict(state['conf_head'])
            self.bias_head.load_state_dict(state['bias_head'])
        else:  # chen
            self.probes.load_state_dict(state['probes'])
            self.meta_head.load_state_dict(state['meta_head'])
        return self
