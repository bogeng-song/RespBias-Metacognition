import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

# ──────────────────────────────────────────────────────────────────────────────
# Deterministic, class-balanced subsets + GPU resize/noise
#
# The metacognitive modules (core/meta_modules.py, scripts_ann/*) evaluate on
# class-balanced MNIST subsets with a fixed seed so that per-instance Gaussian
# noise is reproducible, and apply the resize-to-224 + noise on the GPU. These
# helpers implement that shared pipeline:
#
#   ToTensor -> Normalize (28x28, CPU)  ->  interpolate(224) + Gaussian noise (GPU)
#
# BALANCED_CALIB_SEED / BALANCED_TEST_SEED match the training/eval notebooks so a
# rerun reproduces the exact calibration and test images.
# ──────────────────────────────────────────────────────────────────────────────
NUM_CLASSES = 10
BALANCED_CALIB_SEED = 12345   # balanced MNIST-TRAIN subset (calibration / metacog training)
BALANCED_TEST_SEED = 12346    # balanced MNIST-TEST subset (evaluation)


def base_transform():
    """ToTensor -> Normalize only; native 28x28 tensors (resize+noise happen on GPU)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def _targets_numpy(dataset):
    t = dataset.targets
    return t.cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


def balanced_indices_by_label(targets, num_classes=NUM_CLASSES, seed=0):
    """Deterministic indices holding an equal number of examples per class."""
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    per_label = [np.flatnonzero(targets == k) for k in range(num_classes)]
    count = min(len(idx) for idx in per_label)
    selected = []
    for idx in per_label:
        sh = idx.copy()
        rng.shuffle(sh)
        selected.extend(sh[:count])
    return np.asarray(sorted(selected), dtype=np.int64)


def balanced_train_val_indices(targets, num_classes=NUM_CLASSES, seed=BALANCED_CALIB_SEED,
                               val_fraction=0.10):
    """Deterministic class-balanced train/val split carved out of the given targets."""
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    per_label = [np.flatnonzero(targets == k) for k in range(num_classes)]
    min_count = min(len(idx) for idx in per_label)
    val_count = int(round(min_count * val_fraction))
    val_count = max(1, min(val_count, min_count - 1))
    train_count = min_count - val_count
    train_idx, val_idx = [], []
    for idx in per_label:
        sh = idx.copy()
        rng.shuffle(sh)
        train_idx.extend(sh[:train_count])
        val_idx.extend(sh[train_count:train_count + val_count])
    return (np.asarray(sorted(train_idx), dtype=np.int64),
            np.asarray(sorted(val_idx), dtype=np.int64))


def get_balanced_loader(train, seed, batch_size, root='./data', num_workers=2):
    """Class-balanced MNIST subset loader (shuffle=False so seeded GPU noise is
    reproducible per image). Yields native 28x28 normalized tensors."""
    ds = datasets.MNIST(root=root, train=train, transform=base_transform(), download=True)
    idx = balanced_indices_by_label(_targets_numpy(ds), seed=seed)
    return DataLoader(Subset(ds, idx.tolist()), batch_size=batch_size,
                      shuffle=False, num_workers=num_workers, pin_memory=True)


def resize_and_noise(x, noise_level):
    """Resize 28x28 -> 224x224 on the GPU (bilinear), then add Gaussian noise at
    224x224. Seed with ``torch.manual_seed(...)`` beforehand for reproducibility."""
    if x.shape[-1] != 224:
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    if noise_level and noise_level > 0:
        x = x + torch.randn_like(x) * noise_level
    return x


def get_clean_dataloaders(data_dir='./data', batch_size=128):
    """Loaders for standard ANN training on clean MNIST."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST(root=data_dir, train=True, transform=train_transform, download=True)
    test_data = datasets.MNIST(root=data_dir, train=False, transform=test_transform, download=True)
    return (DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))

class NoisyMNISTDataset(Dataset):
    """Dataset for applying visual noise after resizing, mimicking human experiment conditions."""
    def __init__(self, root='./data', train=False, noise_level=0.0):
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.resize = transforms.Resize((224, 224), antialias=True)
        self.mnist = datasets.MNIST(root=root, train=train, transform=self.base_transform, download=True)
        self.noise_level = noise_level

    def __len__(self): 
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        image = self.resize(image)  # Standardized sequence: Resize first
        if self.noise_level > 0:
            image += torch.randn_like(image) * self.noise_level  # Noise second
        return image, label