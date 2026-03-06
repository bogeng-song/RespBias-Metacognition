import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

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