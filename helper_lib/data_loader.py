import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loader(data_dir, batch_size=32, train=True, image_size=None):
    # Same transform as the class notebook, plus an optional resize
    transform_steps = []
    if image_size is not None:
        transform_steps.append(transforms.Resize((image_size, image_size)))
    transform_steps.append(transforms.ToTensor())
    transform = transforms.Compose(transform_steps)

    dataset = datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=transform
    )

    # Shuffle the training data, keep the test data in order (same as class)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader

from torch.utils.data import random_split


def get_train_val_loaders(data_dir, batch_size=32, image_size=None, val_size=5000):
    # Same transform as get_data_loader
    transform_steps = []
    if image_size is not None:
        transform_steps.append(transforms.Resize((image_size, image_size)))
    transform_steps.append(transforms.ToTensor())
    transform = transforms.Compose(transform_steps)

    full_train = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    # Hold back val_size training images as a validation set (fixed seed so the split is repeatable)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(
        full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader