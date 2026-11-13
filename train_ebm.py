import torch

from helper_lib.data_loader import get_cifar10_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_ebm
from helper_lib.generator import generate_samples
from helper_lib.utils import get_device, save_model

device = get_device()
print(f"Using device: {device}")

# CIFAR-10 with pixels in [-1, 1], batch size 128, as in Module 8 Practical 1
train_loader = get_cifar10_loader('data', batch_size=128, train=True, normalize=True)

# Energy model (CNN with swish activations)
ebm = get_model("EBM")

# Same optimizer and hyperparameters as the class notebook
optimizer = torch.optim.Adam(ebm.parameters(), lr=0.0001, betas=(0.0, 0.999))
ebm = train_ebm(ebm, train_loader, optimizer, device=device, epochs=30,
                alpha=0.1, steps=60, step_size=10, noise=0.005)

# Save the trained energy model for the API
save_model(ebm, 'models/ebm_cifar10.pth')
print("Energy model saved to models/ebm_cifar10.pth")

# Generate 20 images with 256 Langevin steps (same as the notebook's final cell)
generate_samples(ebm, device, num_samples=20, save_path='ebm_samples.png')