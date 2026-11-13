import torch
import torch.nn as nn

from helper_lib.data_loader import get_cifar10_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion
from helper_lib.generator import generate_samples
from helper_lib.utils import get_device

device = get_device()
print(f"Using device: {device}")

# CIFAR-10, pixels in [0, 1] (the diffusion model normalizes them itself), batch size 64 as in Practical 2
train_loader = get_cifar10_loader('data', batch_size=64, train=True)
val_loader = get_cifar10_loader('data', batch_size=64, train=False)

# UNet + offset cosine schedule, wrapped in the DiffusionModel (Module 8 Practical 2)
diffusion_model = get_model("Diffusion")

# Per-channel mean and std of the training images, computed the same way as the class notebook
mean = torch.zeros(3)
std = torch.zeros(3)
total = 0
for imgs, _ in train_loader:
    flat = imgs.view(imgs.size(0), 3, -1)
    mean += flat.mean(dim=(0, 2)) * imgs.size(0)
    std += flat.std(dim=(0, 2)) * imgs.size(0)
    total += imgs.size(0)
mean /= total
std /= total
print("Normalization stats - Mean:", mean, "Std:", std)
diffusion_model.set_normalizer(mean, std)

# Same optimizer and loss as the class notebook
optimizer = torch.optim.AdamW(diffusion_model.network.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.L1Loss()

diffusion_model = train_diffusion(diffusion_model, train_loader, loss_fn, optimizer,
                                  device=device, epochs=10, val_loader=val_loader)

# Save the full model (UNet, EMA UNet and normalization stats) for the API
torch.save(diffusion_model.state_dict(), 'models/diffusion_cifar10.pth')
print("Diffusion model saved to models/diffusion_cifar10.pth")

# Save 20 generated images to check the results
generate_samples(diffusion_model, device, num_samples=20, diffusion_steps=100, save_path='diffusion_samples.png')