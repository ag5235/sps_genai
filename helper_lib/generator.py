import torch
from torchvision.utils import save_image

from .model import DiffusionModel, EnergyModel


def langevin_sampling(energy_model, inp_imgs, steps, step_size, noise_std):
    # Langevin dynamics from Module 8 Practical 1: move images downhill in energy
    energy_model.eval()
    for _ in range(steps):
        # 1. Add a little noise (no gradient tracking needed for this part)
        with torch.no_grad():
            noise = torch.randn_like(inp_imgs) * noise_std
            inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)

        # 2. Track gradients with respect to the input images (not the weights)
        inp_imgs.requires_grad_(True)
        energy = energy_model(inp_imgs)
        grads, = torch.autograd.grad(energy, inp_imgs, grad_outputs=torch.ones_like(energy))

        # 3. Take a small step downhill in energy (gradients clipped for stability)
        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)

    return inp_imgs.detach()


def generate_samples(model, device, num_samples=10, diffusion_steps=100, save_path="samples.png",
                     langevin_steps=256, step_size=10.0, noise_std=0.01):
    model = model.to(device)

    if isinstance(model, DiffusionModel):
        # Diffusion: start from standard normal noise and run the reverse diffusion
        images = model.generate(num_samples, diffusion_steps=diffusion_steps)  # already in [0, 1]

    elif isinstance(model, EnergyModel):
        # Energy model: start from uniform noise in [-1, 1] and run Langevin dynamics
        start = torch.rand((num_samples, 3, 32, 32), device=device) * 2 - 1
        images = langevin_sampling(model, start, langevin_steps, step_size, noise_std)
        images = ((images + 1) / 2).clamp(0, 1)  # [-1, 1] -> [0, 1]

    else:
        # GAN (Assignment 3): random latent vectors through the generator
        generator = model.generator
        generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, model.z_dim, device=device)
            images = (generator(noise) + 1) / 2  # Tanh output [-1, 1] -> [0, 1]

    save_image(images.cpu(), save_path, nrow=10)
    print(f"Saved {num_samples} generated images to {save_path}")