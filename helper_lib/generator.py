import torch
from torchvision.utils import save_image


def generate_samples(model, device, num_samples=10, save_path="gan_samples.png"):
    # Generate num_samples points in the latent space, run the generator, and save the images on a grid
    generator = model.generator.to(device)
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, model.z_dim, device=device)
        images = generator(noise)
    images = (images + 1) / 2  # Tanh output [-1, 1] -> [0, 1] for display
    save_image(images.cpu(), save_path, nrow=10)
    print(f"Saved {num_samples} generated digits to {save_path}")