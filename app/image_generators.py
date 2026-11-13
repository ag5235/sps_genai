import io
import torch
from PIL import Image
from torchvision.utils import make_grid

from helper_lib.model import get_model, EnergyModel
from helper_lib.generator import langevin_sampling


def images_to_png(images, nrow=8, scale=4):
    # Arrange images (values in [0, 1]) on a grid and return them as PNG bytes, enlarged 4x
    grid = make_grid(images.cpu(), nrow=nrow, padding=2)
    array = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
    image = Image.fromarray(array)
    image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class DiffusionImageGenerator:
    def __init__(self, weights_path="models/diffusion_cifar10.pth"):
        self.device = torch.device("cpu")
        self.model = get_model("Diffusion")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def generate(self, num_samples=8, diffusion_steps=50, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        # Start from standard normal noise and run the reverse diffusion (EMA network)
        images = self.model.generate(num_samples, diffusion_steps=diffusion_steps)
        return images_to_png(images, nrow=min(num_samples, 8))


class EnergyImageGenerator:
    def __init__(self, weights_path="models/ebm_cifar10.pth"):
        self.device = torch.device("cpu")
        self.model = EnergyModel(num_channels=3)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def generate(self, num_samples=8, steps=256, step_size=10.0, noise_std=0.01, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        # Start from uniform noise in [-1, 1] and move downhill in energy (Langevin dynamics)
        start = torch.rand((num_samples, 3, 32, 32)) * 2 - 1
        images = langevin_sampling(self.model, start, steps, step_size, noise_std)
        images = ((images + 1) / 2).clamp(0, 1)  # [-1, 1] -> [0, 1]
        return images_to_png(images, nrow=min(num_samples, 8))