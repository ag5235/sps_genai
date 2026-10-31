import io
import torch
from PIL import Image
from torchvision.utils import make_grid

from helper_lib.model import Generator
from helper_lib.utils import load_model


class DigitGenerator:
    def __init__(self, weights_path="models/gan_generator.pth"):
        self.device = torch.device("cpu")
        self.generator = Generator()
        # Load the trained weights and switch to evaluation mode
        self.generator = load_model(self.generator, weights_path, self.device)

    def generate(self, num_samples=16, seed=None):
        # Optional seed: the same seed always gives the same digits
        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            noise = torch.randn(num_samples, self.generator.z_dim, device=self.device)
            images = self.generator(noise)

        # Tanh output [-1, 1] -> [0, 1], then arrange the digits on a grid
        images = (images + 1) / 2
        grid = make_grid(images, nrow=min(num_samples, 8), padding=2)

        # Convert to a PNG image, enlarged 4x so the 28x28 digits are easy to see
        array = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        image = Image.fromarray(array)
        image = image.resize((image.width * 4, image.height * 4), Image.Resampling.NEAREST)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()