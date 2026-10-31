import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_mnist_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_samples
from helper_lib.utils import get_device, save_model

device = get_device()
print(f"Using device: {device}")

# MNIST handwritten digits, normalized to [-1, 1]
train_loader = get_mnist_loader('data', batch_size=128, train=True)

# GAN with the assignment's architecture
gan = get_model("GAN")

# Binary cross-entropy loss (slide 10) and one Adam optimizer per network (slide 14)
criterion = nn.BCELoss()
opt_gen = optim.Adam(gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_disc = optim.Adam(gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

gan = train_gan(gan, train_loader, criterion, (opt_gen, opt_disc), device=device, epochs=20)

# Save only the generator: it's the only part the API needs
save_model(gan.generator, 'models/gan_generator.pth')
print("Generator saved to models/gan_generator.pth")

# Save a grid of 20 generated digits to check the results
generate_samples(gan, device, num_samples=20, save_path='gan_samples.png')