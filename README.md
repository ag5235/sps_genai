# sps_genai

FastAPI project for the Applied Generative AI course (Columbia SPS).
Arnav Goenka (ag5235)

- **Assignment 1:** bigram text generation and spaCy word embeddings (`/generate`, `/embedding`, `/similarity`)
- **Assignment 2:** CNN image classifier trained on CIFAR-10 (`/classify`)
- **Assignment 3:** GAN that generates handwritten digits, trained on MNIST (`/generate_digits`)
- **Assignment 4:** Diffusion model and Energy-Based model that generate images, trained on CIFAR-10 (`/generate_diffusion`, `/generate_ebm`)

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs

## Test the image generators — Assignment 4

Both endpoints return a PNG image with a grid of generated CIFAR-10-style images. In `/docs`, open the endpoint, click **Try it out**, then **Execute**. Generation runs on the CPU and takes a few seconds.

**Diffusion model: `GET /generate_diffusion`**
- `num_samples` (optional, 1–32, default 8): how many images to generate
- `diffusion_steps` (optional, 1–200, default 50): number of reverse-diffusion (denoising) steps
- `seed` (optional): the same seed always gives the same images

```
http://127.0.0.1:8000/generate_diffusion?num_samples=8&diffusion_steps=50&seed=42
```

**Energy-Based model: `GET /generate_ebm`**
- `num_samples` (optional, 1–32, default 8): how many images to generate
- `langevin_steps` (optional, 1–1000, default 256): number of Langevin dynamics steps
- `seed` (optional): the same seed always gives the same images

```
http://127.0.0.1:8000/generate_ebm?num_samples=8&langevin_steps=256&seed=42
```

Example outputs from the trained models: `diffusion_samples.png`, `ebm_samples.png`

## Test the GAN digit generator (`/generate_digits`) — Assignment 3

In `/docs`, open **GET /generate_digits**, click **Try it out**, and click **Execute**.

```
http://127.0.0.1:8000/generate_digits?num_samples=16&seed=42
```

Example output: `gan_samples.png`

## Test the image classifier (`/classify`) — Assignment 2

In `/docs`, open **POST /classify**, click **Try it out**, upload an image (e.g. `sample_images/0_cat.png`) and click **Execute**.

```bash
curl -X POST http://127.0.0.1:8000/classify -F "file=@sample_images/0_cat.png"
```

## Diffusion model (Assignment 4)

Based on Module 8 Practical 2. Defined as `UNet` and `DiffusionModel` in `helper_lib/model.py`.

- **Network:** UNet with sinusoidal noise embedding, 3 down blocks (32 → 16 → 8 → 4), 2 middle residual blocks, 3 up blocks with skip connections
- **Training:** the network predicts the noise added to an image (L1 loss), with an offset cosine diffusion schedule and continuous diffusion times
- **Generation:** reverse diffusion from standard normal noise, using an exponential moving average (EMA) copy of the network
- **Settings:** AdamW (lr = 0.001, weight decay = 0.0001), batch size 64, gradient clipping (max norm 1.0), EMA decay 0.8
- **Training:** 10 epochs; the checkpoint with the lowest validation loss (epoch 6, val L1 = 0.1625) is used
- Weights and normalization statistics saved to `models/diffusion_cifar10.pth`

```bash
uv run python train_diffusion.py
```

## Energy-Based model (Assignment 4)

Based on Module 8 Practical 1. Defined as `EnergyModel` in `helper_lib/model.py`.

- **Network:** CNN with swish activations (32 → 16 → 8 → 4 → 2), outputting one energy value per image
- **Training:** contrastive divergence (lower the energy of real images, raise it for generated ones) plus a regularization term (alpha = 0.1); generated images come from Langevin dynamics with a replay buffer (60 steps, step size 10, noise 0.005)
- **Generation:** 256 Langevin steps from uniform noise in [-1, 1]
- **Settings:** Adam (lr = 0.0001, betas = (0.0, 0.999)), batch size 128, gradient clipping (max norm 0.1), 10 epochs
- Weights saved to `models/ebm_cifar10.pth`

```bash
uv run python train_ebm.py
```

## GAN (Assignment 3)

Defined as `Generator`, `Discriminator` and `GAN` in `helper_lib/model.py`. Generator: noise (100) → fully connected → 128 × 7 × 7 → ConvTranspose2D + BatchNorm + ReLU → 64 × 14 × 14 → ConvTranspose2D + Tanh → 1 × 28 × 28. Discriminator: Conv2D + LeakyReLU → Conv2D + BatchNorm + LeakyReLU → Linear + Sigmoid. Trained on MNIST for 20 epochs with BCE loss and Adam (lr = 0.0002, betas = (0.5, 0.999)).

```bash
uv run python train_gan.py
```

## CNN classifier (Assignment 2)

Defined as `AssignmentCNN` in `helper_lib/model.py`: two Conv2D + ReLU + MaxPool blocks (64×64 input), then fully connected layers (100 → 10). Trained on CIFAR-10 (resized to 64×64) for 10 epochs. **Test accuracy: 63.99%.**

```bash
uv run python train_cnn.py
```

## Project structure

```
app/                  API model code
  bigram_model.py       Assignment 1: bigram text generation
  embedding_model.py    Assignment 1: spaCy word embeddings
  classifier_model.py   Assignment 2: loads the CNN and classifies images
  gan_model.py          Assignment 3: loads the GAN generator and creates digit images
  image_generators.py   Assignment 4: loads the diffusion and energy models and generates images
helper_lib/           Reusable neural network code (Module 4, 6 and 8 class activities)
  model.py              Architectures (FCNN, CNN, EnhancedCNN, AssignmentCNN, GAN, EBM, Diffusion)
  data_loader.py        CIFAR-10 and MNIST data loaders
  trainer.py            Training loops (classifier, GAN, diffusion, energy model)
  evaluator.py          Loss and accuracy evaluation
  checkpoints.py        Save and load training checkpoints
  generator.py          Sampling for the GAN, diffusion and energy models (incl. Langevin dynamics)
  utils.py              Device selection, save and load model weights
models/               Trained model weights
sample_images/        Example CIFAR-10 test images for trying /classify
*_samples.png         Images generated by the trained GAN, diffusion and energy models
train_cnn.py          Trains the Assignment 2 CNN
train_gan.py          Trains the Assignment 3 GAN
train_diffusion.py    Trains the Assignment 4 diffusion model
train_ebm.py          Trains the Assignment 4 energy model
main.py               FastAPI app with all endpoints
Dockerfile            Container setup
```

## Run locally without Docker

```bash
uv sync
uv run fastapi dev main.py
```