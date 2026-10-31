# sps_genai

FastAPI project for the Applied Generative AI course (Columbia SPS).
Arnav Goenka (ag5235)

- **Assignment 1:** bigram text generation and spaCy word embeddings (`/generate`, `/embedding`, `/similarity`)
- **Assignment 2:** CNN image classifier trained on CIFAR-10 (`/classify`)
- **Assignment 3:** GAN that generates handwritten digits, trained on MNIST (`/generate_digits`)

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs

## Test the GAN digit generator (`/generate_digits`) — Assignment 3

In `/docs`, open **GET /generate_digits**, click **Try it out**, and click **Execute**. The response is a PNG image with a grid of generated digits.

Parameters:
- `num_samples` (optional, 1–64, default 16): how many digits to generate
- `seed` (optional): the same seed always gives the same digits

Or open directly in a browser:

```
http://127.0.0.1:8000/generate_digits?num_samples=16&seed=42
```

Or from the terminal:

```bash
curl "http://127.0.0.1:8000/generate_digits?num_samples=16&seed=42" --output digits.png
```

Example output from the trained generator: `gan_samples.png`

## Test the image classifier (`/classify`) — Assignment 2

1. In `/docs`, open **POST /classify** and click **Try it out**.
2. Upload any image, for example one from `sample_images/` (e.g. `0_cat.png`).
3. Click **Execute**.

Or from the terminal:

```bash
curl -X POST http://127.0.0.1:8000/classify -F "file=@sample_images/0_cat.png"
```

The API returns the predicted CIFAR-10 class, the model's confidence, and the probability for every class.

## GAN architecture (Assignment 3)

Defined as `Generator`, `Discriminator` and `GAN` in `helper_lib/model.py`:

**Generator** (noise → image)

| Layer | Output shape |
|---|---|
| Input: noise vector | 100 |
| Fully connected, then reshape | 128 × 7 × 7 |
| ConvTranspose2D (128→64, kernel 4, stride 2, padding 1) + BatchNorm + ReLU | 64 × 14 × 14 |
| ConvTranspose2D (64→1, kernel 4, stride 2, padding 1) + Tanh | 1 × 28 × 28 |

**Discriminator** (image → real/fake probability)

| Layer | Output shape |
|---|---|
| Input: image | 1 × 28 × 28 |
| Conv2D (1→64, kernel 4, stride 2, padding 1) + LeakyReLU(0.2) | 64 × 14 × 14 |
| Conv2D (64→128, kernel 4, stride 2, padding 1) + BatchNorm + LeakyReLU(0.2) | 128 × 7 × 7 |
| Flatten + Linear + Sigmoid | 1 |

## GAN training (Assignment 3)

```bash
uv run python train_gan.py
```

- MNIST, 60,000 training images, normalized to [-1, 1] to match the generator's Tanh output
- Binary cross-entropy loss, with the non-saturating generator loss
- Two Adam optimizers (lr = 0.0002, betas = (0.5, 0.999)), batch size 128, 20 epochs
- Training alternates between updating the discriminator and the generator
- The trained generator is saved to `models/gan_generator.pth`, which the API loads

## CNN architecture (Assignment 2)

Defined as `AssignmentCNN` in `helper_lib/model.py`:

| Layer | Output shape |
|---|---|
| Input (RGB image) | 3 × 64 × 64 |
| Conv2D (16 filters, 3×3, stride 1, padding 1) + ReLU | 16 × 64 × 64 |
| MaxPool2D (2×2, stride 2) | 16 × 32 × 32 |
| Conv2D (32 filters, 3×3, stride 1, padding 1) + ReLU | 32 × 32 × 32 |
| MaxPool2D (2×2, stride 2) | 32 × 16 × 16 |
| Flatten | 8192 |
| Fully connected + ReLU | 100 |
| Fully connected (output) | 10 |

CIFAR-10 images are 32×32, so they are resized to 64×64