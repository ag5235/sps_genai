# sps_genai

FastAPI project for the Applied Generative AI course (Columbia SPS).
Arnav Goenka (ag5235)

- **Assignment 1:** bigram text generation and spaCy word embeddings (`/generate`, `/embedding`, `/similarity`)
- **Assignment 2:** CNN image classifier trained on CIFAR-10 (`/classify`)

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs

## Test the image classifier (`/classify`)

1. In `/docs`, open **POST /classify** and click **Try it out**.
2. Upload any image, for example one from `sample_images/` (e.g. `0_cat.png`).
3. Click **Execute**.

Or from the terminal:

```bash
curl -X POST http://127.0.0.1:8000/classify -F "file=@sample_images/0_cat.png"
```

Example response:

```json
{
  "predicted_class": "cat",
  "confidence": 0.5398,
  "probabilities": { "airplane": 0.0252, "cat": 0.5398, "dog": 0.1635, "...": "..." }
}
```

The API returns the predicted CIFAR-10 class (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), the model's confidence, and the probability for every class.

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

CIFAR-10 images are 32×32, so they are resized to 64×64 to match the required input size.

## Training

```bash
uv run python train_cnn.py
```

- 45,000 training images, 5,000 validation images (used to pick the best checkpoint), 10,000 test images (used once for the final score)
- Adam optimizer (lr = 0.001), cross-entropy loss, batch size 64, 10 epochs
- **Test accuracy: 63.99%**
- Final weights are saved to `models/cnn_cifar10.pth`, which the API loads

## Project structure

```
app/                  API model code
  bigram_model.py       Assignment 1: bigram text generation
  embedding_model.py    Assignment 1: spaCy word embeddings
  classifier_model.py   Assignment 2: loads the CNN and classifies images
helper_lib/           Reusable neural network code (Module 4 class activity)
  model.py              Model architectures (FCNN, CNN, EnhancedCNN, AssignmentCNN)
  data_loader.py        CIFAR-10 data loaders
  trainer.py            Training loop with checkpoints
  evaluator.py          Loss and accuracy evaluation
  checkpoints.py        Save and load training checkpoints
  utils.py              Device selection, save and load model weights
models/               Trained model weights
sample_images/        Example CIFAR-10 test images for trying /classify
train_cnn.py          Script that trains the Assignment 2 CNN
main.py               FastAPI app with all endpoints
Dockerfile            Container setup
```

## Run locally without Docker

```bash
uv sync
uv run fastapi dev main.py
```