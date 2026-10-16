import io
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from helper_lib.model import get_model
from helper_lib.utils import load_model

# Same class order as the class notebooks (CIFAR-10 label numbers 0-9)
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class CNNClassifier:
    def __init__(self, weights_path="models/cnn_cifar10.pth"):
        self.device = torch.device("cpu")
        self.model = get_model("AssignmentCNN")
        self.model = load_model(self.model, weights_path, self.device)
        # Same preprocessing as training: resize to 64x64, convert to tensor
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        predicted_index = int(torch.argmax(probabilities))
        return {
            "predicted_class": CLASSES[predicted_index],
            "confidence": round(float(probabilities[predicted_index]), 4),
            "probabilities": {
                CLASSES[i]: round(float(p), 4) for i, p in enumerate(probabilities)
            },
        }