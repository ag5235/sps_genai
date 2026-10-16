import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader, get_train_val_loaders
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.utils import get_device, save_model

device = get_device()
print(f"Using device: {device}")

# Load data (CIFAR-10 resized to 64x64 to match the assignment architecture)
# 45,000 images for training, 5,000 held back for validation, 10,000 test images for the final score
train_loader, val_loader = get_train_val_loaders('data', batch_size=64, image_size=64)
test_loader = get_data_loader('data', batch_size=64, train=False, image_size=64)

# Initialize model and training components
model = get_model("AssignmentCNN")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train with checkpoint saving (validation set picks the best checkpoint)
trained_model = train_model(
    model, train_loader, val_loader, criterion, optimizer,
    device=device, epochs=10, checkpoint_dir='checkpoints'
)

# Evaluate final model on the test set (used only once, at the end)
avg_loss, accuracy = evaluate_model(trained_model, test_loader, criterion, device)
print(f"Test Accuracy: {accuracy:.2f}%")

# Save the final weights for the API
save_model(trained_model, 'models/cnn_cifar10.pth')
print("Model saved to models/cnn_cifar10.pth")