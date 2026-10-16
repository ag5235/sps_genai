import os
import torch


def get_device():
    # Same device check as the class notebooks: Mac GPU (mps), NVIDIA GPU (cuda), or CPU
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    return device


def save_model(model, path):
    # Save only the learned weights (used by the API)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device='cpu'):
    # Load the learned weights into a model and switch it to evaluation mode
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model