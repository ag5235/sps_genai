import torch


def evaluate_model(model, data_loader, criterion, device='cpu'):
    test_correct = 0
    test_total = 0
    running_loss = 0.0

    model.eval()
    with torch.no_grad():
        for test_images, test_labels in data_loader:
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            test_outputs = model(test_images)

            loss = criterion(test_outputs, test_labels)
            running_loss += loss.item()

            _, test_predicted = torch.max(test_outputs.data, 1)
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * test_correct / test_total
    return avg_loss, accuracy