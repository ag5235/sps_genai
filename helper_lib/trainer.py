import torch
from tqdm import tqdm
from .checkpoints import save_checkpoint
from .evaluator import evaluate_model


def train_model(model, train_loader, val_loader, criterion, optimizer, device='cpu', epochs=10, checkpoint_dir='checkpoints'):
    model = model.to(device)
    best_accuracy = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct, running_total = 0, 0

        model.train()
        train_loader_with_progress = tqdm(iterable=train_loader, ncols=120, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_number, (inputs, labels) in enumerate(train_loader_with_progress):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # log data for tracking
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()

            if batch_number % 100 == 99:
                train_loader_with_progress.set_postfix({'avg accuracy': f'{running_correct/running_total:.3f}', 'avg loss': f'{running_loss/(batch_number+1):.4f}'})

        # Training metrics for this epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * running_correct / running_total

        # Validation metrics for this epoch
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Train Acc={epoch_accuracy:.2f}% | Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.2f}%")

        # Save checkpoint every epoch
        checkpoint_path = save_checkpoint(model, optimizer, epoch + 1, val_loss, val_accuracy, checkpoint_dir)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(model, optimizer, epoch + 1, val_loss, val_accuracy, checkpoint_dir=f'{checkpoint_dir}/best')
            print(f"New best model saved! Val Accuracy: {val_accuracy:.2f}%")

    print("Finished Training")
    return model

def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    # optimizer is a pair: (generator optimizer, discriminator optimizer)
    opt_gen, opt_disc = optimizer
    generator = model.generator.to(device)
    discriminator = model.discriminator.to(device)
    z_dim = model.z_dim

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        loader_with_progress = tqdm(iterable=data_loader, ncols=120, desc=f'Epoch {epoch+1}/{epochs}')
        for real_images, _ in loader_with_progress:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # 1. Train the discriminator: push D(real) toward 1 and D(fake) toward 0
            noise = torch.randn(batch_size, z_dim, device=device)
            fake_images = generator(noise)
            loss_disc_real = criterion(discriminator(real_images), real_labels)
            loss_disc_fake = criterion(discriminator(fake_images.detach()), fake_labels)
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # 2. Train the generator: label fakes as real (non-saturating loss, slide 10)
            loss_gen = criterion(discriminator(fake_images), real_labels)
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            loader_with_progress.set_postfix({'D loss': f'{loss_disc.item():.4f}', 'G loss': f'{loss_gen.item():.4f}'})

        print(f"Epoch {epoch+1}: D loss={loss_disc.item():.4f}, G loss={loss_gen.item():.4f}")

    print("Finished Training")
    return model