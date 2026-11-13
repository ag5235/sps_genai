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




# ---------------------------------------------------------------
# Assignment 4: Diffusion and Energy-Based model training (Module 8)
# ---------------------------------------------------------------
import os
import random
import numpy as np
from .generator import langevin_sampling


def train_diffusion(model, data_loader, criterion, optimizer, device='cpu', epochs=10, val_loader=None, checkpoint_dir='checkpoints'):
    # Training loop from Module 8 Practical 2 (train_diffusion)
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        loader_with_progress = tqdm(data_loader, ncols=120, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, _ in loader_with_progress:
            images = images.to(device)
            loss = model.train_step(images, optimizer, criterion)
            train_losses.append(loss)
            loader_with_progress.set_postfix(loss=f'{loss:.4f}')
        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation loss with the EMA network
        message = f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}"
        if val_loader is not None:
            model.eval()
            val_losses = [model.test_step(images.to(device), criterion) for images, _ in val_loader]
            message += f" | Val Loss={sum(val_losses) / len(val_losses):.4f}"
        print(message)

        # Save a checkpoint every epoch
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'diffusion_epoch_{epoch+1:03d}.pth'))

    print("Finished Training")
    return model


class Buffer:
    # Replay buffer of fake images from Module 8 Practical 1
    def __init__(self, model, device, num_channels=3, image_size=32, batch_size=128):
        self.model = model
        self.device = device
        self.shape = (num_channels, image_size, image_size)
        self.batch_size = batch_size
        # Start with random images in [-1, 1]
        self.examples = [torch.rand((1, *self.shape), device=device) * 2 - 1 for _ in range(batch_size)]

    def sample_new_exmps(self, steps, step_size, noise):
        # About 5% brand-new random images, the rest continued from the buffer
        n_new = np.random.binomial(self.batch_size, 0.05)
        new_rand_imgs = torch.rand((n_new, *self.shape), device=self.device) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=self.batch_size - n_new), dim=0)
        inp_imgs = torch.cat([new_rand_imgs, old_imgs], dim=0)

        # Run Langevin dynamics to lower their energy
        new_imgs = langevin_sampling(self.model, inp_imgs, steps, step_size, noise)

        # Add the new images to the front of the buffer (keep at most 8192)
        self.examples = list(torch.split(new_imgs, 1, dim=0)) + self.examples
        self.examples = self.examples[:8192]
        return new_imgs


def train_ebm(model, data_loader, optimizer, device='cpu', epochs=10, alpha=0.1, steps=60, step_size=10, noise=0.005):
    # Training loop from Module 8 Practical 1 (EBM.train_step)
    model = model.to(device)
    buffer = Buffer(model, device)

    for epoch in range(epochs):
        totals = {"loss": 0.0, "cdiv": 0.0, "real": 0.0, "fake": 0.0}
        loader_with_progress = tqdm(data_loader, ncols=120, desc=f'Epoch {epoch+1}/{epochs}')
        for real_imgs, _ in loader_with_progress:
            real_imgs = real_imgs.to(device)
            real_imgs = (real_imgs + torch.randn_like(real_imgs) * noise).clamp(-1.0, 1.0)

            # Negative phase: fake images from Langevin sampling
            fake_imgs = buffer.sample_new_exmps(steps=steps, step_size=step_size, noise=noise)

            # Energy of real and fake images in one pass
            inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0).detach()
            out_scores = model(inp_imgs)
            real_out, fake_out = torch.split(out_scores, [real_imgs.size(0), fake_imgs.size(0)], dim=0)

            # Contrastive divergence: lower energy on real, raise it on fake (+ regularization)
            cdiv_loss = real_out.mean() - fake_out.mean()
            reg_loss = alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
            loss = cdiv_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            totals["loss"] += loss.item()
            totals["cdiv"] += cdiv_loss.item()
            totals["real"] += real_out.mean().item()
            totals["fake"] += fake_out.mean().item()
            loader_with_progress.set_postfix(cdiv=f'{cdiv_loss.item():.4f}')

        n = len(data_loader)
        print(f"Epoch {epoch+1}: " + ", ".join(f"{k}={v / n:.4f}" for k, v in totals.items()))

    print("Finished Training")
    return model