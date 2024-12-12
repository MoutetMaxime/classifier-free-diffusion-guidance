import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from noise import NoiseConfig
from forward import forward_process
from utils import compute_alpha_lambda, compute_sigma_lambda


class UNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, condition_dim=None, puncond=0.2):
        super(UNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, input_dim, kernel_size=3, padding=1),
        )

        self.condition_layer = None
        if condition_dim:
            self.condition_layer = nn.Linear(condition_dim, hidden_dim)

    def forward(self, x, condition=None):
        x = self.encoder(x)
        if self.condition_layer and condition is not None:
            condition = self.condition_layer(condition).view(-1, self.hidden_dim, 1, 1)
            x = x + condition
        x = self.decoder(x)
        return x

    @staticmethod
    def loss_function(outputs, inputs):
        return F.mse_loss(outputs, inputs)

    def train(
        self,
        dataloader,
        optimizer,
        num_epochs=10,
        puncond=0.2,
        n_classes=10,
        device="cpu",
        verbose=True,
        model_path=None,
    ):
        """
        Train the model.

        Args:
            optimizer: Optimizer for training.
            criterion: Loss function.
            num_epochs: Number of epochs for training.
            device: Device to use for training.
            verbose: Whether to print training progress.
        """
        self.to(device)

        # Sample noise schedule
        diffusion_schedule = NoiseConfig().sample()
        best_loss = float("inf")
        for epoch in range(num_epochs):
            running_loss = 0.0
            start = time.time()
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                lambda_ = diffusion_schedule[np.random.randint(len(diffusion_schedule))]
                noisy_inputs = forward_process(inputs, lambda_)

                optimizer.zero_grad()

                condition = (
                    torch.eye(n_classes)[labels]
                    if torch.rand(1).item() < puncond
                    else None
                )
                noise_prediction = self(noisy_inputs, condition)

                # Equation (5)
                real_noise = (
                    noisy_inputs - compute_alpha_lambda(lambda_) * inputs
                ) / compute_sigma_lambda(lambda_)

                loss = self.loss_function(noise_prediction, real_noise)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                print(f"Loss: {loss.item()}")

            best_loss = min(best_loss, running_loss / len(dataloader))
            if verbose:
                print(
                    f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}, Time: {time.time() - start}"
                )

            if model_path and running_loss / len(dataloader) < best_loss:
                torch.save(self.state_dict(), model_path)

        if verbose:
            print("Finished Training")


if __name__ == "__main__":
    # Test the model
    model = UNet(1, 64)

    # Get MNIST data
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST

    # Normalize the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model.train(
        train_loader,
        optimizer,
        num_epochs=1,
        device="cpu",
        verbose=True,
        model_path="model.pth",
    )
