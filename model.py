import torch
import time
import torch.nn as nn
import torch.nn.functional as F

from forward import forward_process
from utils import alpha_lambda, sigma_lambda


class UNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(UNet, self).__init__()
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

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @staticmethod
    def loss_function(outputs, inputs):
        return F.mse_loss(outputs, inputs)

    def train(
        self,
        dataloader,
        optimizer,
        diffusion_schedule,
        num_epochs=10,
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
        self.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            start = time.time()
            for data in dataloader:
                inputs = data
                inputs = inputs.to(device)

                lambda_ = diffusion_schedule[torch.randint(len(diffusion_schedule))]
                noisy_inputs = forward_process(inputs, lambda_)

                optimizer.zero_grad()

                noise_prediction = self(noisy_inputs)

                # Equation (5)
                real_noise = (
                    noisy_inputs - alpha_lambda(lambda_) * inputs
                ) / sigma_lambda(lambda_)

                loss = self.loss_function(noise_prediction, real_noise)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            if verbose:
                print(
                    f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}, Time: {time.time() - start}"
                )

            if model_path:
                torch.save(self.state_dict(), model_path)

        if verbose:
            print("Finished Training")
