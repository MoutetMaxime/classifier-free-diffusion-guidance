import time
import torch
import torch.nn as nn


from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import UNet
from utils import add_noise


def train(
    model,
    dataloader,
    optimizer,
    noise_schedule,
    n_classes,
    num_epochs=2,
    model_path="model.pth",
):
    model.train()
    for epoch in range(num_epochs):
        start = time.time()
        for batch in dataloader:
            x, y = batch  # x: image, y: label

            x = x.to(device)
            y = y.to(device)

            # Add noise
            t = torch.randint(0, len(noise_schedule), (x.size(0),)).to(device)
            noisy_x, noise = add_noise(x, t, noise_schedule)

            # Conditionnel
            condition = (
                torch.eye(n_classes)[y].to(device) if torch.rand(1) < 0.5 else None
            )
            output = model(noisy_x, condition)

            # Loss
            loss = nn.MSELoss()(output, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), model_path)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Time: {time.time() - start}"
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pipeline
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    n_classes = 10

    # Model and optimizer
    model = UNet(input_dim=1, condition_dim=n_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Noise schedule
    noise_schedule = torch.linspace(1.0, 0.1, steps=100).to(device)

    train(model, dataloader, optimizer, noise_schedule, n_classes)
