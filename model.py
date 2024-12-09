import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, input_dim, condition_dim=None):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_dim, kernel_size=3, padding=1),
        )
        self.condition_layer = None
        if condition_dim:
            self.condition_layer = nn.Linear(condition_dim, 128)

    def forward(self, x, condition=None):
        x = self.encoder(x)
        if self.condition_layer and condition is not None:
            condition = self.condition_layer(condition).view(-1, 128, 1, 1)
            x = x + condition
        x = self.decoder(x)
        return x
