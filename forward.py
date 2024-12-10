import torch

from utils import alpha_lambda, sigma_lambda


def forward_process(x, lambdas):
    """
    Adding progressive noise to a batch of images.

    Args:
        x: Input tensor of shape (B, H, W, C) or (B, H, W).
        lambdas: Log signal-to-noise ratio tensor of shape (L) for L levels.

    Returns:
        z_lambdas: Noisy output tensor of shape (B, L, H, W, C) or (B, L, H, W).
    """
    L = lambdas.shape[0]  # Number of noise levels

    # Calculate alpha and sigma and reshape to broadcast with shape (B, H, W, C)
    alpha_lambdas = alpha_lambda(lambdas).view(1, L, 1, 1, -1)
    sigma_lambdas = sigma_lambda(lambdas).view(1, L, 1, 1, -1)

    # Check if input x has a channel dimension or not
    if x.ndimension() == 3:  # shape: (B, H, W)
        x = x.unsqueeze(-1)  # (B, H, W) -> (B, H, W, 1)
        has_channel = False
    elif x.ndimension() == 4:  # shape: (B, H, W, C)
        has_channel = True
    else:
        raise ValueError("Input tensor x should have 3 or 4 dimensions.")

    # Expand input x and generate noise
    x = x.unsqueeze(1)  # Shape: (B, 1, H, W, C) or (B, 1, H, W, 1)
    noise = torch.randn_like(x)  # Gaussian noise

    # Equation (1): Add progressive noise
    z_lambdas = alpha_lambdas * x + sigma_lambdas * noise

    # If the input did not have a channel dimension, remove the extra channel dimension at the end
    if not has_channel:
        z_lambdas = z_lambdas.squeeze(-1)  # (B, L, H, W, 1) -> (B, L, H, W)

    return z_lambdas
