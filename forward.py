import torch

from utils import (
    compute_alpha_lambda,
    compute_sigma_lambda,
    compute_sigma_lambda_lambda_prime,
)


def forward_process(x, lambda_):
    """
    Adding progressive noise to a batch of images.

    Args:
        x: Input tensor of shape (B, H, W, C) or (B, H, W).
        lambda_: Log signal-to-noise ratio

    Returns:
        z_lambda: Noisy output tensor of shape (B, H, W, C) or (B, H, W).
    """

    # Calculate alpha and sigma from lambda
    alpha_lambda = compute_alpha_lambda(lambda_)
    sigma_lambda = compute_sigma_lambda(lambda_)

    # Generate noise
    noise = torch.randn_like(x)  # Gaussian noise N(0, 1)

    # Equation (1): Add progressive noise N(alpha_lambda * x, sigma_lambda^2)
    z_lambdas = alpha_lambda * x + sigma_lambda * noise

    return z_lambdas


def forward_transition(z_lambda_prime, lambda_prime, lambda_):
    """
    Transition from one noise level to another.

    Args:
        z_lambda_prime: Noisy output tensor of shape (B, H, W, C) or (B, H, W).
        lambda_: Log signal-to-noise ratio
        lambda_prime: Log signal-to-noise ratio

    Returns:
        z_lambda: Noisy output tensor of shape (B, H, W, C) or (B, H, W).
    """
    assert lambda_prime > lambda_

    alpha_lambda = compute_alpha_lambda(lambda_)
    alpha_lambda_prime = compute_alpha_lambda(lambda_prime)
    sigma_lambda_lambda_prime = compute_sigma_lambda_lambda_prime(lambda_, lambda_prime)

    noise = torch.randn_like(z_lambda_prime)

    z_lambda = (
        alpha_lambda / alpha_lambda_prime
    ) * z_lambda_prime + sigma_lambda_lambda_prime * noise

    return z_lambda


if __name__ == "__main__":
    # Test the forward process
    x = torch.rand(2, 32, 32)  # (B, H, W, C)
    lambda_ = torch.tensor(1.0)
    z_lambda = forward_process(x, lambda_)
    z_lambda_next = forward_transition(z_lambda, lambda_, lambda_ - 10)

    print(f"Input shape: {x.shape}")
    print(f"Noisy output shape: {z_lambda.shape}")
    print(f"Noisy output next shape: {z_lambda_next.shape}")

    import matplotlib.pyplot as plt

    plt.imshow(x[0].detach().numpy())
    plt.show()
    plt.imshow(z_lambda[0].detach().numpy())
    plt.show()
    plt.imshow(z_lambda_next[0].detach().numpy())
    plt.show()
