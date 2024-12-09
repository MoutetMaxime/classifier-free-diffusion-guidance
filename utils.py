import torch


def add_noise(x, t, noise_schedule):
    """
    Adds Gaussian noise to the input data x at step t according to the noise schedule.

    Args:
        x (torch.Tensor): Batch of input data, shape (batch_size, *dims).
        t (torch.Tensor): Indices of the noise step for each batch element, shape (batch_size,).
        noise_schedule (torch.Tensor): Noise schedule containing alpha_t coefficients, shape (num_steps,).

    Returns:
        noisy_x (torch.Tensor): The noisy version of x at step t, same shape as x.
        noise (torch.Tensor): Gaussian noise added to x.
    """
    # Generate Gaussian noise
    noise = torch.randn_like(x)

    # Retrieve alpha_t coefficients for the given steps and reshape for broadcasting
    alpha_t = noise_schedule[t].view(
        -1, *[1] * (x.ndim - 1)
    )  # Adjust dimensions to match x

    # Compute square roots for the noise schedule terms
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

    # Compute the noisy version of x
    noisy_x = sqrt_alpha_t * x + sqrt_one_minus_alpha_t * noise

    return noisy_x, noise
