import torch


def compute_alpha_lambda(lambda_):
    """
    Compute alpha from lambda.

    Args:
        lambda_: Log signal-to-noise ratio tensor.
    """
    return torch.sqrt(1 / (1 + torch.exp(-lambda_)))


def compute_sigma_lambda(lambda_):
    """
    Compute sigma from lambda.

    Args:
        lambda_: Log signal-to-noise ratio tensor.
    """
    return torch.sqrt(1 - compute_alpha_lambda(lambda_) ** 2)


def compute_sigma_lambda_prime_lambda(lambda_, lambda_prime):
    """
    Compute sigma of lambda' knowing lambda (Equation 2 and 3)

    Args:
        lambdas: Log signal-to-noise ratio tensor.
        lambda_prime: Log signal-to-noise ratio tensor
    """
    return torch.sqrt((1 - torch.exp(lambda_ - lambda_prime))) * compute_sigma_lambda(
        lambda_prime
    )


def compute_mu_lambda_prime_lambda(
    z_lambda,
    x,
    lambda_,
    lambda_prime,
):
    alpha_lambda = compute_alpha_lambda(lambda_)
    alpha_lambda_prime = compute_alpha_lambda(lambda_prime)
    return (
        torch.exp(lambda_ - lambda_prime)
        * (alpha_lambda_prime / alpha_lambda)
        * z_lambda
        + (1 - torch.exp(lambda_ - lambda_prime)) * alpha_lambda_prime * x
    )
