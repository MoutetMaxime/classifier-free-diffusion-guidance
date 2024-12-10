import torch


def alpha_lambda(lambda_):
    """
    Compute alpha from lambda.

    Args:
        lambda_: Log signal-to-noise ratio tensor.
    """
    return 1 / (1 + torch.exp(-lambda_))


def sigma_lambda(lambda_):
    """
    Compute sigma from lambda.

    Args:
        lambda_: Log signal-to-noise ratio tensor.
    """
    return torch.sqrt(1 - alpha_lambda(lambda_) ** 2)


def sigma_lambda_lambda_prime(lambda_, lambda_prime):
    """
    Compute sigma of lambda' knowing lambda (Equation 2 and 3)

    Args:
        lambdas: Log signal-to-noise ratio tensor.
        lambda_prime: Log signal-to-noise ratio tensor
    """
    return torch.sqrt((1 - torch.exp(lambda_ - lambda_prime))) * sigma_lambda(
        lambda_prime
    )
