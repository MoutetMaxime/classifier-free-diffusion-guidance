import numpy as np


def compute_alpha_lambda(lambda_):
    """
    Compute alpha from lambda.

    Args:
        lambda_: Log signal-to-noise ratio tensor.
    """
    return np.sqrt(1 / (1 + np.exp(-lambda_)))


def compute_sigma_lambda(lambda_):
    """
    Compute sigma from lambda.

    Args:
        lambda_: Log signal-to-noise ratio tensor.
    """
    return np.sqrt(1 - compute_alpha_lambda(lambda_) ** 2)


def compute_sigma_lambda_lambda_prime(lambda_, lambda_prime):
    """
    Compute sigma of lambda knowing lambda (Equation 2)

    Args:
        lambda_: Log signal-to-noise ratio
        lambda_prime: Log signal-to-noise ratio 
    """
    # assert lambda_ < lambda_prime
    return np.sqrt((1 - np.exp(lambda_ - lambda_prime))) * compute_sigma_lambda(
          lambda_
    )

def compute_sigma_lambda_prime_lambda(lambda_prime, lambda_):
    """
    Compute sigma of lambda' knowing lambda (Equation 3)

    Args:
        lambda_prime: Log signal-to-noise ratio
        lambda_: Log signal-to-noise ratio
    """
    assert lambda_ < lambda_prime, f"Assertion failed: lambda_ ({lambda_}) is not less than lambda_prime ({lambda_prime})"
    return np.sqrt((1 - np.exp(lambda_ - lambda_prime))) * compute_sigma_lambda(
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
        np.exp(lambda_ - lambda_prime) * (alpha_lambda_prime / alpha_lambda) * z_lambda
        + (1 - np.exp(lambda_ - lambda_prime)) * alpha_lambda_prime * x
    )
