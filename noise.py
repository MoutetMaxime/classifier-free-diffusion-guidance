import numpy as np


class NoiseConfig:
    def __init__(self, lambdaMin=-10, lambdaMax=10, steps=1000):
        self.lambdaMin = lambdaMin
        self.lambdaMax = lambdaMax
        self.steps = steps

        self.b_ = np.arctan(np.exp(-self.lambdaMax / 2))
        self.a_ = np.arctan(np.exp(-self.lambdaMin / 2)) - self.b_

    def sample_for_training(self, size=None):
        """
        Sample a specified number of 'lambda' values using a modified hyperbolic secant distribution to be used for training.

        The method generates uniformly distributed random numbers and transforms them into
        a specific distribution through logarithmic and trigonometric operations. 
        If no size is provided, the default number of samples is determined by `self.steps`.

        Args:
            size (int, optional): The number of samples to generate. If not provided, 
                                  `self.steps` (default 1000) is used.

        Returns:
            numpy.ndarray: An array of sampled 'lambda' values sorted in descending order.
        """
        if size:
            uniform_draw = np.random.uniform(0, 1, size=size)
        else:
            uniform_draw = np.random.uniform(0, 1, size=self.steps)

        uniform_draw = np.sort(uniform_draw)[::-1]
        return -2 * np.log(np.tan(self.a_ * uniform_draw + self.b_))

    def sample_for_sampling(self):
        """
        Sample a specified number of 'lambda' values using a modified hyperbolic secant distribution corresponding to SNR sequence for generation.

        The method generates uniformly spaced values of u and transforms them into
        a specific distribution through logarithmic and trigonometric operations. 
        The number of samples is determined by `self.steps`.

        Returns:
            numpy.ndarray: An array of 'lambda' values sorted in descending order.
        """

        u_array = np.linspace(0, 1, num=self.steps)
        lambda_array = -2 * np.log(np.tan(self.a_ * u_array + self.b_))
        # lambda_array = np.sort(lambda_array)[::-1]
        lambda_array = np.sort(lambda_array)
        return lambda_array