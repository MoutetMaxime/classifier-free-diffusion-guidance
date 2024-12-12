import numpy as np


class NoiseConfig:
    def __init__(self):
        self.lambdaMin = -10
        self.lambdaMax = 10
        self.steps = 1000

        self.b_ = np.arctan(np.exp(-self.lambdaMax / 2))
        self.a_ = np.arctan(np.exp(-self.lambdaMin / 2)) - self.b_

    def sample(self, size=None):
        if size:
            uniform_draw = np.random.uniform(0, 1, size=size)
        else:
            uniform_draw = np.random.uniform(0, 1, size=self.steps)

        uniform_draw = np.sort(uniform_draw)[::-1]
        return -2 * np.log(np.tan(self.a_ * uniform_draw + self.b_))
