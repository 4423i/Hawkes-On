import numpy as np


class OnlineHawkesEstimator:
    def __init__(self, mu, alpha, beta):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.last_t = None
        self.intensity = mu

    def update(self, t):
        if self.last_t is not None:
            dt = t - self.last_t
            self.intensity = self.mu + (self.intensity - self.mu) * np.exp(
                -self.beta * dt
            )
        self.intensity += self.alpha
        self.last_t = t
        return self.intensity
