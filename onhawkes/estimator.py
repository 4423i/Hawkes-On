import numpy as np
from numpy import exp, log
from scipy.optimize import minimize


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


class OnlineHawkesMLE:

    def __init__(self, mu=0.1, alpha=0.5, beta=1.0, lr=0.01):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.last_t = None
        self.s = 0.0

    def update(self, t):
        if self.last_t is not None:
            dt = t - self.last_t
            self.s *= np.exp(-self.beta * dt)
        self.s += 1.0

        lam = self.mu + self.alpha * self.s

        g_mu = -1 / lam + 1
        g_alpha = -self.s / lam + self.s

        self.mu = max(1e-8, self.mu - self.lr * g_mu)
        self.alpha = max(1e-8, self.alpha - self.lr * g_alpha)

        self.last_t = t
        return self.mu, self.alpha, lam


class HawkesExp:

    def __init__(self, t):
        self.t = np.asarray(t)
        self.T = float(self.t[-1])

    def intensity(self, t, mu, alpha, beta):
        idx = self.t < t
        return mu + np.sum(alpha * np.exp(-beta * (t - self.t[idx])))

    def loglik(self, params):
        mu, alpha, beta = params
        if mu <= 0 or alpha <= 0 or beta <= 0:
            return np.inf

        t = self.t

        lam = np.zeros_like(t)
        s = 0.0
        last_t = t[0]
        for i in range(len(t)):
            dt = t[i] - last_t
            s *= exp(-beta * dt)
            s += 1.0
            lam[i] = mu + alpha * s
            last_t = t[i]

        integral = mu * self.T
        integral += alpha / beta * np.sum(1 - exp(-beta * (self.T - t)))

        return -(np.sum(log(lam)) - integral)

    def fit(self, x0):
        res = minimize(self.loglik, x0, method="L-BFGS-B", bounds=[(1e-8, None)] * 3)
        return res.x, res.fun
