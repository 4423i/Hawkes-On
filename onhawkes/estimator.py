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
    """Projected interval-likelihood gradient updates with fixed beta.

    Observation starts at start_time with no prior excitation. Each update
    consumes one event and the preceding event-free interval. Parameters are
    held fixed when differentiating that interval's negative log-likelihood.
    This stochastic approximation does not solve the full-history batch MLE.
    No likelihood contribution after the last supplied event is included.
    """

    def __init__(self, mu=0.1, alpha=0.5, beta=1.0, lr=0.01, *, start_time=0.0):
        if not all(np.isfinite(x) for x in (mu, alpha, beta, lr, start_time)):
            raise ValueError("parameters and start_time must be finite")
        if mu <= 0 or alpha < 0 or beta <= 0 or lr <= 0:
            raise ValueError("require mu > 0, alpha >= 0, beta > 0, lr > 0")
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.last_t = start_time
        self.s = 0.0

    def update(self, t):
        """Return (updated_mu, updated_alpha, pre_event_intensity).

        Intensity uses parameters before this update. Event times must be
        finite and strictly greater than the previous observation time.
        """
        if not np.isfinite(t) or t <= self.last_t:
            raise ValueError("event time must be finite and strictly increasing")
        dt = t - self.last_t
        decay = np.exp(-self.beta * dt)
        integrated_s = self.s * (-np.expm1(-self.beta * dt) / self.beta)
        pre_s = self.s * decay
        lam = self.mu + self.alpha * pre_s

        g_mu = dt - 1 / lam
        g_alpha = integrated_s - pre_s / lam

        self.mu = max(1e-8, self.mu - self.lr * g_mu)
        self.alpha = max(0.0, self.alpha - self.lr * g_alpha)

        self.s = pre_s + 1.0
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
