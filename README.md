# onhawkes

A lightweight Python library for real-time (online) intensity estimation and parameter fitting of univariate Hawkes processes with exponential kernels.

## Key Features

* **$O(1)$ Online Update**: Efficiently update the process intensity as events arrive without recalculating the entire history.
* **MLE Support**: Includes both Batch MLE (using L-BFGS-B) and an experimental Online Gradient Descent estimator.
* **Minimal Dependencies**: Built on top of `numpy` and `scipy`.

## Mathematical Background

The intensity function $\lambda(t)$ is defined as:

$$\lambda(t) = \mu + \sum_{t_i < t} \alpha e^{-\beta (t - t_i)}$$

Where:
* $\mu$: Background intensity (baseline rate).
* $\alpha$: Jump size (excitement) added by each event.
* $\beta$: Decay rate (how fast the influence of past events fades).

## Installation

```bash
pip install git+[https://github.com/4423i/Hawkes-On.git](https://github.com/4423i/Hawkes-On.git)

```

## Usage

### 1. Online Intensity Tracking

If you already know your parameters and want to track the intensity in real-time:

```python
from onhawkes import OnlineHawkesEstimator

# Initialize parameters
est = OnlineHawkesEstimator(mu=0.1, alpha=0.5, beta=3.0)

# Update as events arrive
event_times = [1.0, 1.2, 1.3, 2.5, 2.6]
for t in event_times:
    intensity = est.update(t)
    print(f"Time: {t:.1f} | Intensity: {intensity:.4f}")

```

### 2. Batch Parameter Fitting

To estimate parameters  from a set of historical event times:

```python
from onhawkes.estimator import HawkesExp

# Historical event data (timestamps)
data = [1.0, 1.2, 1.3, 2.5, 2.6, 3.0, 3.1]
model = HawkesExp(data)

# Initial guess [mu, alpha, beta]
x0 = [0.1, 0.1, 1.0]
params, log_likelihood = model.fit(x0)

print(f"Estimated Parameters: mu={params[0]:.3f}, alpha={params[1]:.3f}, beta={params[2]:.3f}")

```

## Development

### Requirements

* Python 3.12+
* NumPy
* SciPy

### Stability Note

For the process to remain stationary (not explode to infinity), ensure that the branching ratio $\alpha / \beta < 1$.