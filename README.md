# onhawkes

A simple Python library for online intensity estimation of a univariate Hawkes process with an exponential kernel.

## Overview

A Hawkes process is a type of self-exciting point process where the arrival of an event increases the probability of future events. This library provides a simple class, `OnlineHawkesEstimator`, to track and update the process intensity in real-time as new events arrive.

The intensity `λ(t)` is calculated as:
`λ(t) = μ + α * Σ_{t_i < t} exp(-β * (t - t_i))`

## Installation

```bash
pip install git+pip install git+https://github.com/4423i/Hawkes-On.git

```

## Usage

Here is a basic example of how to use `OnlineHawkesEstimator`.

```python
from onhawkes import OnlineHawkesEstimator

# Initialize the estimator with Hawkes process parameters
# mu: background intensity
# alpha: jump size after an event
# beta: decay rate
est = OnlineHawkesEstimator(mu=0.1, alpha=0.5, beta=3.0)

# Simulate some event arrivals and update the intensity
event_times = [1.0, 1.2, 1.3, 2.5, 2.6]
for t in event_times:
    intensity = est.update(t)
    print(f"Intensity at time {t}: {intensity:.4f}")
```
