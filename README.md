# Hawkes-On

Online Hawkes intensity updates without repeatedly scanning the full event history.

Hawkes-On aims to bridge the gap between offline Hawkes-process analysis and integration into real-time event systems. The current implementation is a small Python package, `onhawkes`; the longer-term goal is a language-independent computational core, not a Python-only library.

**Status:** early-stage implementation (package version `0.1.0`). Fixed-parameter online intensity tracking is the core capability. Parameter-fitting routines are experimental and have known mathematical limitations; this is not yet a validated production inference library. See the [implementation review](docs/STATUS.md) and [roadmap](ROADMAP.md).

## Why Hawkes-On?

A Hawkes process models events that can temporarily increase the rate of subsequent events. Potential applications include financial event streams, anomaly detection, and event analysis. These are motivating use cases, not integrations currently provided by this repository.

Directly evaluating intensity from all past events makes each evaluation more expensive as history grows. For a single exponential kernel, the influence of past events can instead be summarized by a decaying state. Hawkes-On uses that recurrence to update intensity when an event arrives, without retaining the full history.

This mathematical property is not unique to Hawkes-On. The project's focus is to make it a small, understandable building block for online systems, and to develop the validation and interfaces needed to use it reliably across languages. The current repository does not yet provide streaming connectors, latency benchmarks, or a Rust implementation.

## What works today

| Component | Import | Current behavior |
| --- | --- | --- |
| `OnlineHawkesEstimator` | `from onhawkes import OnlineHawkesEstimator` | Tracks post-event intensity with fixed `mu`, `alpha`, and `beta`; constant work and state per event, independent of history length. |
| `OnlineHawkesMLE` | `from onhawkes.estimator import OnlineHawkesMLE` | Experimental per-event updates of `mu` and `alpha` with a fixed learning rate and fixed `beta`. Not a validated online MLE. |
| `HawkesExp` | `from onhawkes.estimator import HawkesExp` | Direct history-based intensity evaluation and experimental batch fitting with SciPy L-BFGS-B. The fitting objective has a known event-time convention issue. |

Only `OnlineHawkesEstimator` is re-exported at the package root. The complexity statement above follows from its recurrence; it is not a measured throughput or latency claim. Direct evaluation with `HawkesExp.intensity` scans the stored history, and batch fitting is not constant-time.

## Mathematical model and event-time convention

For a univariate process with a single exponential kernel, the conditional intensity immediately before an event is

$$
\lambda(t^-) = \mu + \sum_{t_i < t} \alpha e^{-\beta(t-t_i)}.
$$

Here `mu` is the baseline rate, `alpha` is the intensity jump per event, and `beta` is the decay rate. Use finite parameters with $\mu > 0$, $\alpha \geq 0$, and $\beta > 0$, and finite, strictly increasing event times expressed in one consistent time unit. Then `mu`, `alpha`, and `beta` have inverse-time units. Input validation is currently the caller's responsibility.

`OnlineHawkesEstimator.update(t)` consumes an event at `t` and returns the intensity **after adding that event's jump**, not the pre-event intensity in the equation above. If $I_{i-1}$ is the previous returned value, the recurrence is

$$
I_i^- = \mu + (I_{i-1} - \mu)e^{-\beta(t_i-t_{i-1})},
\qquad I_i = I_i^- + \alpha.
$$

The first call returns $\mu + \alpha$, assuming no excitation from events before the supplied stream. Only the last event time and the current intensity are retained in addition to the fixed parameters. Each call has $O(1)$ work and $O(1)$ state relative to event count.

For this nonnegative kernel, the usual subcritical stationary regime requires the branching ratio $\alpha/\beta < 1$. This is a stationarity condition, not an input check enforced by the implementation, nor a statement that every nonstationary parameter choice causes finite-time explosion. Change neither parameters nor timestamp units partway through a tracker; construct a new instance if you need to restart with different parameters.

## Installation

Requires **Python 3.12+**, NumPy, and SciPy. Both dependencies are installed with the package; SciPy is currently imported even when using only the online tracker.

Install from GitHub:

```bash
python -m pip install "git+https://github.com/4423i/Hawkes-On.git"
```

Or install a local checkout:

```bash
git clone https://github.com/4423i/Hawkes-On.git
cd Hawkes-On
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
```

These instructions use the repository source; they do not assume a published PyPI release.

## Minimal example: online intensity tracking

```python
from onhawkes import OnlineHawkesEstimator

# Fixed parameters; alpha / beta < 1.
estimator = OnlineHawkesEstimator(mu=0.1, alpha=0.5, beta=3.0)

for t in [1.0, 1.2, 1.3, 2.5, 2.6]:
    post_event_intensity = estimator.update(t)
    print(f"t={t:.1f}, post-event intensity={post_event_intensity:.4f}")
```

Output:

```text
t=1.0, post-event intensity=0.6000
t=1.2, post-event intensity=0.8744
t=1.3, post-event intensity=1.1737
t=2.5, post-event intensity=0.6293
t=2.6, post-event intensity=0.9921
```

Every call represents an event. Calling `update` just to query intensity between events would incorrectly add another jump. There is no separate non-mutating query method on this tracker.

## Experimental fitting: limitations before use

The fitting classes are present for experimentation, but their names should not be taken as evidence of statistically validated estimators:

- `HawkesExp.intensity(t, ...)` uses only events strictly before `t`. In contrast, `HawkesExp.loglik` adds the current event **before** evaluating its log-intensity term. It therefore does not implement the standard pre-event Hawkes log-likelihood. `fit(x0)` minimizes this objective and returns `(parameters, objective_value)`; the value is a negative-log-likelihood-style objective, not a log-likelihood. Optimizer success and diagnostics are not returned.
- `HawkesExp` assumes an observation window starting at zero and sets its end to the final event. Use a nonempty, one-dimensional, sorted array of floating-point times; integer arrays cause its `zeros_like` intensity buffer to use integer storage. Observation windows with a tail after the final event cannot be specified.
- `OnlineHawkesMLE` updates a decayed count including the current event, computes `lam`, and then adjusts `mu` and `alpha`. Its gradients are not a validated full point-process likelihood gradient. It returns `(updated_mu, updated_alpha, lam)`, where `lam` was computed **before** those parameter updates; `beta` is never estimated.
- Neither fitting routine enforces $\alpha/\beta < 1$ or provides convergence/uncertainty guarantees.

For these reasons, the quick start uses only fixed-parameter tracking. Numerical correctness and fitting stability are the first [roadmap](ROADMAP.md#phase-1--reliability-and-mathematical-validation) priorities, ahead of a Rust port.

## Current scope and limitations

- Univariate, single-exponential, nonnegative excitation only; no multivariate or alternative-kernel implementation.
- No rejection of out-of-order, duplicate, nonfinite, or otherwise invalid inputs. Tied times are processed as successive jumps, without a defined simultaneous-event statistical model.
- No historical warm start, persistence API, ingestion/reordering layer, concurrency guarantees, or production monitoring.
- No benchmark suite or measured performance guarantees. The new regression tests cover core recurrence behavior, not complete numerical or statistical validation.
- Runtime dependencies are not version-pinned, and no CI workflow or supported-version matrix is configured yet.

## Development and future direction

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, runnable tests, and guidance on mathematical changes. The [implementation review](docs/STATUS.md) records current strengths and gaps.

The [ROADMAP.md](ROADMAP.md) separates planned work into reliability, a Python-independent Rust core, cross-language APIs, online estimation, and operational readiness. **Rust, Python bindings, advanced online inference, and streaming integrations are future work, not current features.**

Questions, bug reports, and focused contributions are welcome through [GitHub Issues](https://github.com/4423i/Hawkes-On/issues) and pull requests.

## License

[MIT](LICENSE).
