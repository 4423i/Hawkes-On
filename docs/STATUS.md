# Current implementation review

This review describes the source in `onhawkes/estimator.py`, the package exports, and the packaging configuration during the documentation-focused OSS preparation. It is a code-level assessment, not a performance benchmark or proof of statistical correctness. Future work is tracked in [ROADMAP.md](../ROADMAP.md).

## Implemented capabilities

- `OnlineHawkesEstimator(mu, alpha, beta)` starts with baseline intensity and no prior excitation. `update(t)` decays the previous excess intensity, adds `alpha`, records `t`, and returns post-event intensity. It retains only parameters, `last_t`, and `intensity`. For fixed parameters and ordered events the recurrence is equivalent to summing exponential contributions including the current event.
- `OnlineHawkesMLE(mu=0.1, alpha=0.5, beta=1.0, lr=0.01)` retains a decayed event count `s`. Each update adds the current event, computes `lam`, and applies the implemented gradients `-1/lam + 1` and `-s/lam + s` to `mu` and `alpha`. Both are clipped below at `1e-8`; `beta` is fixed. The return tuple mixes updated parameters with the intensity computed before the parameter update.
- `HawkesExp(t)` stores `np.asarray(t)` and sets `T` to the final time. `intensity(t, mu, alpha, beta)` directly sums contributions from stored events strictly earlier than the query. `loglik(params)` calculates a negative-log-likelihood-style objective. `fit(x0)` calls SciPy's L-BFGS-B with positive bounds and returns only `(res.x, res.fun)`.
- `onhawkes/__init__.py` exports only `OnlineHawkesEstimator`. Experimental classes are available through `onhawkes.estimator`.
- `pyproject.toml` defines a setuptools package, `onhawkes` version `0.1.0`, requiring Python 3.12+, NumPy, and SciPy. Dependency versions are unrestricted. The repository includes an MIT license.

## Technical constraints and risks

### Intensity tracking

The single-exponential recurrence is the strongest and simplest part of the implementation: constant work/state per event relative to history length, with no external stream framework required. This is an algorithmic property, not a measured latency or throughput result.

The tracker does not validate parameters or times. Out-of-order events can amplify rather than decay past excitation; nonfinite inputs propagate into state. Tied times add successive jumps. There is no non-mutating between-event query, historical initialization, reset/persistence API, or synchronization mechanism. Public parameters can be mutated, but changing them midstream generally invalidates the meaning of the accumulated state.

### Batch fitting

The standard point-process log-likelihood over `[0, T]`, with no pre-window history, is

$$
\ell = \sum_i \log\lambda(t_i^-) - \mu T
- \frac{\alpha}{\beta}\sum_i\left(1-e^{-\beta(T-t_i)}\right).
$$

The implemented compensator has this form, but the loop adds `1.0` to the decayed count **before** evaluating each event's intensity. Its log terms thus include the current event, unlike both the formula above and `HawkesExp.intensity`. In particular, the first event uses `mu + alpha` instead of `mu`. Fitting results must not be presented as validated standard Hawkes maximum-likelihood estimates.

Additional issues:

- `np.zeros_like(t)` inherits the timestamp dtype, truncating computed intensities for integer arrays. Floating-point times avoid that specific problem but do not fix the likelihood convention.
- Empty input fails when accessing the last element. Array shape, finiteness, nonnegative observation times, and sortedness are not checked.
- The observation starts implicitly at zero and ends at the last event, with no way to specify an observation tail or prehistory.
- Positive bounds do not enforce stationarity. The objective rejects `alpha == 0`, even though zero excitation is a meaningful model boundary.
- Expressions such as `1 - exp(-x)` may lose precision for small `x`; no systematic numerical-boundary validation exists yet.
- `fit` discards optimizer success, message, and other diagnostics. Its second return value is the minimized negative objective, not a log-likelihood.

### Experimental online parameter updates

The class name `OnlineHawkesMLE` overstates what has been established. The update uses post-event intensity and does not implement a documented full observation-window likelihood gradient. There is no gradient validation, decay-parameter estimation, learning-rate schedule, stationarity constraint, or convergence/uncertainty assessment. Its constant-size state does not establish estimator correctness.

## Tests and distribution

The repository initially contained no tests, CI workflows, coverage configuration, benchmarks, examples directory, or developer guide. This preparation adds a small `unittest` regression suite focused on the existing recurrence and selected API semantics. It deliberately does not implement new inference algorithms or change runtime behavior.

There is still no broad edge-case suite, CI/support matrix, statistical validation, reproducible performance baseline, or release automation. The package metadata alone does not establish that a PyPI release is available. Installation instructions therefore use source installation rather than assuming one.

An editable install, the seven regression tests, the README example, and source/wheel builds were checked with Python 3.12. This is a single-environment check, not a supported-platform matrix. `MANIFEST.in` includes the new guides in source distributions so the README's local documentation links have targets there as well. The build emits a setuptools deprecation warning for the existing license classifier; migration to modern SPDX metadata remains a packaging follow-up.

## Documentation findings addressed here

- The original Git installation command contained Markdown link markup inside a shell command and was not runnable as written.
- The original mathematical description did not distinguish the tracker's post-event return from the standard pre-event conditional intensity.
- Batch fitting was described as MLE without disclosing the event-time issue, and an example named the minimized negative objective `log_likelihood`.
- The stationarity note conflated stationarity with finite-time explosion.
- There was no clear account of the project's systems-level purpose, package-root versus module imports, inference limitations, contribution process, or planned language-independent direction.

The updated [README](../README.md), [contribution guide](../CONTRIBUTING.md), and [roadmap](../ROADMAP.md) address these documentation gaps without changing the estimators. GitHub description and topics are suggestions for the maintainer; repository settings are not changed by this work.

## Recommended priority

1. Validate mathematical conventions and strengthen tests/CI before promising reliable inference.
2. Resolve input contracts, dtype handling, observation windows, and optimizer diagnostics through focused, tested changes.
3. Use the validated fixed-parameter recurrence as the first Rust-core scope; keep experimental inference out of parity requirements until independently validated.
4. Introduce cross-language interfaces, improved inference, and operational benchmarks in reviewable stages, as described in the roadmap.
