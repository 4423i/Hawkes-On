# Contributing to Hawkes-On

Thanks for helping make online Hawkes computation easier to understand and trust. Small, focused contributions are welcome, including bug reports, mathematical reviews, tests, and documentation fixes. This is an early-stage, individually maintained project; no elaborate proposal process is required.

Read the [README](README.md), [current implementation review](docs/STATUS.md), and [roadmap](ROADMAP.md) to distinguish existing behavior from planned capabilities. For large changes (especially Rust, bindings, inference algorithms, or API redesign), open an issue before implementation so we can agree on scope.

## Development setup

Use Python **3.12 or newer**:

```bash
git clone https://github.com/4423i/Hawkes-On.git
cd Hawkes-On
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
```

For a pull request, fork the repository and clone your fork instead. The editable install brings in NumPy and SciPy. There are currently no separate development extras, formatter/linter configurations, or CI workflows.

The repository is deliberately small:

- `onhawkes/estimator.py`: fixed-parameter tracker and experimental fitting classes.
- `onhawkes/__init__.py`: package-root exports (`OnlineHawkesEstimator` only).
- `tests/`: standard-library `unittest` regression tests.
- `pyproject.toml`: setuptools build and runtime dependency metadata.

## Run the tests

From the repository root, with the environment activated:

```bash
python -m unittest discover -s tests -v
```

The tests use `unittest` plus the package's runtime dependencies; no pytest installation is needed. They cover fixed-parameter intensity recurrence against a direct sum, initialization, long-gap decay, tied-time behavior, and selected experimental API behavior. They do **not** certify fitting correctness, numerical stability over all parameter regimes, or performance.

To check the package import:

```bash
python -c "from onhawkes import OnlineHawkesEstimator; print(OnlineHawkesEstimator(0.1, 0.5, 3.0).update(1.0))"
```

Expected output: `0.6`. Please also execute any documentation example you change. No coverage threshold or benchmark gate is currently configured; describe what you actually tested rather than claiming broad validation.

## Reporting issues

Use [GitHub Issues](https://github.com/4423i/Hawkes-On/issues). For a bug, include:

- A minimal runnable example, expected behavior, and actual output or traceback.
- Python, NumPy, SciPy, operating system, and package version or Git revision.
- Relevant event times, parameters, time units, and observation window (synthetic or anonymized data is fine).
- For numerical discrepancies, the reference formula or calculation and the tolerance used.

Check existing issues first when practical. Questions and documentation issues are welcome; do not post credentials, private streams, or sensitive data.

## Pull requests

1. Create a branch in your fork and make a focused change.
2. Add or update tests for changed behavior. For bug fixes, include a regression test when practical.
3. Run the test command above and relevant examples.
4. Explain the motivation, behavior/API changes, and validation performed in the PR description. Link a related issue if there is one; an issue is not mandatory for small fixes.
5. Update documentation when semantics, limitations, or setup change. Call out compatibility changes explicitly.

Avoid mixing unrelated cleanup with functional changes. Maintainer review and CI setup are still evolving; small, reviewable PRs are easier to maintain than broad rewrites.

## Code quality

Follow the surrounding Python style, use descriptive names, and keep dependency additions justified. Prefer clear numeric contracts and simple implementations over unmeasured optimizations. Preserve package exports and return conventions unless an intentional API change has been discussed. Document validation requirements and errors rather than silently interpreting invalid data.

Do not describe planned Rust support, bindings, integrations, or inference improvements as available features. Performance claims need reproducible measurements; algorithmic complexity and observed runtime are different things.

## Mathematical and numerical changes

A plausible-looking curve or a successful optimizer return is not enough to establish correctness. For mathematical changes:

- State the kernel parameterization and units, parameter domain, event ordering, observation interval, and initial-history assumptions.
- Specify whether each intensity is evaluated before or after an event. Standard point-process likelihood terms use pre-event intensity; the current experimental batch objective has a documented discrepancy.
- Derive changed objectives/recurrences and compare against an independent reference (for example, a direct history sum or analytic compensator). For gradients, add finite-difference checks with justified step sizes and tolerances.
- Cover limiting and invalid cases relevant to the change: zero excitation, near-critical branching ratios, small/large gaps, empty input, ties, unsorted/nonfinite times, and floating-point versus integer arrays.
- Use appropriate absolute/relative tolerances and explain them. Use fixed seeds for randomized tests. Never preserve a known mathematical defect solely to make an old test pass; document deliberate behavior corrections.
- Separate numerical agreement from statistical validation. Inference changes should eventually include synthetic-data parameter recovery, convergence diagnostics, and initialization sensitivity, not only intensity checks.

The [Phase 1 roadmap](ROADMAP.md#phase-1--reliability-and-mathematical-validation) lists the known priorities. A Rust implementation must be checked against a validated reference, not blindly reproduce the current experimental fitting behavior.

## License

Contributions are made under the repository's [MIT license](LICENSE). Keep contributions compatible with it and acknowledge any external code or data sources.
