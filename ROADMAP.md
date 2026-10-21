# Hawkes-On roadmap

Hawkes-On's goal is an efficient online Hawkes computation engine that can be embedded in real-time systems, not a Python-only interface. The current Python implementation is a starting point for defining and validating that engine.

**All phases below are planned work, not completed features or delivery commitments.** Scope and interfaces may change as validation results and contributor capacity become clearer. There are no committed dates. The [current implementation review](docs/STATUS.md) and [README](README.md) describe what exists today.

## Phase 1 — Reliability and mathematical validation

Establish a trustworthy numerical reference before translating or optimizing it.

- Expand the initial recurrence regression tests with randomized histories, limiting cases, and independent direct-sum references. Cover very small and large time gaps, underflow, near-critical branching ratios, and long streams.
- Specify pre-event versus post-event intensity throughout the API. Review and correct the batch objective's inclusion of the current event in its log-intensity term, with explicit tests against the standard point-process likelihood.
- Define the observation interval, initial excitation, and event ordering contract. Decide how to handle empty histories, ties, negative/unsorted/nonfinite times, invalid parameters, and integer timestamp arrays.
- Improve fitting stability: floating-point buffers, cancellation-resistant expressions where appropriate, optimizer diagnostics, convergence checks, initialization sensitivity, and optional stationarity constraints. Validate recovery on synthetic data before making inference claims.
- Audit `OnlineHawkesMLE` separately: derive its objective and gradients, quantify what the existing updates actually optimize, and decide whether to correct, rename, or deprecate this experimental interface. Do not make it the reference estimator merely because of its name.
- Add continuous integration, dependency/Python compatibility checks, and broader test coverage. Verify installation from a built distribution as well as editable source.

**Readiness criteria:** documented event/observation conventions; independently checked intensity and likelihood references; explicit failure behavior; tests for numerical boundaries; fitting diagnostics and evidence sufficient to describe estimator limitations accurately. This phase need not claim production readiness.

## Phase 2 — Python-independent Rust core

Start with fixed-parameter univariate exponential-kernel tracking, rather than porting every experimental fitting routine.

- Build a Rust library that can operate without Python or its runtime dependencies.
- Preserve $O(1)$ per-event work and state relative to history length using the exponential recurrence. Define numeric precision, ownership of state, reset/initialization behavior, and errors before stabilizing the API.
- Compare results to the validated Python reference across ordinary and boundary-case histories using shared fixtures and justified absolute/relative tolerances. Do not assume bitwise agreement across math libraries.
- Evaluate allocations, state layout, memory use, and update latency. Treat reduced latency and memory use as goals to measure, not guaranteed outcomes of choosing Rust.
- Keep inference outside the minimal tracking core unless a clear interface and validated algorithm justify including it.

**Readiness criteria:** a standalone Rust crate with tests, Python/Rust parity evidence, and reproducible measurements documenting environment and workload. Phase 1 correctness decisions must precede any claim that a Rust port is equivalent.

## Phase 3 — Cross-language access

Design interfaces around the independent core rather than binding the core to one host language.

- Provide a documented Rust-native API with explicit state and error handling.
- Evaluate PyO3 (or an alternative) for Python bindings. Review call overhead, per-event versus batch ingestion, GIL behavior, array ownership, packaging, wheels, and supported platforms.
- Decide whether to preserve the current Python import and return conventions through adapters or introduce a documented migration. Do not silently change pre/post-event semantics.
- Investigate demand for a stable C ABI or other language adapters. Specify ownership, lifetimes, error representation, and ABI/versioning policy before committing to additional languages.
- Test that bindings preserve the core's numerical and failure semantics.

**Readiness criteria:** usable Rust and Python APIs, installation tests for supported targets, cross-language contract tests, and a documented compatibility policy. Additional languages remain exploratory until there is a concrete use case.

## Phase 4 — Advanced online parameter estimation

Develop inference as a separately validated capability, not an automatic consequence of fast intensity tracking.

- Define a streaming inference objective, observation intervals, and treatment of periods with no events.
- Derive and check gradients or other update rules against numerical references, including any additional state required to estimate decay parameters.
- Investigate learning-rate schedules, positivity/stationarity constraints, regularization, and adaptation to nonstationary streams.
- Compare with validated batch inference on synthetic streams; evaluate estimation error, convergence, initialization sensitivity, and robustness near mathematical boundaries.
- State time and memory complexity for each algorithm. The fixed-parameter tracker's $O(1)$ property must not be assumed to apply to all inference methods.

**Readiness criteria:** documented assumptions and derivations, reproducible recovery experiments, numerical stability tests, and honest descriptions of bias, limitations, and computational costs. Research/prototyping can begin after Phase 1; integration into Rust or bindings should follow validation, not precede it.

## Phase 5 — Operational readiness

Turn a tested computational component into something easier to integrate and operate.

- Build a reproducible benchmark environment: event rates and gaps, stream lengths, parameter regimes, hardware/software versions, warm-up, and repeatability. Measure throughput, latency distributions, and memory; distinguish Python call overhead from core computation.
- Add practical examples for stream ingestion, timestamp normalization, out-of-order/duplicate-event policies, and recovery. Keep transport-specific connectors separate from mathematical state updates where feasible.
- Explore integrations with streaming systems based on actual user requirements; clarify backpressure, ordering, and restart semantics.
- Improve production reliability through state persistence/versioning decisions, stress and soak tests, release automation, and troubleshooting/observability guidance.
- Publish support boundaries and measured results rather than general claims of production safety or superiority to other projects.

**Readiness criteria:** reproducible benchmarks, representative integration examples, tested lifecycle/failure scenarios, and a release/support process maintainable by a small OSS project.

## Where support can help

Codex for OSS support would be most useful for expanding mathematical regression tests and CI, auditing fitting objectives and edge cases, developing and reviewing the independent Rust core, generating cross-language parity fixtures, and building reproducible integration and performance experiments. Each contribution should have a reviewable scope and explicit correctness evidence.

Large rewrites, bindings, new inference algorithms, and benchmark infrastructure are intentionally not part of the current documentation-focused preparation work. Proposals and focused issues are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
