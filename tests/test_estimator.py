"""Regression checks, not a certification of the experimental fitting routines."""

import math
import unittest

import numpy as np

from onhawkes import OnlineHawkesEstimator
from onhawkes.estimator import HawkesExp, OnlineHawkesMLE


class OnlineIntensityTests(unittest.TestCase):
    def test_initial_state_and_first_event_are_post_event(self):
        estimator = OnlineHawkesEstimator(mu=0.1, alpha=0.5, beta=3.0)
        self.assertIsNone(estimator.last_t)
        self.assertEqual(estimator.intensity, 0.1)
        self.assertAlmostEqual(estimator.update(1.0), 0.6)
        self.assertEqual(estimator.last_t, 1.0)

    def test_recurrence_matches_direct_history_sum(self):
        mu, alpha, beta = 0.1, 0.5, 3.0
        times = [1.0, 1.2, 1.3, 2.5, 2.6]
        estimator = OnlineHawkesEstimator(mu, alpha, beta)
        for i, t in enumerate(times):
            expected = mu + math.fsum(
                alpha * math.exp(-beta * (t - past)) for past in times[: i + 1]
            )
            with self.subTest(t=t):
                self.assertTrue(
                    math.isclose(estimator.update(t), expected, rel_tol=1e-12, abs_tol=1e-12)
                )

    def test_zero_excitation_stays_at_baseline(self):
        estimator = OnlineHawkesEstimator(mu=0.1, alpha=0.0, beta=3.0)
        for t in [1.0, 1.2, 100.0]:
            self.assertEqual(estimator.update(t), 0.1)

    def test_long_gap_removes_previous_excitation_before_new_jump(self):
        estimator = OnlineHawkesEstimator(mu=0.1, alpha=0.5, beta=3.0)
        estimator.update(1.0)
        self.assertAlmostEqual(estimator.update(1000.0), 0.6)

    def test_tied_times_are_successive_jumps_in_current_api(self):
        # Characterizes implementation behavior, not a simultaneous-event model.
        estimator = OnlineHawkesEstimator(mu=0.1, alpha=0.5, beta=3.0)
        estimator.update(1.0)
        self.assertAlmostEqual(estimator.update(1.0), 1.1)


class ExperimentalApiTests(unittest.TestCase):
    def test_batch_intensity_excludes_event_at_query_time(self):
        times = np.array([1.0, 1.2, 1.3, 2.5], dtype=float)
        model = HawkesExp(times)
        self.assertAlmostEqual(model.intensity(1.0, 0.1, 0.5, 3.0), 0.1)
        for query in [1.2, 1.3, 2.0, 3.0]:
            expected = 0.1 + math.fsum(
                0.5 * math.exp(-3.0 * (query - t)) for t in times if t < query
            )
            with self.subTest(query=query):
                self.assertAlmostEqual(model.intensity(query, 0.1, 0.5, 3.0), expected)

    def test_online_parameter_return_uses_pre_update_parameters_for_lam(self):
        # A return-contract check only; it does not validate the gradient as MLE.
        model = OnlineHawkesMLE(mu=0.1, alpha=0.5, beta=1.0, lr=0.01)
        mu, alpha, lam = model.update(1.0)
        self.assertAlmostEqual(lam, 0.6)
        self.assertEqual((mu, alpha), (model.mu, model.alpha))
        self.assertNotAlmostEqual(lam, mu + alpha * model.s)
        self.assertEqual(model.beta, 1.0)


if __name__ == "__main__":
    unittest.main()
