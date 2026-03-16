"""
Unit tests for the Oil Shock Model.

Run with:
    python -m pytest tests/ -v
"""

import numpy as np
import pytest
from oil_shock_model import (
    logistic_pulse,
    gaussian_pulse,
    multi_logistic_discovery,
    exponential_kernel,
    run_osm,
    run_osm_convolution,
    OilShockModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_time(t_start, t_end, n=500):
    return np.linspace(t_start, t_end, n)


# ---------------------------------------------------------------------------
# Discovery functions
# ---------------------------------------------------------------------------

class TestLogisticPulse:
    def test_peak_at_t0(self):
        """Peak of logistic pulse should occur at t0."""
        t = np.linspace(1900, 2100, 2001)
        t0, k, scale = 1960.0, 0.5, 100.0
        D = logistic_pulse(t, t0, k, scale)
        assert abs(t[np.argmax(D)] - t0) < 1.0

    def test_non_negative(self):
        t = np.linspace(1900, 2100, 201)
        D = logistic_pulse(t, 1960.0, 0.3, 50.0)
        assert np.all(D >= 0)

    def test_integral_equals_scale(self):
        """Integral of pulse ≈ scale (total cumulative discovery)."""
        t = np.linspace(1800, 2200, 40001)
        scale = 200.0
        D = logistic_pulse(t, 1970.0, 0.2, scale)
        integral = np.trapezoid(D, t)
        assert abs(integral - scale) / scale < 0.01

    def test_scalar_input(self):
        """logistic_pulse should work with a scalar time."""
        val = logistic_pulse(1960.0, 1960.0, 0.3, 100.0)
        assert val > 0


class TestGaussianPulse:
    def test_peak_at_t0(self):
        t = np.linspace(1900, 2050, 1001)
        D = gaussian_pulse(t, 1950.0, 10.0, 80.0)
        assert abs(t[np.argmax(D)] - 1950.0) < 0.5

    def test_integral_equals_scale(self):
        t = np.linspace(1800, 2200, 40001)
        scale = 150.0
        D = gaussian_pulse(t, 1960.0, 15.0, scale)
        integral = np.trapezoid(D, t)
        assert abs(integral - scale) / scale < 0.01


class TestMultiLogistic:
    def test_sum_of_pulses(self):
        t = np.linspace(1900, 2050, 1001)
        params = [(1940.0, 0.3, 100.0), (2000.0, 0.5, 50.0)]
        D = multi_logistic_discovery(t, params)
        D1 = logistic_pulse(t, 1940.0, 0.3, 100.0)
        D2 = logistic_pulse(t, 2000.0, 0.5, 50.0)
        np.testing.assert_allclose(D, D1 + D2)


# ---------------------------------------------------------------------------
# Exponential kernel
# ---------------------------------------------------------------------------

class TestExponentialKernel:
    def test_integral_is_one(self):
        """Kernel should integrate to 1."""
        t = np.linspace(0, 500, 50001)
        tau = 20.0
        g = exponential_kernel(t, tau)
        integral = np.trapezoid(g, t)
        assert abs(integral - 1.0) < 0.001

    def test_zero_for_negative_t(self):
        t = np.array([-10.0, -1.0, 0.0, 1.0])
        g = exponential_kernel(t, 5.0)
        assert g[0] == 0.0
        assert g[1] == 0.0
        assert g[2] > 0.0  # at t=0: g(0) = 1/tau

    def test_mean_is_tau(self):
        """E[T] = integral of t*g(t)dt = tau."""
        t = np.linspace(0, 1000, 100001)
        tau = 15.0
        g = exponential_kernel(t, tau)
        mean = np.trapezoid(t * g, t)
        assert abs(mean - tau) / tau < 0.01


# ---------------------------------------------------------------------------
# ODE solver
# ---------------------------------------------------------------------------

class TestRunOSM:
    def _constant_discovery(self, rate):
        """Return a constant discovery function."""
        return lambda t: rate

    def test_returns_expected_keys(self):
        t = _uniform_time(1900, 2000, 100)
        D = self._constant_discovery(1.0)
        result = run_osm(t, D, 5.0, 3.0, 20.0)
        for key in ('t', 'discovery', 'fallow', 'construction',
                    'producing', 'production', 'cumulative'):
            assert key in result

    def test_non_negative_production(self):
        t = _uniform_time(1900, 2050, 500)
        params = [(1940.0, 0.2, 200.0)]
        D = lambda ti: multi_logistic_discovery(ti, params)
        result = run_osm(t, D, 5.0, 3.0, 20.0)
        assert np.all(result['production'] >= 0.0 - 1e-10)  # allow tiny floating-point error

    def test_mass_conservation(self):
        """Total discovery ≈ cumulative production (at t→∞)."""
        t = np.linspace(1900, 2300, 4000)
        scale = 200.0
        params = [(1950.0, 0.2, scale)]
        D = lambda ti: multi_logistic_discovery(ti, params)
        result = run_osm(t, D, 5.0, 3.0, 20.0)
        cum_prod = result['cumulative'][-1]
        cum_disc = np.trapezoid(result['discovery'], result['t'])
        assert abs(cum_prod - cum_disc) / cum_disc < 0.05

    def test_production_peak_after_discovery_peak(self):
        """Production peak should lag discovery peak due to delays."""
        t = _uniform_time(1900, 2100, 2000)
        t0_disc = 1950.0
        params = [(t0_disc, 0.3, 200.0)]
        D = lambda ti: multi_logistic_discovery(ti, params)
        result = run_osm(t, D, 5.0, 3.0, 20.0)
        disc_peak_year = t[np.argmax(result['discovery'])]
        prod_peak_year = t[np.argmax(result['production'])]
        assert prod_peak_year > disc_peak_year

    def test_zero_discovery_gives_zero_production(self):
        t = _uniform_time(1900, 2050, 200)
        D = lambda ti: 0.0
        result = run_osm(t, D, 5.0, 3.0, 20.0)
        assert np.allclose(result['production'], 0.0, atol=1e-12)

    def test_initial_conditions_respected(self):
        """Non-zero initial conditions should influence the output."""
        t = _uniform_time(0, 50, 100)
        D = lambda ti: 0.0
        # F0=10 → should produce non-zero output at t>0
        result = run_osm(t, D, 5.0, 3.0, 20.0, F0=10.0)
        assert np.max(result['production']) > 0.01


# ---------------------------------------------------------------------------
# Convolution vs ODE equivalence
# ---------------------------------------------------------------------------

class TestConvolutionEquivalence:
    def test_matches_ode_solution(self):
        """Convolution and ODE methods should give similar results."""
        t = np.linspace(1900, 2100, 2001)
        params = [(1960.0, 0.2, 200.0)]
        D = lambda ti: multi_logistic_discovery(ti, params)
        tau_f, tau_c, tau_x = 5.0, 3.0, 20.0

        result_ode = run_osm(t, D, tau_f, tau_c, tau_x)
        prod_conv = run_osm_convolution(t, D, tau_f, tau_c, tau_x)

        # Allow some tolerance due to numerical differences
        scale = np.max(result_ode['production'])
        norm_diff = np.max(np.abs(result_ode['production'] - prod_conv)) / scale
        assert norm_diff < 0.05  # within 5%


# ---------------------------------------------------------------------------
# OilShockModel class
# ---------------------------------------------------------------------------

class TestOilShockModel:
    def test_default_construction(self):
        m = OilShockModel()
        assert m.tau_f > 0
        assert m.tau_c > 0
        assert m.tau_x > 0
        assert len(m.disc_params) >= 1

    def test_run_returns_dict(self):
        m = OilShockModel(tau_f=5.0, tau_c=3.0, tau_x=20.0,
                          disc_params=[(1950.0, 0.3, 100.0)])
        t = np.linspace(1900, 2050, 300)
        result = m.run(t)
        assert 'production' in result
        assert len(result['production']) == len(t)

    def test_production_non_negative(self):
        m = OilShockModel(tau_f=5.0, tau_c=3.0, tau_x=20.0,
                          disc_params=[(1950.0, 0.3, 100.0)])
        t = np.linspace(1900, 2050, 300)
        result = m.run(t)
        assert np.all(result['production'] >= 0.0 - 1e-10)  # allow tiny floating-point error

    def test_discovery_rate(self):
        params = [(1960.0, 0.3, 100.0)]
        m = OilShockModel(disc_params=params)
        t = np.linspace(1900, 2050, 200)
        D = m.discovery_rate(t)
        assert len(D) == len(t)
        assert np.all(D >= 0)
