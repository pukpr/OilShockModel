"""
Oil Shock Model (OSM) - Python Implementation

Based on the mathematical framework from:
  Mathematical GeoEnergy: Discovery, Depletion, and Renewal
  Paul Pukite, Dennis Coyne, Dan Challou (Wiley, 2019)

The Oil Shock Model is a compartmental Markov-chain model that simulates
oil production as oil moves through a series of stages:

    Discovery → Fallow → Construction → Production → Extraction

Each transition is modeled as an exponential (memoryless) delay, which
converts the convolution integral into a system of linear ODEs:

    dF/dt = D(t) - F(t) / tau_f
    dC/dt = F(t) / tau_f - C(t) / tau_c
    dP/dt = C(t) / tau_c - P(t) / tau_x
    Production rate = P(t) / tau_x

where:
    D(t)  - discovery input rate (Gb/year)
    F(t)  - reserves in fallow stage (Gb)
    C(t)  - reserves under construction/development (Gb)
    P(t)  - producing reserves (Gb)
    tau_f - mean fallow delay (years)
    tau_c - mean construction delay (years)
    tau_x - mean extraction/depletion time (years)

The discovery function is modeled as a sum of logistic growth pulses.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize, differential_evolution
from scipy.signal import convolve


# ---------------------------------------------------------------------------
# Discovery input functions
# ---------------------------------------------------------------------------

def logistic_pulse(t, t0, k, scale):
    """
    First derivative of the logistic function — a bell-shaped pulse.

    D(t) = scale * k * exp(-k*(t-t0)) / (1 + exp(-k*(t-t0)))^2

    Parameters
    ----------
    t : array-like
        Time vector (years).
    t0 : float
        Inflection point (peak of pulse, years).
    k : float
        Growth rate (1/years).  Larger k → narrower pulse.
    scale : float
        Total cumulative discovery (Gb).

    Returns
    -------
    ndarray
        Discovery rate (Gb/year).
    """
    t = np.asarray(t, dtype=float)
    z = k * (t - t0)
    # Numerically stable formulation: clip exponent to avoid overflow
    ez = np.exp(-np.clip(z, -500, 500))
    sig = ez / (1.0 + ez) ** 2
    return scale * k * sig


def gaussian_pulse(t, t0, sigma, scale):
    """
    Gaussian-shaped discovery pulse.

    Parameters
    ----------
    t : array-like
        Time vector (years).
    t0 : float
        Peak year.
    sigma : float
        Standard deviation (years).
    scale : float
        Total cumulative discovery (Gb).

    Returns
    -------
    ndarray
        Discovery rate (Gb/year).
    """
    t = np.asarray(t, dtype=float)
    return scale / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def multi_logistic_discovery(t, params):
    """
    Discovery rate as a sum of logistic pulses.

    Parameters
    ----------
    t : array-like
        Time vector (years).
    params : list of (t0, k, scale) tuples
        Parameters for each logistic pulse.

    Returns
    -------
    ndarray
        Total discovery rate (Gb/year).
    """
    t = np.asarray(t, dtype=float)
    D = np.zeros_like(t)
    for t0, k, scale in params:
        D += logistic_pulse(t, t0, k, scale)
    return D


# ---------------------------------------------------------------------------
# Core ODE system
# ---------------------------------------------------------------------------

def _osm_odes(state, t, D_func, tau_f, tau_c, tau_x):
    """
    Right-hand side of the Oil Shock Model ODEs.

    State vector: [F, C, P]
        F - fallow reserves (Gb)
        C - construction/development reserves (Gb)
        P - producing reserves (Gb)

    Returns dState/dt.
    """
    F, C, P = state
    D = D_func(t)

    dF_dt = D - F / tau_f
    dC_dt = F / tau_f - C / tau_c
    dP_dt = C / tau_c - P / tau_x

    return [dF_dt, dC_dt, dP_dt]


def run_osm(t, discovery_func, tau_f, tau_c, tau_x,
            F0=0.0, C0=0.0, P0=0.0):
    """
    Integrate the Oil Shock Model ODEs and return production rate.

    Parameters
    ----------
    t : array-like
        Time vector (years), must be monotonically increasing.
    discovery_func : callable
        Function D(t) → discovery rate (Gb/year).
    tau_f : float
        Mean fallow delay (years).
    tau_c : float
        Mean construction/development delay (years).
    tau_x : float
        Mean extraction/depletion time (years).
    F0, C0, P0 : float, optional
        Initial conditions for each compartment (Gb).

    Returns
    -------
    dict with keys:
        't'          : time array
        'discovery'  : D(t) - discovery rate (Gb/year)
        'fallow'     : F(t) - fallow reserves (Gb)
        'construction' : C(t) - construction reserves (Gb)
        'producing'  : P(t) - producing reserves (Gb)
        'production' : production rate = P(t)/tau_x (Gb/year)
        'cumulative' : cumulative production (Gb)
    """
    t = np.asarray(t, dtype=float)
    y0 = [F0, C0, P0]

    sol = odeint(_osm_odes, y0, t,
                 args=(discovery_func, tau_f, tau_c, tau_x),
                 mxstep=5000, rtol=1e-8, atol=1e-10)

    F = sol[:, 0]
    C = sol[:, 1]
    P = sol[:, 2]

    D = np.array([discovery_func(ti) for ti in t])
    production = P / tau_x
    # Cumulative production via trapezoidal integration (works for any spacing)
    if len(t) > 1:
        dt = np.diff(t)
        cumulative = np.concatenate(
            ([0.0], np.cumsum(0.5 * (production[:-1] + production[1:]) * dt))
        )
    else:
        cumulative = np.zeros_like(production)

    return {
        't': t,
        'discovery': D,
        'fallow': F,
        'construction': C,
        'producing': P,
        'production': production,
        'cumulative': cumulative,
    }


# ---------------------------------------------------------------------------
# Convolution-based formulation (equivalent to ODE approach for verification)
# ---------------------------------------------------------------------------

def exponential_kernel(t, tau):
    """
    Exponential impulse-response kernel: g(t) = (1/tau) * exp(-t/tau), t >= 0.

    Parameters
    ----------
    t : array-like
        Time lags (years), must be >= 0.
    tau : float
        Mean delay (years).

    Returns
    -------
    ndarray
    """
    t = np.asarray(t, dtype=float)
    return np.where(t >= 0, (1.0 / tau) * np.exp(-t / tau), 0.0)


def run_osm_convolution(t, discovery_func, tau_f, tau_c, tau_x):
    """
    Compute OSM production via triple convolution of discovery with
    exponential kernels.

    This is mathematically equivalent to ``run_osm`` for zero initial
    conditions and is useful for verification.

    Parameters
    ----------
    t : array-like
        Uniformly-spaced time vector (years).
    discovery_func : callable
        D(t) → discovery rate (Gb/year).
    tau_f, tau_c, tau_x : float
        Stage delay times (years).

    Returns
    -------
    ndarray
        Production rate (Gb/year) on the same time grid as ``t``.
    """
    t = np.asarray(t, dtype=float)
    dt = t[1] - t[0]
    n = len(t)

    # Kernel evaluated over twice the data length to avoid wrap-around
    t_kern = np.arange(2 * n) * dt
    g_f = exponential_kernel(t_kern, tau_f)
    g_c = exponential_kernel(t_kern, tau_c)
    g_x = exponential_kernel(t_kern, tau_x)

    D = np.array([discovery_func(ti) for ti in t])

    # Triple convolution (each normalised by dt so that sum*dt ≈ 1)
    h = convolve(g_f, g_c)[:2 * n] * dt
    h = convolve(h, g_x)[:2 * n] * dt
    production = convolve(D, h[:n])[:n] * dt

    return production


# ---------------------------------------------------------------------------
# Parameter fitting
# ---------------------------------------------------------------------------

class OilShockModel:
    """
    Oil Shock Model with parameter estimation.

    Attributes
    ----------
    tau_f : float  - fallow delay (years)
    tau_c : float  - construction delay (years)
    tau_x : float  - extraction time (years)
    disc_params : list of (t0, k, scale) - logistic pulse parameters
    """

    def __init__(self, tau_f=5.0, tau_c=3.0, tau_x=20.0, disc_params=None):
        self.tau_f = tau_f
        self.tau_c = tau_c
        self.tau_x = tau_x
        self.disc_params = disc_params or [(1960.0, 0.3, 200.0)]

    def discovery_rate(self, t):
        """Return discovery rate D(t) in Gb/year."""
        return multi_logistic_discovery(t, self.disc_params)

    def run(self, t):
        """Run the model and return result dict (see ``run_osm``)."""
        return run_osm(t,
                       self.discovery_rate,
                       self.tau_f,
                       self.tau_c,
                       self.tau_x)

    # ------------------------------------------------------------------
    # Fitting helpers
    # ------------------------------------------------------------------

    def fit(self, t_obs, prod_obs, bounds=None, method='differential_evolution',
            n_pulses=1, **kwargs):
        """
        Fit model parameters to observed production data.

        Parameters
        ----------
        t_obs : array-like
            Observed time points (years).
        prod_obs : array-like
            Observed production rates (Gb/year).
        bounds : list of (min, max) pairs, optional
            Parameter bounds.  If None, sensible defaults are used.
        method : str
            Optimisation method: 'differential_evolution' or 'minimize'.
        n_pulses : int
            Number of logistic discovery pulses to fit.

        Returns
        -------
        scipy.optimize.OptimizeResult
        """
        t_obs = np.asarray(t_obs, dtype=float)
        prod_obs = np.asarray(prod_obs, dtype=float)

        # Parameter layout:
        # [tau_f, tau_c, tau_x, t0_1, k_1, scale_1, ..., t0_N, k_N, scale_N]
        n_params = 3 + 3 * n_pulses

        if bounds is None:
            t_min, t_max = t_obs[0] - 50, t_obs[-1]
            bounds = (
                [(1.0, 30.0),   # tau_f
                 (0.5, 15.0),   # tau_c
                 (5.0, 100.0)]  # tau_x
                + [(t_min, t_max),  # t0
                   (0.01, 2.0),     # k
                   (0.1, 5000.0)]   # scale (Gb)
                * n_pulses
            )

        def _pack(params):
            tau_f, tau_c, tau_x = params[0], params[1], params[2]
            dp = []
            for i in range(n_pulses):
                base = 3 + 3 * i
                dp.append((params[base], params[base + 1], params[base + 2]))
            return tau_f, tau_c, tau_x, dp

        def _residuals(params):
            try:
                tau_f, tau_c, tau_x, dp = _pack(params)
                if tau_f <= 0 or tau_c <= 0 or tau_x <= 0:
                    return 1e20
                if any(k <= 0 or s <= 0 for _, k, s in dp):
                    return 1e20

                def D(ti):
                    return multi_logistic_discovery(ti, dp)

                result = run_osm(t_obs, D, tau_f, tau_c, tau_x)
                prod_model = result['production']
                # Normalised RMSE
                scale = np.max(prod_obs) if np.max(prod_obs) > 0 else 1.0
                return np.sqrt(np.mean(((prod_model - prod_obs) / scale) ** 2))
            except Exception:
                return 1e20

        if method == 'differential_evolution':
            result = differential_evolution(
                _residuals, bounds,
                seed=42, tol=1e-6, maxiter=2000,
                popsize=15, mutation=(0.5, 1.5), recombination=0.7,
                **kwargs)
        else:
            x0 = [0.5 * (lo + hi) for lo, hi in bounds]
            result = minimize(_residuals, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 5000, 'ftol': 1e-12}, **kwargs)

        if result.success or result.fun < 1.0:
            tau_f, tau_c, tau_x, dp = _pack(result.x)
            self.tau_f = tau_f
            self.tau_c = tau_c
            self.tau_x = tau_x
            self.disc_params = dp

        return result
