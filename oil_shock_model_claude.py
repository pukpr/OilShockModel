"""
Oil Shock Model (OSM) — convolution-based implementation.

Reference: Pukite, Coyne & Challou, "Mathematical GeoEnergy" (Wiley, 2019).

Core idea
---------
Oil production P(t) is the convolution of a discovery rate D(t) with a
composite transfer function h(t) that represents the multi-stage pipeline
from discovery to extraction:

    P(t) = D(t) * h_1(t) * h_2(t) * ... * h_n(t)

Each stage i is modelled as a first-order (exponential) process:

    h_i(t) = k_i * exp(-k_i * t),  t >= 0

so the composite h(t) is a hypo-exponential (Erlang if all k_i are equal).
"Shocks" enter through abrupt changes in the discovery / booking rate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class OSMParams:
    """All tuneable parameters for the Oil Shock Model.

    Stage rates
    -----------
    k_fallow      : rate constant for the fallow / booking stage  [1/yr]
    k_build       : rate constant for the build / development stage [1/yr]
    k_maturation  : rate constant for the maturation / production stage [1/yr]

    Extra stages can be added via `extra_stages` (list of additional k values).

    Discovery model
    ---------------
    Uses a sum of Gaussians so multiple discovery waves (e.g. Middle-East,
    North-Sea, deep-water) can be represented.

    Each entry in `discovery_pulses` is a dict with keys:
        peak_year  – centre of the Gaussian [calendar year]
        amplitude  – peak discovery rate [Gb/yr]
        width      – half-width (σ) [yr]

    Time grid
    ---------
    t_start, t_end, dt  [calendar years]

    Shocks
    ------
    List of (year, scale_factor) tuples.  A shock multiplies the discovery
    rate by scale_factor from that year onward (or use a ramp — see apply_shocks).
    """

    # --- pipeline stage rate constants (1/yr) ---
    k_fallow: float = 0.3
    k_build: float = 0.2
    k_maturation: float = 0.07

    # --- additional pipeline stages (optional) ---
    extra_stages: List[float] = field(default_factory=list)

    # --- discovery model ---
    discovery_pulses: List[Dict] = field(default_factory=lambda: [
        dict(peak_year=1960.0, amplitude=20.0, width=12.0),
    ])

    # --- time grid ---
    t_start: float = 1900.0
    t_end: float = 2100.0
    dt: float = 0.5  # yr

    # --- shocks: list of (year, scale_factor) ---
    shocks: List[Tuple[float, float]] = field(default_factory=list)

    def all_stage_rates(self) -> List[float]:
        return [self.k_fallow, self.k_build, self.k_maturation] + list(self.extra_stages)


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class OilShockModel:
    """Convolution-based Oil Shock Model."""

    def __init__(self, params: OSMParams):
        self.params = params
        self._build_time_grid()
        self._run()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_time_grid(self) -> None:
        p = self.params
        self.t = np.arange(p.t_start, p.t_end + p.dt * 0.5, p.dt)
        self.n = len(self.t)
        # Impulse-response time axis (starts at 0)
        self.tau = np.arange(0, self.n) * p.dt

    # ------------------------------------------------------------------
    # Discovery rate
    # ------------------------------------------------------------------

    def discovery_rate(self) -> np.ndarray:
        """Build the discovery rate D(t) from Gaussian pulses + shocks."""
        D = np.zeros(self.n)
        for pulse in self.params.discovery_pulses:
            D += pulse["amplitude"] * np.exp(
                -0.5 * ((self.t - pulse["peak_year"]) / pulse["width"]) ** 2
            )
        D = np.clip(D, 0.0, None)
        D = self._apply_shocks(D)
        return D

    def _apply_shocks(self, D: np.ndarray) -> np.ndarray:
        """Multiply discovery rate by a step-function shock at each shock year."""
        D = D.copy()
        for shock_year, scale in self.params.shocks:
            idx = np.searchsorted(self.t, shock_year)
            D[idx:] *= scale
        return D

    # ------------------------------------------------------------------
    # Transfer function (impulse response)
    # ------------------------------------------------------------------

    def stage_impulse_response(self, k: float) -> np.ndarray:
        """Single exponential stage: h(τ) = k·exp(-k·τ)."""
        return k * np.exp(-k * self.tau) * self.params.dt  # dt for Riemann sum

    def composite_impulse_response(self) -> np.ndarray:
        """h(τ) = h_1 * h_2 * ... * h_n  (sequential convolution of stages)."""
        rates = self.params.all_stage_rates()
        if not rates:
            raise ValueError("Need at least one pipeline stage.")

        h = self.stage_impulse_response(rates[0])
        for k in rates[1:]:
            h_stage = self.stage_impulse_response(k)
            # Full convolution, then truncate to original length
            h = fftconvolve(h, h_stage)[:self.n]

        # Normalise so total weight = 1 (conservation of oil)
        total = h.sum()
        if total > 0:
            h /= total
        return h

    # ------------------------------------------------------------------
    # Production
    # ------------------------------------------------------------------

    def production(self) -> np.ndarray:
        """P(t) = D(t) * h(t)  (discrete convolution, same-length output)."""
        D = self.discovery_rate()
        h = self.composite_impulse_response()
        # Use 'full' mode then keep causal part of length n
        P = fftconvolve(D, h, mode="full")[:self.n]
        return np.clip(P, 0.0, None)

    # ------------------------------------------------------------------
    # Cumulative production
    # ------------------------------------------------------------------

    def cumulative_production(self) -> np.ndarray:
        return np.cumsum(self.production()) * self.params.dt

    # ------------------------------------------------------------------
    # Run / cache results
    # ------------------------------------------------------------------

    def _run(self) -> None:
        self.D = self.discovery_rate()
        self.h = self.composite_impulse_response()
        self.P = self.production()
        self.CP = self.cumulative_production()


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def pack_params(params: OSMParams) -> np.ndarray:
    """Flatten stage rates + discovery amplitudes into a 1-D array for optimisation."""
    rates = params.all_stage_rates()
    amplitudes = [p["amplitude"] for p in params.discovery_pulses]
    return np.array(rates + amplitudes, dtype=float)


def unpack_params(x: np.ndarray, template: OSMParams) -> OSMParams:
    """Reconstruct an OSMParams from a flat array produced by pack_params."""
    import copy
    p = copy.deepcopy(template)
    n_stages = 3 + len(template.extra_stages)
    p.k_fallow, p.k_build, p.k_maturation = x[0], x[1], x[2]
    p.extra_stages = list(x[3:n_stages])
    for i, pulse in enumerate(p.discovery_pulses):
        pulse["amplitude"] = x[n_stages + i]
    return p


def fit_to_empirical(
    t_data: np.ndarray,
    P_data: np.ndarray,
    template: OSMParams,
    bounds_lo: Optional[np.ndarray] = None,
    bounds_hi: Optional[np.ndarray] = None,
) -> Tuple[OSMParams, optimize.OptimizeResult]:
    """Least-squares fit of OSM to empirical production data.

    Parameters
    ----------
    t_data   : calendar years of observations
    P_data   : observed production rates (same units as OSMParams amplitudes)
    template : starting-point OSMParams (also defines grid, shocks, etc.)

    Returns
    -------
    best_params : fitted OSMParams
    result      : scipy OptimizeResult
    """
    x0 = pack_params(template)
    n_vars = len(x0)

    if bounds_lo is None:
        bounds_lo = np.full(n_vars, 1e-4)
    if bounds_hi is None:
        bounds_hi = np.full(n_vars, 1e4)

    def residuals(x: np.ndarray) -> np.ndarray:
        try:
            p = unpack_params(np.abs(x), template)
            model = OilShockModel(p)
            P_model = np.interp(t_data, model.t, model.P)
            return P_model - P_data
        except Exception:
            return np.full_like(P_data, 1e9)

    result = optimize.least_squares(
        residuals, x0,
        bounds=(bounds_lo, bounds_hi),
        method="trf",
        verbose=0,
    )
    best_params = unpack_params(result.x, template)
    return best_params, result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_model(
    model: OilShockModel,
    empirical: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    empirical_label: str = "Empirical",
    title: str = "Oil Shock Model",
    show_discovery: bool = True,
    show_impulse: bool = True,
    show_cumulative: bool = True,
    figsize: Tuple[float, float] = (12, 9),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Comprehensive 4-panel plot of OSM results.

    Parameters
    ----------
    model       : fitted / run OilShockModel instance
    empirical   : optional (t_emp, P_emp) arrays for overlay
    """
    n_rows = 1 + int(show_discovery) + int(show_impulse) + int(show_cumulative)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(n_rows, 1, figure=fig)

    axes: List[plt.Axes] = [fig.add_subplot(gs[i]) for i in range(n_rows)]
    ax_iter = iter(axes)

    # --- Production rate ---
    ax = next(ax_iter)
    ax.plot(model.t, model.P, color="steelblue", lw=2, label="OSM production")
    if empirical is not None:
        t_emp, P_emp = empirical
        ax.scatter(t_emp, P_emp, color="tomato", s=18, zorder=5,
                   label=empirical_label, alpha=0.8)
    _mark_shocks(ax, model.params.shocks)
    ax.set_ylabel("Production rate [Gb/yr]")
    ax.set_title("Production rate")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Discovery rate ---
    if show_discovery:
        ax = next(ax_iter)
        ax.fill_between(model.t, model.D, alpha=0.4, color="goldenrod", label="Discovery D(t)")
        ax.plot(model.t, model.D, color="goldenrod", lw=1.5)
        _mark_shocks(ax, model.params.shocks)
        ax.set_ylabel("Discovery rate [Gb/yr]")
        ax.set_title("Discovery / booking rate")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # --- Impulse response ---
    if show_impulse:
        ax = next(ax_iter)
        ax.plot(model.tau, model.h / model.params.dt, color="mediumpurple", lw=2,
                label="Composite h(τ)")
        for i, k in enumerate(model.params.all_stage_rates()):
            h_stage = k * np.exp(-k * model.tau)
            ax.plot(model.tau, h_stage, lw=1, ls="--", alpha=0.6,
                    label=f"Stage {i+1}: k={k:.3f}")
        ax.set_xlabel("Lag τ [yr]")
        ax.set_ylabel("h(τ)  [1/yr]")
        ax.set_title("Pipeline transfer function (impulse response)")
        ax.set_xlim(0, min(80, model.tau[-1]))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- Cumulative ---
    if show_cumulative:
        ax = next(ax_iter)
        ax.plot(model.t, model.CP, color="seagreen", lw=2, label="Cumulative production")
        ax.set_xlabel("Year")
        ax.set_ylabel("Cumulative [Gb]")
        ax.set_title("Cumulative production")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # shared x-axis label cleanup
    for ax in axes[:-1]:
        ax.set_xlabel("")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def _mark_shocks(ax: plt.Axes, shocks: List[Tuple[float, float]]) -> None:
    for year, scale in shocks:
        ax.axvline(year, color="red", ls=":", lw=1.2, alpha=0.7)
        ax.text(year + 0.3, ax.get_ylim()[1] * 0.92,
                f"×{scale:.2f}", color="red", fontsize=7, va="top")


def plot_sensitivity(
    base_params: OSMParams,
    param_name: str,
    values: Sequence[float],
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Overlay production curves for a range of one parameter.

    Parameters
    ----------
    base_params : template OSMParams
    param_name  : attribute name on OSMParams (e.g. 'k_maturation')
    values      : sequence of values to sweep
    """
    import copy
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(values) - 1, 1)) for i in range(len(values))]

    for val, color in zip(values, colors):
        p = copy.deepcopy(base_params)
        setattr(p, param_name, val)
        m = OilShockModel(p)
        ax.plot(m.t, m.P, color=color, lw=1.8, label=f"{param_name}={val:.3g}")

    ax.set_xlabel("Year")
    ax.set_ylabel("Production rate [Gb/yr]")
    ax.set_title(f"Sensitivity: {param_name}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=min(values), vmax=max(values)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=param_name)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Demo / example
# ---------------------------------------------------------------------------

def demo() -> None:
    """Run a self-contained demonstration of the OSM."""

    # --- 1. Base model with two discovery waves ---
    params = OSMParams(
        k_fallow=0.30,
        k_build=0.15,
        k_maturation=0.06,
        discovery_pulses=[
            dict(peak_year=1955, amplitude=22.0, width=12),   # conventional peak
            dict(peak_year=1975, amplitude=10.0, width=8),    # second wave
            dict(peak_year=2010, amplitude=6.0,  width=7),    # deep-water / tight oil
        ],
        shocks=[
            (1973, 0.55),   # OPEC embargo
            (1979, 0.75),   # Iranian revolution
        ],
        t_start=1900,
        t_end=2080,
        dt=0.5,
    )

    model = OilShockModel(params)

    # --- 2. Synthetic "empirical" data (model + noise) for demonstration ---
    rng = np.random.default_rng(42)
    t_emp = np.arange(1960, 2025, 2.0)
    P_true = np.interp(t_emp, model.t, model.P)
    P_emp = np.clip(P_true * (1 + rng.normal(0, 0.08, size=t_emp.shape)), 0, None)

    # --- 3. Plot base model vs synthetic data ---
    fig = plot_model(
        model,
        empirical=(t_emp, P_emp),
        empirical_label="Synthetic data",
        title="Oil Shock Model — demonstration",
    )
    plt.show()

    # --- 4. Sensitivity sweep on k_maturation ---
    fig2 = plot_sensitivity(
        params,
        param_name="k_maturation",
        values=[0.03, 0.05, 0.07, 0.10, 0.15],
    )
    plt.show()

    # --- 5. Quick fit to the synthetic data ---
    print("\nFitting OSM to synthetic data …")
    best_params, result = fit_to_empirical(t_emp, P_emp, params)
    best_model = OilShockModel(best_params)
    print(f"  k_fallow={best_params.k_fallow:.4f}  "
          f"k_build={best_params.k_build:.4f}  "
          f"k_maturation={best_params.k_maturation:.4f}")
    print(f"  cost = {result.cost:.4f}")

    fig3 = plot_model(
        best_model,
        empirical=(t_emp, P_emp),
        empirical_label="Synthetic data",
        title="OSM — fitted parameters",
    )
    plt.show()


if __name__ == "__main__":
    demo()
