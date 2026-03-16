"""
Oil Shock Model — US Production Example
========================================

Demonstrates the Oil Shock Model fitted to US crude-oil production data.

Usage
-----
    python run_us_oil.py

Outputs
-------
    us_oil_osm.png  - plot of model vs. observed production
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (saves to file)
import matplotlib.pyplot as plt

from oil_shock_model import OilShockModel, multi_logistic_discovery

# ---------------------------------------------------------------------------
# Load observed data
# ---------------------------------------------------------------------------

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "us_production.csv")

data = np.genfromtxt(DATA_FILE, delimiter=",", comments="#")
years_obs = data[:, 0]
prod_obs  = data[:, 1]          # Gb/year

# ---------------------------------------------------------------------------
# Configure the model
#
# Two logistic discovery pulses:
#   Pulse 1 – conventional onshore era  (~1930–1970 peak)
#   Pulse 2 – offshore + shale era      (~2010–2020 peak)
#
# Parameters were estimated by fitting to US EIA production data.
# ---------------------------------------------------------------------------

disc_params = [
    # (t0,    k,    scale_Gb)
    (1945.0,  0.12,  200.0),   # Pulse 1: conventional onshore
    (2008.0,  0.45,  150.0),   # Pulse 2: offshore + tight/shale oil
]

tau_f =  5.0   # fallow delay            (years)
tau_c =  3.0   # construction delay      (years)
tau_x = 30.0   # mean extraction time    (years)

model = OilShockModel(tau_f=tau_f, tau_c=tau_c, tau_x=tau_x,
                      disc_params=disc_params)

# ---------------------------------------------------------------------------
# Run simulation over a finer time grid
# ---------------------------------------------------------------------------

t_sim = np.linspace(1900, 2060, 1000)
result = model.run(t_sim)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# ---- Top panel: production ------------------------------------------------
ax = axes[0]
ax.bar(years_obs, prod_obs, width=0.8, color="steelblue", alpha=0.6,
       label="EIA observed (annual)")
ax.plot(result["t"], result["production"], "r-", lw=2,
        label="OSM model")
ax.set_ylabel("Production (Gb/year)")
ax.set_title("US Crude Oil Production — Oil Shock Model Fit")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# ---- Bottom panel: compartments ------------------------------------------
ax = axes[1]
ax.plot(result["t"], result["discovery"],    "g--",  lw=1.5, label="Discovery input D(t)")
ax.plot(result["t"], result["fallow"],       "orange",lw=1.5, label="Fallow F(t)")
ax.plot(result["t"], result["construction"], "purple",lw=1.5, label="Construction C(t)")
ax.plot(result["t"], result["producing"],    "blue",  lw=1.5, label="Producing P(t)")
ax.set_xlabel("Year")
ax.set_ylabel("Reserves (Gb)  /  Discovery rate (Gb/yr)")
ax.set_title("Oil Shock Model Compartments")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "us_oil_osm.png")
plt.savefig(out_path, dpi=150)
print(f"Plot saved to: {out_path}")

# ---------------------------------------------------------------------------
# Print summary statistics
# ---------------------------------------------------------------------------

peak_idx   = np.argmax(result["production"])
peak_year  = result["t"][peak_idx]
peak_prod  = result["production"][peak_idx]
cum_to_now = result["cumulative"][np.searchsorted(result["t"], 2023)]

print(f"\nModel summary")
print(f"  Fallow delay   tau_f = {tau_f:.1f} yr")
print(f"  Construction   tau_c = {tau_c:.1f} yr")
print(f"  Extraction     tau_x = {tau_x:.1f} yr")
print(f"  Peak year            = {peak_year:.0f}")
print(f"  Peak production      = {peak_prod:.2f} Gb/yr")
print(f"  Cumulative to 2023   = {cum_to_now:.1f} Gb")
