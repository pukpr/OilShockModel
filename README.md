# Oil Shock Model

Python implementation of the **Oil Shock Model (OSM)** from  
*Mathematical GeoEnergy: Discovery, Depletion, and Renewal*  
Paul Pukite, Dennis Coyne, Dan Challou (Wiley, 2019)

---

## Overview

The Oil Shock Model is a compartmental Markov-chain model that tracks
oil as it moves through a series of pipeline stages from underground
discovery to surface production:

```
Discovery ──► Fallow ──► Construction ──► Producing ──► Extracted
   D(t)         F(t)          C(t)           P(t)        cumul.
```

Each stage transition is memoryless (exponentially distributed delay),
which converts the underlying convolution integral into a compact system
of linear ODEs:

```
dF/dt = D(t) - F(t) / τ_f
dC/dt = F(t) / τ_f  - C(t) / τ_c
dP/dt = C(t) / τ_c  - P(t) / τ_x

Production rate = P(t) / τ_x
```

| Symbol  | Meaning                                    | Typical value |
|---------|--------------------------------------------|---------------|
| D(t)    | Discovery input rate (Gb/yr)               | logistic pulse|
| τ\_f    | Mean **fallow** delay (years)              | 5–10 yr       |
| τ\_c    | Mean **construction/development** delay    | 2–5 yr        |
| τ\_x    | Mean **extraction/depletion** time         | 10–40 yr      |

The discovery function D(t) is modelled as a sum of logistic-growth
pulses, each parameterised by a peak year t₀, a growth rate k, and a
total ultimate discovery scale (Gb).

---

## Repository structure

```
OilShockModel/
├── oil_shock_model.py   # Core model module
├── run_us_oil.py        # US production example and plot
├── data/
│   └── us_production.csv  # EIA annual US crude-oil production (Gb/yr)
├── tests/
│   └── test_oil_shock_model.py  # pytest unit tests
└── README.md
```

---

## Quick start

### Requirements

```bash
pip install numpy scipy matplotlib
```

### Run the US example

```bash
python run_us_oil.py
```

This fits the OSM to EIA historical US crude-oil production data and
saves a plot to `us_oil_osm.png`.

### Run the tests

```bash
python -m pytest tests/ -v
```

---

## API reference

### `oil_shock_model.run_osm(t, discovery_func, tau_f, tau_c, tau_x, ...)`

Integrate the OSM ODEs and return a result dictionary with keys:

| Key             | Description                              |
|-----------------|------------------------------------------|
| `t`             | time array                               |
| `discovery`     | D(t) – discovery rate (Gb/yr)            |
| `fallow`        | F(t) – fallow-stage reserves (Gb)        |
| `construction`  | C(t) – construction-stage reserves (Gb) |
| `producing`     | P(t) – producing reserves (Gb)           |
| `production`    | P(t)/τ\_x – production rate (Gb/yr)     |
| `cumulative`    | cumulative production (Gb)               |

### `oil_shock_model.OilShockModel`

Object-oriented wrapper with a `.fit()` method for parameter estimation
against observed production data.

```python
from oil_shock_model import OilShockModel
import numpy as np

model = OilShockModel(
    tau_f=5.0,  tau_c=3.0,  tau_x=30.0,
    disc_params=[(1945.0, 0.12, 200.0),   # conventional era
                 (2008.0, 0.45, 150.0)],  # shale era
)
t = np.linspace(1900, 2060, 1000)
result = model.run(t)
print(result['production'])   # Gb/yr
```

### Discovery input functions

| Function                           | Description                       |
|------------------------------------|-----------------------------------|
| `logistic_pulse(t, t0, k, scale)`  | Logistic-growth bell curve        |
| `gaussian_pulse(t, t0, sigma, scale)` | Gaussian bell curve            |
| `multi_logistic_discovery(t, params)` | Sum of logistic pulses         |

---

## Mathematical background

The Oil Shock Model was developed to explain the systematic lag between
oil discovery and peak production.  A key observation is that oil
production follows discovery with a delay governed by the time constants
of the fallow, construction, and extraction stages.

The equivalent convolution representation is:

```
P(t) = D(t) * g_f(t) * g_c(t) * g_x(t)
```

where `*` denotes convolution and each `g_i(t) = (1/τ_i) exp(-t/τ_i)`
is an exponential impulse-response kernel.  The ODE formulation is
numerically preferred for fitting because it avoids discretisation errors
in long convolutions.

For further reading see Chapter 8 of *Mathematical GeoEnergy* (Wiley, 2019)
and the associated blog at https://geoenergymath.com/.

---

## AI content

Note that the above was initially created by GitHub Copilot.  An alternate implementation 
```
oil_shock_model_claude.py
```

was created by Claude Code as described in the post https://geoenergymath.com/2026/03/15/claude-code-oil-shock-model/


## License

See repository licence file.
