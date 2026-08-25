# Polymer Moisture Diffusion & Adhesive Performance

**A reproducible scientific-computing project for studying moisture transport in polymer adhesive films.**

## Overview

Moisture can change the performance of polymer adhesives, but the bondline is often the last part of a film to become wet. This project models that transport process and estimates when moisture reaches the far face of the film.

The central question is:

> How do film thickness, temperature, and diffusivity affect the time required for moisture to reach the bondline?

This is a transport model, not a measured adhesive-strength predictor. The reported threshold time is a proxy for possible performance change, and the material values are model parameters rather than universal constants.

## What I Built

- An analytical plane-sheet solution for one-dimensional Fickian diffusion.
- A cell-centred finite-volume solver for humidity schedules and surface-transfer resistance.
- An interactive dashboard for exploring moisture profiles, thresholds, and temperature effects.
- A command-line interface for reproducible simulations, parameter sweeps, sensitivity analysis, and CSV/JSON export.
- Strict YAML/JSON configuration validation with practical unit conversions.

## Key Result

For a normalized moisture threshold, the model recovers the expected scaling:

| Variable | Effect on threshold time |
| --- | --- |
| Film thickness, `L` | `t ∝ L²` |
| Diffusivity, `D` | `t ∝ 1/D` |
| Temperature, `T` | Acts through temperature-dependent diffusivity |

The numerical sensitivity analysis recovers elasticities of **+2.000** for thickness and **-1.000** for diffusivity. These are model-based results, not experimental claims.

## Run the Dashboard

### Windows PowerShell

```powershell
py -m pip install --user numpy pyyaml
$env:PYTHONPATH = "$(Get-Location)\src"
py -m hygroadh.cli serve
```

Open <http://127.0.0.1:8765/> in a browser.

### macOS/Linux

```bash
python3 -m pip install --user numpy pyyaml
PYTHONPATH=src python3 -m hygroadh.cli serve
```

The repository also includes a small launcher for Unix-like environments:

```bash
./hygroadh serve
```

## Reproducible Examples

```bash
# Inspect resolved parameters
PYTHONPATH=src python3 -m hygroadh.cli info configs/epoxy_on_steel.yaml

# Run one case and export results
PYTHONPATH=src python3 -m hygroadh.cli run \
  configs/epoxy_on_steel.yaml --outdir out/

# Run the configured parameter sweep
PYTHONPATH=src python3 -m hygroadh.cli sweep \
  configs/epoxy_on_steel.yaml --outdir out/
```

Example configurations are provided for a baseline case, an epoxy film on steel, and humidity cycling.

## Verification

The test suite checks both numerical correctness and physical behaviour:

- The analytical solution is checked against known diffusion results.
- The finite-volume solver is compared with the analytical solution.
- Mass conservation, bounds, and spatial convergence are tested.
- The expected thickness and diffusivity scaling laws are tested.
- Dashboard responses, exports, and configuration validation are tested.

Run the tests with:

```bash
PYTHONPATH=src python3 -m pytest -q
```

## Assumptions and Limitations

The model assumes one-dimensional Fickian diffusion, a single diffusivity at a given temperature, Arrhenius temperature dependence, Henry's-law sorption, and a uniform dry initial condition.

It does not model hydrolysis, plasticization, swelling, mechanical response, glass-transition changes, interfacial degradation, non-Fickian sorption, or measured bond strength. Results should therefore be read as qualitative trends and order-of-magnitude transport estimates.

## Project Structure

```text
src/hygroadh/     Model, solvers, dashboard, CLI, and exports
tests/            Numerical, physics, configuration, and API tests
configs/          Reproducible example cases
```

## License

This project is provided for educational and research use.
