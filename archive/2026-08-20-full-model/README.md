# hygroadh

A diffusion-based computational framework that predicts how **polymer film
thickness**, **temperature**, and **moisture diffusivity** influence

1. **normalized moisture uptake** `M(t)/M∞` of the film, and
2. a modelled **Adhesion Retention Index (ARI)** for the film/substrate
   interface,

with an interactive browser dashboard, a CLI, parametric sweeps, and sensitivity
analysis.

---

## Quick start

No install step is needed — `./hygroadh` runs the CLI straight from `src/`:

```bash
# Interactive dashboard (opens a browser)
./hygroadh serve

# ...seeded from a config file
./hygroadh serve configs/epoxy_on_steel.yaml

# One case, with CSV and JSON output
./hygroadh run configs/epoxy_on_steel.yaml --outdir out/

# Parametric sweep over thickness and temperature
./hygroadh sweep configs/epoxy_on_steel.yaml --outdir out/

# Rank the three design variables
./hygroadh sensitivity configs/epoxy_on_steel.yaml --morris

# Resolved parameters, without running anything
./hygroadh info configs/epoxy_on_steel.yaml
```

The dashboard binds `127.0.0.1:8765` by default and walks forward if that port is
taken. Every slider move posts back to the Python model and redraws; the page
contains no physics of its own, so what you see on screen is what the test suite
verifies. A full update — one simulation, a 7×7 design map, and a sensitivity
pass — takes about 18 ms.

---

## Running on another machine

Only `numpy` and Python 3.10+ are needed. There is no build step and no
virtualenv to activate.

```bash
# 1. copy the project across (from this machine)
rsync -av --exclude __pycache__ --exclude .pytest_cache ~/swastik/ user@host:~/swastik/

# 2. on the new machine
python3 -m pip install --user numpy          # required
python3 -m pip install --user pyyaml         # optional: only for YAML configs
cd ~/swastik && python3 -m pytest tests/ -q  # optional: 267 tests, ~5 s

# 3. run the dashboard
./hygroadh serve
```

### Reaching it from a different machine

By default the server binds `127.0.0.1`, so it is reachable only from the
machine it runs on. Two ways to view it from elsewhere, in order of preference:

**SSH tunnel (recommended).** Leave the server on loopback and forward the port:

```bash
# on the remote machine
./hygroadh serve --no-browser

# on your laptop
ssh -N -L 8765:localhost:8765 user@host
# then open http://localhost:8765
```

**Bind all interfaces.** Simpler, but the dashboard has **no authentication** and
anyone who can reach the port can drive it, so only do this on a network you
trust:

```bash
./hygroadh serve --host 0.0.0.0 --port 8765 --no-browser
# reachable at http://<machine-ip>:8765
```

If the port is already taken the server walks forward and prints where it
actually landed, so a stale instance will not block a new one.

### Keeping it alive after you log out

```bash
setsid nohup ./hygroadh serve --no-browser > dashboard.log 2>&1 < /dev/null &
```

Stop it with `pkill -f "hygroadh.cli serve"`.

---

## The idea the framework is built around

Global uptake `M(t)/M∞` is what a gravimetric experiment measures. But adhesion
degrades because of water **at the bondline**, and on a film bonded to an
impermeable substrate the bondline is the *last* place to wet.

That distinction has teeth. Normalized uptake is a function of the single group
`Fo = D·t/l²`, so thickness and diffusivity are interchangeable in their effect
on it — scale `D` with the square of thickness and the entire gravimetric curve
is unchanged. They are **not** interchangeable in their effect on adhesion,
because ARI also depends on absolute time (irreversible hydrolysis accumulates)
and on temperature by a route independent of `D` (the wet `Tg`).

The framework therefore always reports both the film average and the
interfacial history, and drives ARI from the latter. Two consequences the test
suite pins down:

- At a **fixed Fourier number**, three films of different thickness have the same
  normalized uptake *and* the same normalized bondline moisture, yet different
  ARI — the thicker film took four times as long to get there and hydrolysed the
  whole while.
- Whether thickness is a useful remedy depends on which mechanism binds. When
  wetting sets the life, service life scales as `L²` and doubling thickness
  quadruples it. When slow hydrolysis sets the life, doubling thickness buys
  only 10–50% — and the right remedy is a better coupling agent, not a thicker
  film.

---

## The model

### Transport

One-dimensional diffusion through the film, with an impermeable substrate:

```
∂C/∂t = ∂/∂z ( D(C,T) ∂C/∂z ),    C(0,t) = C_sat(T, RH(t)),    ∂C/∂z|_{z=L} = 0
```

Two solvers share one interface:

| Solver | Use |
|---|---|
| `diffusion/analytical.py` | Exact plane-sheet series (Crank §4.3). Constant `D`, constant surface condition. Fast default, and the verification reference for the other. |
| `diffusion/fd.py` | Cell-centred finite volume, θ-method. Adds concentration-dependent `D(C)`, humidity schedules, and surface mass-transfer resistance. |

A free film exposed on both faces is handled by symmetry: a one-sided film of
thickness `L` maps onto a `2L` plane sheet, because the no-flux substrate plane
and a symmetry mid-plane impose the same condition.

Three choices in the numerical solver worth knowing about:

- **Cell-centred, not node-centred.** Mass is conserved exactly (total uptake is
  `mean(u)`, with no quadrature error), and the discontinuous initial condition
  gives `M(0) = 0` exactly rather than a spurious `O(dz)` uptake from a surface
  node carrying half a cell of weight.
- **Boundaries as conductances in series.** `g_s = 1/(1/h + dz/2D)` makes an
  ideal Dirichlet face the `h → ∞` limit of the Robin condition, so one code
  path covers both and stays well conditioned.
- **A time step that grows with elapsed time.** A step fixed at `cfl·dz²/D`
  needs ~1.4 million steps for a 60-day exposure at 160 cells. Since a diffusing
  profile's timescale of change is itself `t`, letting the step grow makes the
  count scale with the number of decades instead: the same problem runs in 0.01 s
  at the same accuracy.

### Temperature

```
D(T)        = D_ref · exp(-(Ea/R)(1/T - 1/T_ref))
M_sat(T,RH) = M_ref · (RH/RH_ref)^b · exp(-(ΔH_s/R)(1/T - 1/T_ref))
```

Temperature moves both the rate and the equilibrium, and reaches adhesion by a
third route through the wet `Tg`. The Arrhenius form is written against a
reference temperature rather than a pre-exponential factor, which is far better
conditioned — `D_ref` stays in the numerical range of the measurement instead of
being extrapolated to infinite temperature.

### Adhesion Retention Index

ARI ∈ (0, 1] is the modelled fraction of dry interfacial fracture energy
retained, and is the product of three factors, each reported separately so the
binding mechanism is visible:

```
ARI(t) = R_plast · R_Tg · R_hyd
```

| Factor | Mechanism | Reversible? |
|---|---|---|
| `R_plast = r_min + (1-r_min)·exp(-k_p ψ^m)` | Water plasticizes the interphase | yes |
| `R_Tg = logistic((T - Tg(w) + Δ)/s)`, normalized by its dry value | Absorbed water depresses `Tg` toward the service temperature | yes |
| `R_hyd = 1 - ξ_max·ξ`, with `dξ/dt = k_h(T)·ψ^q·(1-ξ)` | Thermally activated interfacial hydrolysis | **no** |

Two details that matter:

- **The driver `ψ` is absolute water content in wt%**, not normalized moisture.
  Normalized moisture `C/C_sat` tends to 1 given enough time whatever the
  humidity, because it is normalized against *this* exposure's saturation.
  Driving degradation from it would make a film at 30% RH degrade exactly as
  fast as one at 90% RH, and would have a bone-dry film destroyed as thoroughly
  as an immersed one. With the absolute driver, service life spans a factor of
  15 across the humidity range.
- **`R_Tg` is normalized by its own dry value.** ARI is the fraction of *dry*
  adhesion retained, and the dry measurement already contains whatever softening
  the service temperature causes at `Tg(0)`. Without the normalization a bone-dry
  interface would report ARI = 0.993.

The damage integral is solved in closed form — `1 - ξ = exp(-∫k_h ψ^q dt)`,
evaluated by running trapezoid — which is exact for piecewise-linear `ψ` and keeps
`ξ ∈ [0,1]` on any time grid, with no step-size condition.

---

## Standing of the numbers

**The transport half is verified physics.** The solvers are checked against exact
analytical solutions (`<1e-3` in uptake), shown to be second-order convergent in
space, and shown to conserve mass to machine precision against the integrated
surface flux. Independent evidence that the pieces are wired together correctly:
a numerical derivative of the *full coupled model* recovers the analytical
scaling exponents for the wetting-limited service life to three digits —
thickness `+1.998` against a theoretical `+2`, diffusivity `−0.999` against `−1`.

**The ARI half is a phenomenological model with parameters you supply.** Its
functional forms encode accepted degradation mechanisms and are constructed to be
bounded and monotone, but absolute ARI values are **not** validated against
joint-strength measurements. Treat ARI as a comparative index for ranking
designs and exposure conditions, not as a prediction of residual bond strength.
This statement travels with the numbers: it appears in the dashboard and in every
CLI report, not only here.

The parameters in `configs/` are literature-plausible for structural epoxies, not
measurements of any specific product.

---

## Sensitivity analysis

Two complementary views:

- **Local elasticities** (`∂ln y/∂ln x`, by central differences) are
  dimensionless, so thickness in metres, temperature in Kelvin, and diffusivity
  in m²/s can be ranked directly against each other.
- **Morris elementary effects** screen globally, reporting both the average
  magnitude of each factor's effect (`mu*`) and how much that effect varies with
  where you are (`sigma`). A large `sigma` relative to `mu*` means the factor
  interacts, so a single local elasticity will not describe it everywhere.

One degeneracy is worth knowing about, and is pinned by a test: once an exposure
is long enough to saturate the film, **`final_ari` becomes exactly independent of
thickness and diffusivity** — those two set how fast the bondline wets, not where
it ends up. `time_to_ari_threshold` is therefore the default response.

Sobol variance decomposition is deliberately not implemented: it needs a
quasi-random sequence and many more model evaluations, and buys little over
Morris for three factors.

---

## Configuration

YAML (or JSON). Keys accept the units people actually write — `thickness_um`,
`temperature_c`, `duration_days` — and **unknown keys are errors**, because a
silently ignored `tg_depresion_per_pct` would leave the run on a default and
quietly report the wrong physics. Errors name the section:
`exposure: relative_humidity must lie in [0, 1]`.

See `configs/`:

| File | Shows |
|---|---|
| `baseline.yaml` | The minimum viable config; everything else defaulted |
| `epoxy_on_steel.yaml` | A worked example with a sweep section, fully commented |
| `humidity_cycling.yaml` | Wet/dry cycling, where the irreversible mechanism earns its place |

---

## Layout

```
src/hygroadh/
  units.py         constants, conversions, validators, exception hierarchy
  sorption.py      equilibrium isotherm, M_sat(T, RH)
  materials.py     Polymer: Arrhenius D(T), wet Tg (linear and Fox)
  diffusion/
    base.py        DiffusionResult, geometry, threshold interpolation
    analytical.py  exact plane-sheet series (both convergence branches)
    fd.py          cell-centred finite volume, Thomas solver, schedules
  adhesion.py      the three-mechanism ARI model
  config.py        strict YAML/JSON loading into typed dataclasses
  simulate.py      orchestration: the one seam where transport meets adhesion
  sweep.py         parametric grids and response surfaces
  sensitivity.py   local elasticities and Morris screening
  io_data.py       CSV and JSON output
  dashboard.py     stdlib HTTP server
  dashboard.html   self-contained page (no external scripts, styles, or fonts)
  cli.py           serve / run / sweep / sensitivity / info
```

`docs/superpowers/specs/` holds the design document.

---

## Requirements and testing

**numpy is the only runtime dependency.** `PyYAML` is optional and imported
lazily --- it is needed only to read YAML configs, and JSON configs work without
it. The dashboard, both solvers, sweeps, and sensitivity analysis all run on
numpy alone. The system `setuptools` here predates PEP 660, so `pip install -e .`
cannot work and a non-editable install would leave a stale copy shadowing
`src/`; the `./hygroadh` launcher avoids both by setting `PYTHONPATH` itself. There is no scipy dependency — the tridiagonal solve and the Nelder–Mead
optimizer are implemented directly, because scipy is not available in the target
environment.

```bash
python3 -m pytest tests/ -q     # 259 tests, ~5 s
```

`tests/conftest.py` puts `src/` on `sys.path`, so no install step is needed.

The suite is verification-led rather than coverage-led. `test_analytical.py`
checks the series against published constants (the half-uptake Fourier number
`0.049182685` is computed, not asserted from memory); `test_fd.py` checks the
numerical solver against the exact one; and `test_physics_trends.py` asserts the
qualitative claims the tool exists to make, so a refactor that breaks the science
fails loudly instead of returning plausible-looking numbers.
