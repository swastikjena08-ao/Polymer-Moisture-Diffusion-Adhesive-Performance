# Polymer Moisture Diffusion & Adhesive Performance

**A Computational Study of Moisture Diffusion and Adhesion Loss in Polymer Adhesives**

> **Research question.** How do polymer-film thickness, temperature, and water
> diffusivity influence the time-dependent loss of adhesive performance?

An interactive dashboard and CLI for a simplified one-dimensional model of
moisture transport through a polymer adhesive film. It predicts **when moisture
reaches the far face of the film** — the bondline, for a film on an impermeable
substrate — and uses that as a **proxy** for when adhesive performance might
begin to change.

> ### Scope
> **This model predicts moisture transport, not measured adhesive strength.**
> It reports a moisture-penetration time, in moisture units. Any link between
> that and loss of bond strength is an assumption of the study, not an output of
> the model. The reference diffusivity and activation energy are **model
> parameters**, not universal material constants.

---

## Quick start

No install step is needed — `./hygroadh` runs the CLI straight from `src/`:

```bash
./hygroadh serve                                    # interactive dashboard
./hygroadh serve configs/epoxy_on_steel.yaml        # seeded from a config
./hygroadh run configs/epoxy_on_steel.yaml --outdir out/
./hygroadh sweep configs/epoxy_on_steel.yaml --outdir out/
./hygroadh sensitivity configs/epoxy_on_steel.yaml --morris
./hygroadh info configs/epoxy_on_steel.yaml         # resolved parameters only
```

The dashboard binds `127.0.0.1:8765` and uses the next available port if needed.
It prints the URL instead of opening a browser. Updates are computed by the
Python model, including the parameter studies and sensitivity results.

Only **numpy** and Python 3.10+ are required. See *Running on another machine*
below.

---

## The model

### Governing equation

One-dimensional Fickian diffusion through the film thickness:

```
∂C/∂t = D(T) · ∂²C/∂x²
```

with the wetted face held at equilibrium with the environment from `t = 0`, and
the far face impermeable (one-sided) or a symmetry plane (two-sided).

### Temperature dependence

```
D(T) = D_ref · exp[ -Ea/R · (1/T - 1/T_ref) ]        T in kelvin
```

Written against a reference temperature rather than a pre-exponential factor,
which is far better conditioned: `D_ref` stays in the numerical range of the
measurement instead of being extrapolated to infinite temperature. Temperature
is converted from °C to K before this is evaluated.

### Characteristic time

```
t_diffusion ~ L² / D
```

### Equilibrium uptake

Henry's law — linear in relative humidity, with an optional van 't Hoff term
that defaults to zero:

```
M_sat(T, RH) = M_ref · RH · exp[ -ΔH_s/R · (1/T - 1/T_ref) ]
```

With the default of zero, **temperature changes only how fast the film wets, not
how wet it ends up.**

### Reported criterion

The time for moisture at the far face to reach a chosen level, on either of two
bases:

| Basis | Meaning | Depends on humidity? |
|---|---|---|
| `normalized` | a fraction of the film's own equilibrium concentration | no |
| `wt_pct` | an absolute water content in weight percent | yes |

The `normalized` basis isolates the transport question, which is what the
thickness, temperature, and diffusivity studies are about. The `wt_pct` basis is
unreachable when the film cannot hold that much water — an analytic condition,
reported as such rather than as a numerical accident.

---

## What the model predicts

Three results the test suite pins down, each following from one fact: a
normalized threshold is a level on the far-face curve, and that curve is a
function of `Fo = D·t/l²` alone, so **the threshold is crossed at the same
Fourier number whatever the thickness, diffusivity, or temperature.**

| Input | Effect on penetration time | Verified as |
|---|---|---|
| Film thickness `L` | `t ∝ L²` | ratios 4.00, 4.00, 4.00, 4.00 across 50→800 µm |
| Diffusivity `D` | `t ∝ 1/D` | ratios 10.0 across 1e-14→1e-11 m²/s |
| Temperature `T` | through `D(T)` only | time ratio equals the inverse diffusivity ratio to 1e-3 |

The calculated log-log sensitivities (elasticities) come out at **+2.000** for
thickness and **−1.000** for diffusivity — the `t ~ L²/D` signature recovered
numerically from the full coupled model rather than assumed. Temperature has a
large negative elasticity because one percent of an absolute temperature is a few
kelvin and `D` depends on it exponentially; per degree it is the strongest of the
three over the ranges studied.

One consequence worth knowing: because all three inputs set how *fast* moisture
arrives rather than how much arrives eventually, **any end-of-run quantity is
blind to all three** once the film has saturated. That is why the time to cross a
threshold, not a final concentration, is the reported response.

---

## Dashboard sections

| # | Section | Contents |
|---|---|---|
| 1 | Model inputs | Thickness, temperature, RH, `D_ref`, `T_ref`, `Ea`, equilibrium uptake, threshold, window, geometry, study range. Slider plus numeric box for each, so exact values can be typed and reproduced. Model parameters are tagged as such. |
| 2 | Temperature effect | `D(T)` against temperature; penetration time against temperature; a comparison table making the chain `T ↑ → D ↑ → rate ↑ → time ↓` explicit |
| 3 | Diffusion profile | Four labelled stages at fixed Fourier numbers, a time slider to watch moisture penetrate, and the uptake history for film average and far face |
| 4 | Key results | Temperature, thickness, `D(T)`, threshold time, equilibrium uptake, threshold level, half-uptake time, characteristic `l²/D` |
| 5 | Thickness study | Factor-4 span either side of your current thickness, against an **anchored, unfitted** `L²` reference curve |
| 6 | Diffusivity study | Factor-100 span either side of your current `D_ref`, log–log, reporting both `D_ref` and the effective `D(T)` |
| 7 | Temperature study | Tabulated sweep with the Kelvin conversion and `D(T)` for every point |
| 8 | Comparison | Calculated elasticities for all three inputs, with an explanation of the exponents |
| 9 | Assumptions | The full list, on screen |
| 10 | Export | Summary, time history, and parameter studies as CSV; every chart as SVG |

**Study sample points.** The thickness and diffusivity sweeps are spaced
geometrically and **centred on the value you have set**, rather than being fixed
lists, so a sweep always straddles the operating point instead of drifting off
it — the middle sample *is* your current value. At the default 200 µm the
factor-4 span lands exactly on the conventional 50/100/200/400/800 µm series.
The temperature study range is set directly by its own two inputs.

**Simulation window.** By default it is set from the diffusion time itself
(Fourier number 1.5, near equilibrium) rather than a fixed number of hours,
because `l²/D` spans nine orders of magnitude across the allowed input ranges.
Each study point gets a window scaled to itself — otherwise a sweep over 50–800 µm
would report "not reached" for much of its range. You can set the window by hand
when a fixed exposure time is what you want to examine.

---

## Assumptions

- One-dimensional diffusion through the film thickness only
- Fickian diffusion with a single diffusion coefficient
- Constant diffusivity at a given temperature — no concentration dependence
- Arrhenius temperature dependence of diffusivity
- Simplified boundary conditions: wetted face at equilibrium from `t = 0`; far
  face impermeable or a symmetry plane
- Henry's law sorption — equilibrium uptake linear in relative humidity
- Uniform, dry initial condition
- No hydrolysis model
- No plasticization model
- No glass-transition model
- No swelling and no mechanical response
- No experimentally measured adhesion-strength prediction
- The moisture threshold is a proxy for potential performance change only

## Limitations

These limitations are documented here and summarized in the dashboard header and
assumptions section.

Real polymer adhesives may involve mechanisms outside this scope:
concentration-dependent diffusivity, swelling and the stresses it generates,
plasticization, hydrolysis, depression of the glass-transition temperature,
interfacial effects, mechanical degradation, non-Fickian or two-stage sorption,
and curved sorption isotherms.

Earlier versions modelled some of these effects. Working snapshots are preserved
under `archive/`, with notes in `archive/*/WHY-THIS-EXISTS.md`.

---

## Numerical method

| Solver | Use |
|---|---|
| `diffusion/analytical.py` | Exact plane-sheet series (Crank, *The Mathematics of Diffusion*, §4.3). The default, and the verification reference for the other. |
| `diffusion/fd.py` | Cell-centred finite volume, θ-method. Used when a humidity schedule or surface mass-transfer resistance is present. |

A one-sided film of thickness `L` maps onto a `2L` plane sheet, because the
no-flux substrate plane and a symmetry mid-plane impose the same condition. That
collapse onto a single length scale is what lets one Fourier number describe both
geometries.

Three choices in the numerical solver worth knowing about:

- **Cell-centred, not node-centred.** Mass is conserved exactly, and the
  discontinuous initial condition gives `M(0) = 0` exactly rather than a spurious
  `O(dz)` uptake from a surface node carrying half a cell of weight.
- **Boundaries as conductances in series.** `g_s = 1/(1/h + dz/2D)` makes an ideal
  face the `h → ∞` limit of a resistive one, so one code path covers both and
  stays well conditioned.
- **A time step that grows with elapsed time.** A step fixed at `cfl·dz²/D` needs
  ~1.4 million steps for a 60-day simulation at 160 cells. Since a diffusing
  profile's timescale of change is itself `t`, letting the step grow makes the
  count scale with the number of decades instead: the same problem runs in 0.01 s
  at the same accuracy.

**Verification.** The finite-volume solver matches the exact series to under
1e-3 in uptake, is second-order convergent in space, and conserves mass to
machine precision against the integrated surface flux. The half-uptake Fourier
number `0.049182685` used by the tests was computed, not quoted from memory.

---

## Configuration

YAML (or JSON). Keys accept the units people actually write — `thickness_um`,
`temperature_c`, `duration_days`, `activation_energy_kj` — and **unknown keys are
errors**, because a silently ignored key would leave the run on a default and
quietly report the wrong physics. Errors name the section:
`exposure: relative_humidity must lie in [0, 1]`.

| File | Shows |
|---|---|
| `configs/baseline.yaml` | The minimum viable config; everything else defaulted |
| `configs/epoxy_on_steel.yaml` | A worked example with a sweep section, fully commented |
| `configs/humidity_cycling.yaml` | Wet/dry cycling, and an absolute-wt% threshold |

---

## Running on another machine

The environment is managed with [uv](https://docs.astral.sh/uv/). **A virtual
environment is not portable** — it holds absolute paths and platform-specific
binaries. What travels between machines is `pyproject.toml` plus `uv.lock`; `uv
sync` rebuilds an identical environment from those, down to the exact package
versions.

```bash
# 1. copy the project across, WITHOUT the virtual environment
rsync -av --exclude .venv --exclude __pycache__ --exclude .pytest_cache \
      ~/swastik/ user@host:~/swastik/

# 2. on the new machine, install uv if it is not already there
curl -LsSf https://astral.sh/uv/install.sh | sh     # or: pip install --user uv

# 3. build the environment from the lockfile and run
cd ~/swastik
uv sync                 # creates .venv, installs the locked versions
uv run pytest -q        # optional: 318 tests, ~6 s
./hygroadh serve
```

`uv sync` fetches a suitable Python itself if the machine has none satisfying
`requires-python = ">=3.10"`, so there is no separate interpreter to install. The
`./hygroadh` launcher prefers `.venv` when it exists and falls back to a bare
`python3` with `PYTHONPATH=src` when it does not, so the CLI works either way.

Equivalent uv commands:

| Task | Command |
|---|---|
| Build/refresh the environment | `uv sync` |
| Run the test suite | `uv run pytest -q` |
| Run the dashboard | `uv run hygroadh serve` |
| Re-resolve dependencies | `uv lock --upgrade` |
| Exact reproduction, no re-resolve | `uv sync --frozen` |

### Dependencies

`numpy` is the only runtime dependency. `pyyaml` is needed only to read the YAML
configs — JSON configs work without it — and is in the `dev` group so `uv sync`
installs it by default.

The declared floor of `numpy>=1.23` is **tested, not assumed**: the suite passes
on Python 3.10.12 with numpy 1.26.4 and on Python 3.12.13 with numpy 2.5.2. The
two tests that integrate a curve go through `tests/_compat.py`, because
`np.trapezoid` only exists from numpy 2.0 and the older spelling is `np.trapz`.

### Without uv

If uv is unavailable, the project still runs on a bare interpreter:

```bash
python3 -m pip install --user numpy pyyaml
./hygroadh serve
```

### On WSL

Fully supported, with two adjustments made for it:

**No browser is opened.** WSL usually has no Linux browser, and `webbrowser.open`
can hang there. The server prints the URL and you open it yourself. Pass `--open`
if you do want it launched — under WSL that goes through `wslview` or
`explorer.exe` rather than the Linux `webbrowser` module.

**The URL you need is printed for you.** Bound to loopback (the default) the
dashboard is reached from a Windows browser through WSL's localhost forwarding:

```
$ ./hygroadh serve
hygroadh dashboard is serving on port 8765.

  open in your browser: http://localhost:8765/           (from this machine)

  Running under WSL. Windows forwards localhost into WSL, so the URL
  above should work in a Windows browser. If it does not, stop this
  and rerun with:
      ./hygroadh serve --host 0.0.0.0
  then open  http://172.20.1.5:8765/  from Windows.

  Physics runs in Python; the page only draws it. Ctrl-C to stop.
```

Localhost forwarding is the normal path and usually just works. When it does not
— older WSL builds, or a distro with `networkingMode` changed — bind all
interfaces and use the printed address. Inside WSL2 that exposes the port to the
Windows host, not to your LAN, since WSL2 sits behind NAT; Windows does not
forward it outward without an explicit `netsh portproxy` rule.

Two further WSL notes:

- Keep the project on the **Linux filesystem** (`~/swastik`), not under
  `/mnt/c/...`. Cross-filesystem I/O is an order of magnitude slower, which the
  test suite and `uv sync` both feel.
- `uv` installs cleanly in WSL with the standard installer, and will fetch its
  own Python if the distro has none new enough.

### Reaching it from a different machine

By default the server binds `127.0.0.1`, reachable only from the machine it runs
on. **SSH tunnel (recommended):**

```bash
./hygroadh serve                         # on the remote machine
ssh -N -L 8765:localhost:8765 user@host  # on your laptop
```

**Binding all interfaces** is simpler, but the dashboard has **no
authentication** — only do this on a network you trust:

```bash
./hygroadh serve --host 0.0.0.0 --port 8765
```

### Keeping it alive after logout

```bash
setsid nohup ./hygroadh serve > dashboard.log 2>&1 < /dev/null &
```

Stop it with `pkill -f "hygroadh.cli serve"`.

---

## Layout

```
src/hygroadh/
  units.py         constants, conversions, validators, exception hierarchy
  sorption.py      Henry's-law isotherm, M_sat(T, RH)
  materials.py     Polymer: Arrhenius D(T)
  threshold.py     the moisture-penetration criterion
  diffusion/
    base.py        result container, geometry, threshold interpolation
    analytical.py  exact plane-sheet series (both convergence branches)
    fd.py          cell-centred finite volume, Thomas solver, schedules
  config.py        strict YAML/JSON loading into typed dataclasses
  simulate.py      orchestration: the one seam where transport meets the criterion
  sweep.py         parametric grids and response surfaces
  sensitivity.py   local elasticities and Morris screening
  io_data.py       CSV and JSON output
  dashboard.py     stdlib HTTP server and CSV export
  dashboard.html   self-contained page (no external scripts, styles, or fonts)
  cli.py           serve / run / sweep / sensitivity / info
tests/_compat.py   numpy 1.x / 2.x shim, so the declared floor stays testable
archive/           earlier, fuller models, preserved with restoration notes
docs/superpowers/specs/   the original design document
pyproject.toml     packaging metadata; uv.lock pins the exact versions
```

---

## Testing

```bash
uv run pytest -q                # 318 tests, ~6 s
python3 -m pytest tests/ -q     # same, without uv
```

`tests/conftest.py` puts `src/` and `tests/` on `sys.path`, so the suite runs
with or without an install step. `PyYAML` is optional and imported lazily.

The suite is verification-led rather than coverage-led:

- `test_analytical.py` checks the series against published constants
- `test_fd.py` checks the numerical solver against the exact one
- `test_physics_trends.py` asserts the qualitative claims the study makes,
  including its limitations, so a refactor that breaks the science fails loudly
- `test_dashboard.py` includes a page/server contract check, because the page is
  hand-written with no build step and a renamed field would otherwise leave a
  blank chart with no error anywhere

Every equation in the model is one that can be written down and explained: the
diffusion equation, the Arrhenius relationship, Henry's law, and a threshold
comparison. Nothing else is hidden in the numbers.
