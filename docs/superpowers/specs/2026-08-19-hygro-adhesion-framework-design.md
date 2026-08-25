# Hygro-Adhesion Framework — Design

> **Superseded in part, 2026-08-20.** The framework was deliberately simplified
> after this document was written. Irreversible hydrolysis, reversible
> plasticization, wet-`Tg` softening, the three-mechanism ARI product, the
> thermodynamic work-of-adhesion screen, concentration-dependent diffusivity,
> and the Freundlich humidity exponent were all removed from the live code; the
> adhesion model is now a single moisture knockdown,
> `ARI = clip(1 - k·w_interface, 0, 1)`, and the isotherm is linear in humidity.
>
> This document is kept unedited because it remains the reference for the
> physics and the literature behind each removed mechanism — read it when
> restoring one. A working snapshot of the code as described here is preserved
> under `archive/2026-08-20-full-model/`. For what the code does *now*, read the
> README.


**Date:** 2026-08-19
**Status:** Approved design, ready for implementation planning
**Package name:** `hygroadh`

## Purpose

A diffusion-based computational framework that predicts how **polymer film
thickness**, **temperature**, and **moisture diffusivity** influence

1. **normalized moisture uptake** `M(t)/M∞` of the film, and
2. a **modelled Adhesion Retention Index (ARI)** for the film/substrate
   interface.

The framework must support three uses: a single forward simulation, a
parametric sweep across the three design variables, and inverse
calibration of transport parameters against measured gravimetric data.

## Scientific basis

### Why interfacial concentration, not global uptake, drives adhesion

Global uptake `M(t)/M∞` is what a gravimetric experiment measures, but
adhesion degrades because of water *at the bondline*. For a film on an
impermeable substrate, the interface is the farthest point from the
exposed face, so it wets last. Two films with identical `M(t)/M∞`
histories but different thicknesses have very different interfacial
histories. The framework therefore always reports both, and ARI is
driven by the interfacial value.

This is the central modelling decision. A framework that drove ARI from
global uptake would predict that thickness has no effect once uptake is
normalized — which is wrong.

### Transport

One-dimensional diffusion through a film of thickness `L`, coordinate
`z ∈ [0, L]` measured from the exposed face, with an impermeable
substrate at `z = L`:

```
∂C/∂t = ∂/∂z ( D(C,T) ∂C/∂z )
C(0, t) = C_sat(T, RH(t))        (Dirichlet)  or  Robin, see below
∂C/∂z |_{z=L} = 0                (impermeable substrate)
```

A free film exposed on both faces is supported by symmetry: solve the
half-thickness `L/2` with the no-flux plane at the mid-plane.

**Analytical reference solution.** For constant `D`, constant surface
concentration, and zero initial concentration, the plane-sheet series
solution (Crank, *The Mathematics of Diffusion*, 2nd ed., §4.3) applies
with equivalent sheet thickness `ℓ = 2L` for the one-sided case:

```
M(t)/M∞ = 1 - Σ_{n=0}^∞  8 / ((2n+1)²π²) · exp( -(2n+1)²π² Fo )
Fo = D t / ℓ²
```

with the concentration profile

```
C(x,t)/C_sat = 1 - (4/π) Σ_{n=0}^∞ ((-1)^n/(2n+1))
                       · exp(-(2n+1)²π² Fo) · cos((2n+1)πx/ℓ)
```

where `x` is measured from the no-flux mid-plane, so the interface is
`x = 0` and the exposed face is `x = ℓ/2`.

The trigonometric series converges slowly at small `Fo`, so the
implementation switches to the complementary short-time series below
`Fo = 0.06`:

```
M(t)/M∞ = 4 √(Fo) [ 1/√π + 2 Σ_{n=1}^∞ (-1)^n ierfc( n / (2√Fo) ) ]
```

Known checkpoints used as tests: the short-time limit
`M/M∞ = (4/ℓ)√(Dt/π)`, and the half-time `Fo₁ᐟ₂ ≈ 0.04919`.

**Numerical solver.** A θ-method (Crank–Nicolson by default, θ = 1 for
the reactive case) finite-difference solver on a uniform grid, with the
resulting tridiagonal system solved by the Thomas algorithm. This is
what enables the features the analytical solution cannot express:

- concentration-dependent diffusivity `D(C) = D_ref · exp(β_c · C/C_sat)`,
  handled by Picard iteration within each step
- a time-varying boundary humidity schedule `RH(t)`, which allows
  humidity cycling and drying/desorption legs
- a Robin (surface-resistance) boundary condition for cases where
  surface mass transfer, not bulk diffusion, is rate-limiting
- the dual-stage Langmuir model below

**Dual-stage (Langmuir / Carter–Kibler) transport.** Many epoxies and
polyimides show two-stage, non-Fickian uptake. Water partitions into a
mobile population `n` and a bound population `N`:

```
∂n/∂t = D ∂²n/∂z² - γ n + β N
∂N/∂t = γ n - β N
```

`N` has no spatial coupling, so the local ODE is eliminated
analytically before the spatial solve. With backward Euler on the
reaction terms,

```
N^{k+1} = ( N^k + Δt γ n^{k+1} ) / (1 + Δt β)
```

Substituting into the `n` equation leaves a tridiagonal system with an
effective sink `-γ/(1 + βΔt)` and an explicit source
`β N^k/(1 + βΔt)`. Equilibrium partitioning is `N∞/n∞ = γ/β`, so total
saturation is `M∞ = n∞ (1 + γ/β)`. The model must reduce to Fickian as
`γ → 0`; this is a test.

### Temperature and humidity dependence

Diffusivity, Arrhenius in reference form (better conditioned than a
pre-exponential `D₀`):

```
D(T) = D_ref · exp( -(Ea/R) (1/T - 1/T_ref) )
```

Saturation uptake, a power-law (Freundlich-like) isotherm in relative
humidity with a van 't Hoff temperature term:

```
M_sat(T, RH) = M_ref · (RH/RH_ref)^b · exp( -(ΔH_s/R) (1/T - 1/T_ref) )
```

`b = 1` recovers Henry's law. `ΔH_s` may be either sign; for most epoxies
saturation uptake rises weakly with temperature.

Glass transition depression by absorbed water, two selectable models:

```
linear:  Tg(w) = max( Tg_dry - k_g · w , Tg_floor )
Fox:     1/Tg  = (1 - w_w)/Tg_dry + w_w/Tg_water,   Tg_water = 155 K
```

where `w` is local water content in wt% and `w_w` the mass fraction.

### Adhesion Retention Index

ARI ∈ [0, 1] is the modelled fraction of dry interfacial fracture
energy retained. It is a product of three mechanism factors, each
reported separately so the dominant mechanism is visible:

```
ARI(t) = R_plast(φ) · R_Tg(T, w_int) · R_hyd(t)
```

with `φ(t) = C_int(t)/C_sat` the normalized interfacial moisture and
`w_int = φ · M_sat` the local water content in wt%.

**1. Reversible plasticization.** Water at the interphase reduces
cohesive and interfacial toughness. Decays from 1 toward a residual
floor:

```
R_plast(φ) = r_min + (1 - r_min) · exp( -k_p φ^m )
```

**2. Thermal softening near the wet Tg.** As absorbed water depresses
`Tg` toward the service temperature, the interphase softens and
adhesion collapses. A logistic in the gap between service temperature
and wet `Tg`:

```
R_Tg = 1 / ( 1 + exp( (T - Tg(w_int) + Δ_off) / s_Tg ) )
```

`Δ_off > 0` starts the knee below `Tg`, since softening precedes the
transition. This factor is what couples temperature into adhesion by a
route independent of its effect on `D`.

**3. Irreversible interfacial hydrolysis.** Accumulated, thermally
activated chemical attack on interfacial bonds, driven by local water:

```
dξ/dt = k_h(T) · φ(t)^q · (1 - ξ),    ξ(0) = 0
k_h(T) = k_h,ref · exp( -(Ea_h/R)(1/T - 1/T_ref) )
R_hyd  = 1 - ξ_max · ξ
```

Because `ξ` is monotone non-decreasing, ARI does not fully recover when
the film dries. This makes ARI history-dependent and is the mechanism
that distinguishes cyclic from steady exposure.

**Optional thermodynamic screening.** Independent of the kinetics, the
sign of the wet work of adhesion indicates whether water can
spontaneously displace the polymer from the substrate:

```
W_A,wet = γ_sw + γ_pw - γ_sp
```

`W_A,wet < 0` flags a thermodynamically unstable interface. Reported as
a standalone boolean diagnostic, not folded into ARI.

### Derived engineering outputs

- `t_x` — time for `M/M∞` to reach a given fraction
- `t_ARI` — **time-to-threshold**: time for ARI to fall to a threshold
  (default 0.8). This is the primary sweep response.
- steady-state / end-of-exposure ARI and its three components

## Architecture

Layered, with dependencies pointing only downward. No layer imports
from a layer above it.

```
cli  ──►  report ──►  sweep / sensitivity / calibrate
                              │
                              ▼
                          simulate                (orchestration)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
          diffusion        adhesion       materials       (models)
        (analytical,                     (+ sorption)
          fd, langmuir)
              └───────────────┼───────────────┘
                              ▼
                    units / config / io           (foundation)
```

### Modules

| Module | Responsibility |
|---|---|
| `units.py` | Physical constants, °C↔K, unit conversions, validation helpers |
| `config.py` | Load and validate YAML/JSON config into typed dataclasses |
| `io_data.py` | Read/write gravimetric CSV data and result tables |
| `sorption.py` | Isotherm models, `M_sat(T, RH)`, inverse `RH(M_sat)` |
| `materials.py` | `Polymer`, `Interface` dataclasses; `D(T)`, `Tg(w)` |
| `diffusion/base.py` | `DiffusionResult` container, solver protocol |
| `diffusion/analytical.py` | Plane-sheet series: uptake, profile, interface, `t_x` |
| `diffusion/fd.py` | θ-method FD solver, Thomas algorithm, Picard, Robin BC |
| `diffusion/langmuir.py` | Dual-stage Carter–Kibler solver |
| `adhesion.py` | ARI, its three factors, damage integration, work of adhesion |
| `simulate.py` | Build model from config → run transport → run ARI → bundle |
| `sweep.py` | Grids over `(L, T, D)`, response surfaces, time-to-threshold maps |
| `sensitivity.py` | Normalized local elasticities + Morris elementary effects |
| `calibrate.py` | Nelder–Mead fit of `(D, M∞)` or Langmuir params to data |
| `report.py` | Matplotlib figures + self-contained HTML report |
| `cli.py` | `run`, `sweep`, `sensitivity`, `calibrate`, `report` subcommands |

### Data flow

A forward run: config → `materials`/`sorption` evaluate `D(T)` and
`M_sat(T,RH)` → transport solver produces `M(t)/M∞` and `C_int(t)` →
`adhesion` integrates damage along that history and produces ARI(t) and
its components → `simulate` bundles everything into a `SimulationResult`
→ `report` renders it.

A sweep calls `simulate` once per grid point and reduces each result to
scalars (`t_ARI`, final ARI, `t_50` uptake), yielding response surfaces.

## Dependencies

Core is **numpy-only** — no scipy. `matplotlib` and `pyyaml` are
optional, imported lazily, with clear errors if a command needs them.
This matters because scipy is unavailable in the target environment;
the tridiagonal solve and the Nelder–Mead optimizer are therefore
implemented directly.

## Error handling

Configuration and physical-validity errors are caught at construction,
not mid-solve. A dedicated exception hierarchy (`HygroadhError` →
`ConfigError`, `PhysicsError`, `ConvergenceError`, `MissingDependencyError`)
lets the CLI report a single clear message. Specific guarantees:

- non-positive thickness, diffusivity, or temperature is rejected
- `RH` outside `[0, 1]` is rejected
- ARI parameters that could push the index outside `[0, 1]` are rejected
  at construction, and the returned index is clipped as a backstop
- the FD solver reports a `ConvergenceError` if Picard iteration fails
  rather than silently returning an unconverged field
- calibration reports the optimizer's termination reason and never
  presents a non-converged fit as converged

## Testing strategy

Verification against known solutions is the backbone, since the physics
has exact reference results.

| Test area | What it establishes |
|---|---|
| `test_analytical` | Short-time `√t` limit, half-time `Fo ≈ 0.04919`, `M/M∞ → 1`, monotonicity, series-crossover continuity |
| `test_fd_vs_analytical` | FD matches the series to < 1e-3; second-order spatial convergence; mass conservation via flux integral |
| `test_langmuir` | Reduces to Fickian as `γ → 0`; equilibrium split `γ/β`; two-stage shape |
| `test_sorption`/`test_materials` | Arrhenius and van 't Hoff limits, isotherm round-trip, Tg models bracket correctly |
| `test_adhesion` | Bounds `[0,1]`, monotone decrease in `φ`, damage irreversibility under a wet→dry cycle, factor isolation |
| `test_physics_trends` | The headline claims: thicker film ⇒ later `t_ARI`; higher `T` ⇒ earlier; higher `D` ⇒ earlier; global uptake alone does not determine ARI |
| `test_calibrate` | Recovers known `D` and `M∞` from synthetic noisy data within tolerance |
| `test_sweep`/`test_sensitivity` | Grid shapes, threshold interpolation, elasticity signs, Morris ranking |
| `test_cli` | Each subcommand runs end-to-end on a temp config and writes expected artifacts |

`test_physics_trends` is the one that matters most: it asserts the
qualitative behaviour the framework exists to predict, so a refactor
that breaks the science fails loudly.

## Scope boundaries (deliberately excluded)

- 2-D/3-D geometry and edge effects — 1-D through-thickness only
- mechanical stress, swelling strain, and hygro-mechanical coupling
- residual cure or physical ageing
- fitting ARI parameters to peel/lap-shear data (ARI is a *model*, and
  its parameters are user-supplied inputs, not fitted here)
- Sobol variance decomposition — Morris screening plus local
  elasticities cover the screening need without a scipy dependency

## Honest statement of standing

The transport half is verified physics: the diffusion solvers are
checked against exact analytical solutions and conserve mass. The ARI
half is a *phenomenological model* with user-supplied parameters. Its
functional forms are chosen to encode accepted degradation mechanisms
and to be monotone and bounded, but absolute ARI values are not
validated against joint-strength measurements. ARI should be read as a
comparative index for ranking designs and exposure conditions, not as
an absolute prediction of residual bond strength. All reports state
this.
