"""Finite-volume moisture diffusion solver.

Handles what the analytical solution cannot: a surface condition that varies in
time, and finite surface mass-transfer resistance. Diffusivity is constant ---
independent of moisture content --- so the equation stays linear and each time
step is a single tridiagonal solve.

Discretization
--------------
A **cell-centred finite volume** scheme on ``n_cells`` cells of width
``dz`` spanning the wetted face to the no-flux plane. Two properties
motivate the choice over a node-centred grid:

* Mass is conserved exactly. The total uptake is ``mean(u)`` with no
  quadrature error, so the reported ``M/M_inf`` is the true integral of the
  discrete solution.
* The initial condition is discontinuous at the wetted face --- the surface
  jumps to saturation at ``t = 0``. On a node-centred grid the surface node
  carries half a cell of weight, giving a spurious ``O(dz)`` uptake at
  ``t = 0``. With cell averages every cell starts dry and ``M(0) = 0``
  exactly.

The surface flux is written as a conductance in series, which unifies the
two boundary conditions instead of branching on them::

    Dirichlet:  g_s = D / (dz/2)                 (no surface resistance)
    Robin:      g_s = 1 / (1/h + dz/(2 D))       (mass-transfer film + half cell)

Dirichlet is exactly the ``h -> inf`` limit of the Robin form, so one code
path covers both and stays well conditioned --- unlike imposing Dirichlet
as a large-``h`` penalty.

Time integration is the theta-method: ``theta=0.5`` is Crank-Nicolson
(second order), ``theta=1`` is backward Euler (first order,
unconditionally monotone). The first few steps optionally use backward
Euler regardless of ``theta`` --- Rannacher startup --- which damps the
oscillation Crank-Nicolson otherwise produces against the initial
discontinuity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..units import (
    ConvergenceError,
    PhysicsError,
    require_non_negative,
    require_positive,
    require_temperature,
)
from .base import DiffusionResult, Exposure, diffusion_length

SurfaceCondition = float | Callable[[float], float]


@dataclass(frozen=True)
class Schedule:
    """A piecewise-linear function of time, used for time-varying exposure.

    Values are interpolated between the given knots and held constant
    outside them. Expressed in whatever unit the consumer needs --- this
    class carries no physics, so it serves equally as a relative-humidity
    history or a normalized surface-concentration history.
    """

    times: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        t = np.atleast_1d(np.asarray(self.times, dtype=float))
        v = np.atleast_1d(np.asarray(self.values, dtype=float))
        if t.shape != v.shape:
            raise PhysicsError("Schedule times and values must have the same shape")
        if t.size == 0:
            raise PhysicsError("Schedule needs at least one knot")
        if np.any(np.diff(t) < 0.0):
            raise PhysicsError("Schedule times must be ascending")
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "values", v)

    def __call__(self, time: float) -> float:
        return float(np.interp(float(time), self.times, self.values))

    @property
    def maximum(self) -> float:
        """Largest value the schedule takes, for step-size selection."""
        return float(np.max(self.values))

    @classmethod
    def constant(cls, value: float) -> "Schedule":
        """A schedule that never changes."""
        return cls(np.array([0.0]), np.array([float(value)]))

    @classmethod
    def cycle(cls, high: float, low: float, period: float, n_cycles: int,
              duty: float = 0.5, ramp: float = 0.0) -> "Schedule":
        """A repeating high/low exposure cycle.

        Parameters
        ----------
        high, low:
            The two exposure levels.
        period:
            Duration of one full cycle in seconds.
        n_cycles:
            How many cycles to generate.
        duty:
            Fraction of each period spent at ``high``.
        ramp:
            Transition time in seconds between levels. Zero gives square
            steps, which the piecewise-linear interpolation renders as
            near-vertical edges.
        """
        period = require_positive(period, "period")
        if n_cycles < 1:
            raise PhysicsError("n_cycles must be at least 1")
        if not 0.0 < duty < 1.0:
            raise PhysicsError("duty must lie strictly in (0, 1)")
        ramp = min(require_non_negative(ramp, "ramp"), 0.25 * period)
        edge = max(ramp, 1e-9 * period)
        times: list[float] = []
        values: list[float] = []
        for k in range(int(n_cycles)):
            t0 = k * period
            t_switch = t0 + duty * period
            times += [t0, t_switch, t_switch + edge, t0 + period]
            values += [high, high, low, low]
        return cls(np.array(times), np.array(values))


def _as_callable(condition: SurfaceCondition) -> tuple[Callable[[float], float], float]:
    """Normalize a surface condition to a callable plus its maximum value."""
    if callable(condition):
        probe = getattr(condition, "maximum", None)
        return condition, float(probe) if probe is not None else 1.0
    value = float(condition)
    return (lambda _t, _v=value: _v), value


def solve_tridiagonal(lower, diag, upper, rhs) -> np.ndarray:
    """Solve a tridiagonal system by the Thomas algorithm.

    ``lower[i]`` multiplies ``x[i-1]`` and ``upper[i]`` multiplies
    ``x[i+1]``, so ``lower[0]`` and ``upper[-1]`` are ignored. Operates on
    Python lists internally because the recurrence is inherently
    sequential, and per-element numpy indexing costs far more than plain
    floats in that loop.

    Raises
    ------
    ConvergenceError
        If a pivot vanishes, which for these diffusion operators means the
        matrix was built with non-physical coefficients.
    """
    a = np.asarray(lower, dtype=float).tolist()
    b = np.asarray(diag, dtype=float).tolist()
    c = np.asarray(upper, dtype=float).tolist()
    d = np.asarray(rhs, dtype=float).tolist()
    n = len(b)
    if not (len(a) == len(c) == len(d) == n):
        raise ValueError("tridiagonal bands and right-hand side must be the same length")

    cp = [0.0] * n
    dp = [0.0] * n
    pivot = b[0]
    if pivot == 0.0:
        raise ConvergenceError("tridiagonal solve failed: zero pivot at row 0")
    cp[0] = c[0] / pivot
    dp[0] = d[0] / pivot
    for i in range(1, n):
        pivot = b[i] - a[i] * cp[i - 1]
        if pivot == 0.0:
            raise ConvergenceError(f"tridiagonal solve failed: zero pivot at row {i}")
        cp[i] = c[i] / pivot
        dp[i] = (d[i] - a[i] * dp[i - 1]) / pivot

    x = [0.0] * n
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return np.array(x)


def _face_conductance(n_cells: int, d_ref: float, dz: float,
                      surface_transfer: float | None) -> tuple[float, np.ndarray]:
    """Return the surface conductance and the interior face conductances.

    Conductances have units of m/s, so that ``g * delta_u`` is a flux. With a
    constant diffusivity they depend only on the geometry, so they are the same
    at every time step and for every solution state.
    """
    g_interior = np.full(n_cells - 1, d_ref / dz)
    half_cell_resistance = dz / (2.0 * d_ref)
    if surface_transfer is None:
        g_surface = 1.0 / half_cell_resistance
    else:
        g_surface = 1.0 / (1.0 / surface_transfer + half_cell_resistance)
    return float(g_surface), g_interior


def _operator(n_cells: int, d_ref: float, dz: float,
              surface_transfer: float | None,
              surface_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble ``du/dt = A u + b`` as tridiagonal bands plus a source vector."""
    n = n_cells
    g_s, g = _face_conductance(n_cells, d_ref, dz, surface_transfer)

    lower = np.zeros(n)
    diag = np.zeros(n)
    upper = np.zeros(n)
    src = np.zeros(n)

    # Cell 0: environment on one side, first interior face on the other.
    diag[0] = -(g_s + g[0]) / dz
    upper[0] = g[0] / dz
    src[0] = g_s * surface_value / dz

    if n > 2:
        lower[1:-1] = g[:-1] / dz
        diag[1:-1] = -(g[:-1] + g[1:]) / dz
        upper[1:-1] = g[1:] / dz

    # Last cell: no flux through the substrate, so only the interior face acts.
    lower[-1] = g[-1] / dz
    diag[-1] = -g[-1] / dz
    return lower, diag, upper, src


def _step(u_old: np.ndarray, dt: float, theta: float, d_ref: float,
          dz: float, surface_transfer: float | None, surface_old: float,
          surface_new: float) -> np.ndarray:
    """Advance one time step of the theta-method.

    Constant diffusivity makes the equation linear, so this is a single
    tridiagonal solve --- no iteration, and no possibility of a convergence
    failure inside a step. Only the boundary source differs between the old and
    new time levels.
    """
    n = u_old.size
    lo, di, up, src_old = _operator(n, d_ref, dz, surface_transfer, surface_old)
    _, _, _, src_new = _operator(n, d_ref, dz, surface_transfer, surface_new)
    rhs = u_old + (1.0 - theta) * dt * (
        _apply_bands(lo, di, up, u_old) + src_old
    ) + theta * dt * src_new
    return solve_tridiagonal(
        -theta * dt * lo, 1.0 - theta * dt * di, -theta * dt * up, rhs
    )


def _apply_bands(lower, diag, upper, x) -> np.ndarray:
    """Multiply a tridiagonal operator by a vector."""
    out = diag * x
    out[:-1] += upper[:-1] * x[1:]
    out[1:] += lower[1:] * x[:-1]
    return out


def solve(
    time,
    thickness: float,
    diffusivity: float,
    saturation_pct: float,
    temperature_k: float,
    exposure: Exposure = "one_sided",
    n_cells: int = 80,
    surface: SurfaceCondition = 1.0,
    surface_transfer: float | None = None,
    initial: float = 0.0,
    theta: float = 0.5,
    cfl: float = 4.0,
    step_growth: float = 0.08,
    rannacher_steps: int = 2,
    store_profile: bool = True,
) -> DiffusionResult:
    """Integrate the diffusion equation and return uptake and interface histories.

    Parameters
    ----------
    time:
        Ascending output times in seconds. ``t = 0`` records the initial state.
    thickness:
        Film thickness in metres.
    diffusivity:
        Diffusivity in m^2/s, already corrected to ``temperature_k``.
    saturation_pct:
        Equilibrium uptake in wt% that ``u = 1`` corresponds to.
    temperature_k:
        Temperature of the solve, K.
    exposure:
        ``"one_sided"`` or ``"two_sided"``.
    n_cells:
        Number of finite volumes across the diffusion length.
    surface:
        Normalized surface concentration, either a constant or a callable of
        time (a :class:`Schedule`, or anything built by
        :mod:`hygroadh.simulate` from a humidity history). ``1.0`` means the
        face sits at the saturation the result is normalized against.
    surface_transfer:
        Surface mass-transfer coefficient in m/s for a Robin condition, or
        ``None`` for an ideal Dirichlet face.
    initial:
        Uniform initial normalized concentration.
    theta:
        ``0.5`` for Crank-Nicolson, ``1.0`` for backward Euler.
    cfl:
        Target ``D dt / dz**2`` per step while the profile still has structure
        at the grid scale. The scheme is unconditionally stable, so this
        controls accuracy rather than stability.
    step_growth:
        Permitted step size as a fraction of elapsed time, which is what makes
        long exposures affordable. A diffusing profile's timescale of change is
        itself ``t`` --- self-similar early, exponentially relaxing late --- so a
        step fixed at ``cfl dz**2/D`` is only necessary near ``t = 0`` and is
        wildly over-resolved by the end. Holding it fixed makes the step count
        scale with ``duration * D / dz**2``: a 60-day exposure at 160 cells
        needs about 1.4 million steps. Allowing the step to grow makes the
        count scale with the number of *decades* instead, roughly
        ``ln(10)/step_growth`` per decade, which is a few hundred steps for the
        same problem at the same accuracy. Set to 0 to recover a fixed step.
    rannacher_steps:
        Number of leading steps forced to backward Euler to damp the initial
        discontinuity.
    store_profile:
        Whether to record the through-thickness profile at each output time.
    """
    t_out = np.atleast_1d(np.asarray(time, dtype=float))
    if np.any(t_out < 0.0):
        raise ValueError("time must be non-negative")
    if np.any(np.diff(t_out) < 0.0):
        raise ValueError("time must be ascending")
    length = require_positive(thickness, "thickness")
    d_ref = require_positive(diffusivity, "diffusivity")
    require_temperature(temperature_k)
    if n_cells < 2:
        raise PhysicsError("n_cells must be at least 2")
    if not 0.0 <= theta <= 1.0:
        raise PhysicsError("theta must lie in [0, 1]")
    require_positive(cfl, "cfl")
    require_non_negative(step_growth, "step_growth")
    if surface_transfer is not None:
        require_positive(surface_transfer, "surface_transfer")

    span = diffusion_length(length, exposure)
    dz = span / n_cells
    centres = (np.arange(n_cells) + 0.5) * dz

    surface_fn, _surface_max = _as_callable(surface)
    dt_diffusive = cfl * dz**2 / d_ref

    u = np.full(n_cells, float(initial))
    uptake = np.empty(t_out.size)
    interface = np.empty(t_out.size)
    profile = np.empty((t_out.size, n_cells)) if store_profile else None

    def record(index: int) -> None:
        uptake[index] = float(np.mean(u))
        interface[index] = float(u[-1])
        if profile is not None:
            profile[index] = u

    steps_taken = 0
    t_now = 0.0
    start = 0
    if t_out[0] == 0.0:
        record(0)
        start = 1

    for index in range(start, t_out.size):
        t_target = float(t_out[index])
        if t_target <= t_now:
            record(index)
            continue
        # Sub-step to the next output time, letting the step grow with elapsed
        # time but never overshooting the output sample.
        while t_now < t_target:
            allowed = max(dt_diffusive, step_growth * t_now)
            dt = min(t_target - t_now, allowed)
            step_theta = 1.0 if steps_taken < rannacher_steps else theta
            u = _step(
                u,
                dt,
                step_theta,
                d_ref,
                dz,
                surface_transfer,
                surface_fn(t_now),
                surface_fn(t_now + dt),
            )
            t_now = min(t_now + dt, t_target)
            steps_taken += 1
        t_now = t_target
        record(index)

    return DiffusionResult(
        time=t_out,
        uptake_normalized=uptake,
        interface_normalized=interface,
        saturation_pct=float(saturation_pct),
        thickness=length,
        exposure=exposure,
        diffusivity=d_ref,
        temperature_k=float(temperature_k),
        solver="finite_volume",
        depth=centres,
        profile_normalized=profile,
    )
