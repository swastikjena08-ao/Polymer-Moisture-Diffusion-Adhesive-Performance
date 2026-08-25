"""Exact plane-sheet Fickian solution --- the framework's reference standard.

Everything here assumes constant diffusivity, a constant surface
concentration, and a uniform initial concentration of zero. Those
restrictions are why the finite-difference solvers exist; the value of
this module is that within them the answer is exact, so it serves both as
a fast default solver and as the yardstick the numerical solvers are
verified against.

Two complementary series are used. The trigonometric (eigenfunction)
series converges rapidly at long times but needs prohibitively many terms
as ``Fo -> 0``; the short-time series built from repeated images of the
semi-infinite solution behaves the other way round. Each is evaluated
only where it converges quickly, with the crossover at
``Fo = SHORT_TIME_CROSSOVER``.

Reference: J. Crank, *The Mathematics of Diffusion*, 2nd ed., 1975, §4.3.
"""

from __future__ import annotations

import math

import numpy as np

from ..units import require_positive, require_temperature
from .base import (
    DiffusionResult,
    Exposure,
    diffusion_length,
    equivalent_sheet_thickness,
)

#: Fourier number below which the short-time series is used instead of the
#: eigenfunction series. Both converge comfortably here, so the switch is
#: continuous to well below single precision.
SHORT_TIME_CROSSOVER = 0.05

#: Fourier number at which ``M/M_inf`` reaches one half. A classical
#: constant, and the basis of the standard "half-time" method for
#: extracting a diffusivity from a gravimetric curve.
FOURIER_HALF_UPTAKE = 0.049_182_685

_MAX_TERMS = 4096
_TERM_TOLERANCE = 1e-16

_erfc = np.vectorize(math.erfc, otypes=[float])


def _ierfc(z: np.ndarray) -> np.ndarray:
    """Integral of the complementary error function, ``exp(-z^2)/sqrt(pi) - z erfc(z)``."""
    z = np.asarray(z, dtype=float)
    return np.exp(-(z**2)) / math.sqrt(math.pi) - z * _erfc(z)


def _uptake_eigen(fo: np.ndarray) -> np.ndarray:
    """Long-time eigenfunction series for normalized uptake."""
    total = np.zeros_like(fo)
    for n in range(_MAX_TERMS):
        m = 2 * n + 1
        term = (8.0 / (m**2 * math.pi**2)) * np.exp(-(m**2) * math.pi**2 * fo)
        total += term
        if term.size and np.max(term) < _TERM_TOLERANCE:
            break
    return 1.0 - total


def _uptake_short(fo: np.ndarray) -> np.ndarray:
    """Short-time image series for normalized uptake."""
    root = np.sqrt(fo)
    total = np.full_like(fo, 1.0 / math.sqrt(math.pi))
    for n in range(1, _MAX_TERMS):
        term = 2.0 * (-1.0) ** n * _ierfc(n / (2.0 * root))
        total += term
        if term.size and np.max(np.abs(term)) < _TERM_TOLERANCE:
            break
    return 4.0 * root * total


def fickian_uptake(fourier_number) -> np.ndarray:
    """Return ``M/M_inf`` as a function of Fourier number ``Fo = D t / l**2``.

    ``l`` is the equivalent sheet thickness from
    :func:`hygroadh.diffusion.base.equivalent_sheet_thickness`, so this one
    curve describes every thickness, diffusivity, and exposure geometry ---
    which is precisely why thickness and diffusivity are interchangeable in
    their effect on *normalized* uptake, and why they are not
    interchangeable in their effect on adhesion.
    """
    fo = np.asarray(fourier_number, dtype=float)
    scalar = fo.ndim == 0
    fo = np.atleast_1d(fo)
    if np.any(fo < 0.0):
        raise ValueError("fourier_number must be non-negative")
    out = np.zeros_like(fo)
    short = (fo > 0.0) & (fo < SHORT_TIME_CROSSOVER)
    long_ = fo >= SHORT_TIME_CROSSOVER
    if np.any(short):
        out[short] = _uptake_short(fo[short])
    if np.any(long_):
        out[long_] = _uptake_eigen(fo[long_])
    out = np.clip(out, 0.0, 1.0)
    return float(out[0]) if scalar else out


def _profile_eigen(fo: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Long-time eigenfunction series for the concentration profile."""
    block = np.ones((fo.size, xi.size))
    fo_col = fo[:, None]
    cos_cache = xi[None, :]
    for n in range(_MAX_TERMS):
        m = 2 * n + 1
        decay = np.exp(-(m**2) * math.pi**2 * fo_col)
        term = (
            (4.0 / math.pi)
            * ((-1.0) ** n / m)
            * decay
            * np.cos(m * math.pi * cos_cache)
        )
        block -= term
        # Bound the remaining tail by the decay factor alone: the cosine can
        # vanish at a particular position without the series having converged
        # there, so testing the full term risks stopping early.
        if np.max(decay) * (4.0 / math.pi) / m < _TERM_TOLERANCE:
            break
    return block


def _profile_short(fo: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Short-time series built from successive images of the semi-infinite solution."""
    root = np.sqrt(fo)[:, None]
    block = np.zeros((fo.size, xi.size))
    for n in range(_MAX_TERMS):
        half = n + 0.5
        term = (-1.0) ** n * (
            _erfc((half - xi)[None, :] / (2.0 * root))
            + _erfc((half + xi)[None, :] / (2.0 * root))
        )
        block += term
        if np.max(np.abs(term)) < _TERM_TOLERANCE:
            break
    return block


def fickian_profile(fourier_number, position_over_sheet) -> np.ndarray:
    """Return ``C/C_sat`` on a grid of times and positions.

    Parameters
    ----------
    fourier_number:
        Shape ``(nt,)`` dimensionless times.
    position_over_sheet:
        Shape ``(nx,)`` positions ``x/l`` measured from the no-flux plane,
        each in ``[-1/2, 1/2]``. The wetted faces are at ``+/-1/2``.

    Returns
    -------
    Array of shape ``(nt, nx)``.
    """
    fo = np.atleast_1d(np.asarray(fourier_number, dtype=float))
    xi = np.atleast_1d(np.asarray(position_over_sheet, dtype=float))
    if np.any(fo < 0.0):
        raise ValueError("fourier_number must be non-negative")
    if np.any(np.abs(xi) > 0.5 + 1e-12):
        raise ValueError("position_over_sheet must lie in [-1/2, 1/2]")
    xi = np.clip(xi, -0.5, 0.5)

    out = np.zeros((fo.size, xi.size), dtype=float)
    long_ = fo >= SHORT_TIME_CROSSOVER
    short = (fo > 0.0) & (fo < SHORT_TIME_CROSSOVER)
    if np.any(long_):
        out[long_] = _profile_eigen(fo[long_], xi)
    if np.any(short):
        out[short] = _profile_short(fo[short], xi)
    return np.clip(out, 0.0, 1.0)


def fickian_interface(fourier_number) -> np.ndarray:
    """Return ``C/C_sat`` at the no-flux plane --- the substrate interface.

    This is the adhesion-relevant history: for a one-sided film it is the
    concentration at the bondline, the last place in the film to wet.
    """
    fo = np.asarray(fourier_number, dtype=float)
    scalar = fo.ndim == 0
    values = fickian_profile(np.atleast_1d(fo), np.array([0.0]))[:, 0]
    return float(values[0]) if scalar else values


def fourier_number_for_uptake(fraction: float) -> float:
    """Invert the uptake curve: the ``Fo`` at which ``M/M_inf`` equals ``fraction``.

    Bisection on a strictly increasing function, so it is robust without
    needing a derivative. Used by the half-time diffusivity estimate.
    """
    target = float(fraction)
    if not 0.0 < target < 1.0:
        raise ValueError(f"fraction must lie strictly in (0, 1), got {target!r}")
    lo, hi = 1e-12, 1.0
    while fickian_uptake(hi) < target:
        hi *= 2.0
        if hi > 1e6:  # pragma: no cover - unreachable for fraction < 1
            raise RuntimeError("failed to bracket the target uptake fraction")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if fickian_uptake(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14 * max(hi, 1.0):
            break
    return 0.5 * (lo + hi)


def half_time(thickness: float, diffusivity: float,
              exposure: Exposure = "one_sided") -> float:
    """Return the time in seconds for the film to reach half saturation."""
    sheet = equivalent_sheet_thickness(thickness, exposure)
    d = require_positive(diffusivity, "diffusivity")
    return FOURIER_HALF_UPTAKE * sheet**2 / d


def diffusivity_from_half_time(thickness: float, half_time_s: float,
                               exposure: Exposure = "one_sided") -> float:
    """Estimate diffusivity from an observed half-saturation time.

    The textbook single-point estimate. Useful as a starting guess for the
    full least-squares calibration, and as a sanity check on its result.
    """
    sheet = equivalent_sheet_thickness(thickness, exposure)
    t_half = require_positive(half_time_s, "half_time_s")
    return FOURIER_HALF_UPTAKE * sheet**2 / t_half


def solve(
    time,
    thickness: float,
    diffusivity: float,
    saturation_pct: float,
    temperature_k: float,
    exposure: Exposure = "one_sided",
    n_depth: int | None = 41,
) -> DiffusionResult:
    """Evaluate the exact Fickian solution on the given time grid.

    Parameters
    ----------
    time:
        Ascending times in seconds. ``t = 0`` is allowed.
    thickness:
        Film thickness in metres.
    diffusivity:
        Diffusivity in m^2/s, already corrected to ``temperature_k``.
    saturation_pct:
        Equilibrium uptake in wt%, used to convert the normalized solution
        into an absolute moisture content.
    temperature_k:
        Recorded on the result; the solution itself depends on temperature
        only through ``diffusivity`` and ``saturation_pct``.
    exposure:
        ``"one_sided"`` or ``"two_sided"``.
    n_depth:
        Number of points at which to resolve the through-thickness profile,
        or ``None`` to skip it.
    """
    t = np.atleast_1d(np.asarray(time, dtype=float))
    if np.any(t < 0.0):
        raise ValueError("time must be non-negative")
    if np.any(np.diff(t) < 0.0):
        raise ValueError("time must be ascending")
    length = require_positive(thickness, "thickness")
    d = require_positive(diffusivity, "diffusivity")
    require_temperature(temperature_k)

    sheet = equivalent_sheet_thickness(length, exposure)
    fo = d * t / sheet**2

    depth = None
    profile = None
    if n_depth is not None:
        if n_depth < 2:
            raise ValueError("n_depth must be at least 2")
        depth = np.linspace(0.0, diffusion_length(length, exposure), int(n_depth))
        profile = fickian_profile(fo, 0.5 - depth / sheet)

    return DiffusionResult(
        time=t,
        uptake_normalized=fickian_uptake(fo),
        interface_normalized=fickian_interface(fo),
        saturation_pct=float(saturation_pct),
        thickness=length,
        exposure=exposure,
        diffusivity=d,
        temperature_k=float(temperature_k),
        solver="analytical",
        depth=depth,
        profile_normalized=profile,
    )
