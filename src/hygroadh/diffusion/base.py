"""Shared result container and geometry handling for the diffusion solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..units import PhysicsError, require_positive

#: ``"one_sided"``  -- film bonded to an impermeable substrate, wetted on the
#: free face only. The substrate interface is the last point to wet, and is
#: the location that governs adhesion.
#: ``"two_sided"`` -- free film wetted on both faces. Solved on the
#: half-thickness by symmetry, with the mid-plane taking the role of the
#: no-flux boundary.
Exposure = Literal["one_sided", "two_sided"]


def equivalent_sheet_thickness(thickness: float, exposure: Exposure) -> float:
    """Return the plane-sheet thickness whose series solution applies.

    The classical plane-sheet result is written for a sheet of thickness
    ``l`` wetted on both faces. A one-sided film of thickness ``L`` on an
    impermeable substrate has an identical solution to the half of a
    ``2L`` sheet, because the no-flux substrate plane and the symmetry
    mid-plane impose the same condition. A two-sided free film of
    thickness ``L`` is simply ``l = L``.

    Collapsing both geometries onto one length scale is what lets the
    framework quote a single Fourier number ``Fo = D t / l**2``.
    """
    length = require_positive(thickness, "thickness")
    if exposure == "one_sided":
        return 2.0 * length
    if exposure == "two_sided":
        return length
    raise PhysicsError(
        f"exposure must be 'one_sided' or 'two_sided', got {exposure!r}"
    )


def diffusion_length(thickness: float, exposure: Exposure) -> float:
    """Return the distance from the wetted face to the no-flux plane."""
    return equivalent_sheet_thickness(thickness, exposure) / 2.0


@dataclass
class DiffusionResult:
    """Time histories produced by a diffusion solve.

    Attributes
    ----------
    time:
        Times in seconds, ascending, shape ``(nt,)``.
    uptake_normalized:
        ``M(t)/M_inf`` on ``time``, shape ``(nt,)``. This is the quantity a
        gravimetric experiment measures.
    interface_normalized:
        ``C_interface(t)/C_sat`` on ``time``, shape ``(nt,)``. Evaluated at
        the no-flux plane, i.e. the substrate interface for a one-sided
        film. This is what drives the adhesion model.
    depth:
        Grid coordinates in metres measured from the wetted face, shape
        ``(nx,)``, or ``None`` for solvers that do not resolve the profile.
    profile_normalized:
        ``C(z, t)/C_sat``, shape ``(nt, nx)``, or ``None``.
    saturation_pct:
        Equilibrium moisture content in wt% used to normalize, so absolute
        uptake can be recovered as ``uptake_normalized * saturation_pct``.
    thickness:
        Film thickness in metres.
    exposure:
        Exposure geometry the solve used.
    diffusivity:
        Diffusivity in m^2/s actually used (already temperature-corrected).
    temperature_k:
        Temperature of the solve, K.
    solver:
        Name of the solver that produced this result.
    bound_fraction:
        For the dual-stage model, the fraction of total absorbed water that
        is in the bound population at each time, shape ``(nt,)``. ``None``
        for single-phase solvers.
    """

    time: np.ndarray
    uptake_normalized: np.ndarray
    interface_normalized: np.ndarray
    saturation_pct: float
    thickness: float
    exposure: Exposure
    diffusivity: float
    temperature_k: float
    solver: str
    depth: np.ndarray | None = None
    profile_normalized: np.ndarray | None = None
    bound_fraction: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.uptake_normalized = np.asarray(self.uptake_normalized, dtype=float)
        self.interface_normalized = np.asarray(self.interface_normalized, dtype=float)
        if self.time.ndim != 1:
            raise PhysicsError("time must be one-dimensional")
        if self.uptake_normalized.shape != self.time.shape:
            raise PhysicsError("uptake_normalized must have the same shape as time")
        if self.interface_normalized.shape != self.time.shape:
            raise PhysicsError("interface_normalized must have the same shape as time")

    @property
    def uptake_pct(self) -> np.ndarray:
        """Absolute moisture uptake in weight percent."""
        return self.uptake_normalized * self.saturation_pct

    @property
    def interface_pct(self) -> np.ndarray:
        """Interfacial moisture content in weight percent."""
        return self.interface_normalized * self.saturation_pct

    @property
    def fourier_number(self) -> np.ndarray:
        """Dimensionless time ``D t / l**2`` on the equivalent sheet thickness."""
        sheet = equivalent_sheet_thickness(self.thickness, self.exposure)
        return self.diffusivity * self.time / sheet**2

    def time_to_uptake(self, fraction: float) -> float:
        """Return the time at which ``M/M_inf`` first reaches ``fraction``.

        Linearly interpolates between the two bracketing samples. Returns
        ``inf`` if the fraction is never reached within the simulated window,
        which is a meaningful answer rather than an error --- a thick, cold
        film genuinely may not get there.
        """
        return _first_crossing(self.time, self.uptake_normalized, fraction,
                               rising=True)


def _first_crossing(x: np.ndarray, y: np.ndarray, level: float,
                    rising: bool) -> float:
    """Interpolate the first ``x`` where ``y`` crosses ``level``.

    ``rising`` selects whether the crossing is from below or from above.
    Returns ``inf`` when no crossing occurs in the sampled window.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    hit = y >= level if rising else y <= level
    if not np.any(hit):
        return float("inf")
    idx = int(np.argmax(hit))
    if idx == 0:
        return float(x[0])
    y0, y1 = y[idx - 1], y[idx]
    if y1 == y0:
        return float(x[idx])
    weight = (level - y0) / (y1 - y0)
    return float(x[idx - 1] + weight * (x[idx] - x[idx - 1]))
