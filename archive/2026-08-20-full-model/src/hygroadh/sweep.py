"""Parametric sweeps over the three design variables.

Each grid point is a full simulation reduced to a handful of scalars, which
turns the design space into response surfaces that can be contoured. The
headline response is ``time_to_ari_threshold_s``: the service life before
modelled adhesion falls to the configured threshold.

Unreachable thresholds are reported as ``inf`` rather than raised. A thick,
cold film genuinely may never cross the threshold inside the simulated window,
and "survives the whole exposure" is the most useful answer a design tool can
give --- not an error.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .config import SweepAxes
from .simulate import Case, SimulationResult, run
from .units import KELVIN_OFFSET, PhysicsError

#: Axis name -> how to apply a value of it to a case.
AXIS_SETTERS: dict[str, Callable[[Case, float], Case]] = {
    "thickness": lambda case, value: case.with_thickness(value),
    "temperature_k": lambda case, value: case.with_temperature(value),
    "diffusivity": lambda case, value: case.with_diffusivity(value),
}

#: Scalars extracted at every grid point.
RESPONSE_NAMES = (
    "time_to_ari_threshold_s",
    "final_ari",
    "time_to_half_uptake_s",
    "final_uptake_normalized",
    "final_interface_normalized",
    "diffusivity_m2_s",
    "saturation_pct",
)

#: Human-readable labels and units, used by reports and the dashboard.
RESPONSE_LABELS = {
    "time_to_ari_threshold_s": ("Time to ARI threshold", "s"),
    "final_ari": ("Final ARI", "-"),
    "time_to_half_uptake_s": ("Time to half uptake", "s"),
    "final_uptake_normalized": ("Final M/M_inf", "-"),
    "final_interface_normalized": ("Final interfacial C/C_sat", "-"),
    "diffusivity_m2_s": ("Diffusivity", "m^2/s"),
    "saturation_pct": ("Saturation uptake", "wt%"),
}

AXIS_LABELS = {
    "thickness": ("Film thickness", "um", 1e6),
    "temperature_k": ("Temperature", "degC", None),
    "diffusivity": ("Diffusivity", "m^2/s", 1.0),
}


def axis_display_values(name: str, values: np.ndarray) -> np.ndarray:
    """Convert an axis to the units a reader expects (um, degC, m^2/s)."""
    if name == "temperature_k":
        return np.asarray(values, dtype=float) - KELVIN_OFFSET
    _, _, scale = AXIS_LABELS[name]
    return np.asarray(values, dtype=float) * (scale if scale is not None else 1.0)


@dataclass
class SweepResult:
    """Response surfaces over a design-space grid."""

    base_case: Case
    axis_names: tuple[str, ...]
    axis_values: tuple[np.ndarray, ...]
    responses: dict[str, np.ndarray]
    records: list[dict[str, object]]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(values.size for values in self.axis_values)

    def axis(self, name: str) -> np.ndarray:
        """Return the values of one swept axis."""
        if name not in self.axis_names:
            raise KeyError(f"{name!r} is not a swept axis; swept: {self.axis_names}")
        return self.axis_values[self.axis_names.index(name)]

    def surface(self, name: str) -> np.ndarray:
        """Return one response as an array shaped like the grid."""
        if name not in self.responses:
            raise KeyError(
                f"unknown response {name!r}; available: {sorted(self.responses)}"
            )
        return self.responses[name]

    def finite_fraction(self, name: str) -> float:
        """Fraction of grid points where a response is finite.

        Worth reporting alongside a ``time_to_ari_threshold_s`` surface: if half
        the grid never reaches the threshold, a contour plot of it is mostly
        empty and the reader should be told so rather than left to infer it.
        """
        surface = self.surface(name)
        return float(np.count_nonzero(np.isfinite(surface)) / surface.size)


def run_sweep(case: Case, axes: SweepAxes | dict[str, Iterable[float]],
              progress: Callable[[int, int], None] | None = None) -> SweepResult:
    """Run one simulation per grid point and collect the response surfaces.

    Parameters
    ----------
    case:
        The base case. Axes not swept keep their values from it.
    axes:
        Either a :class:`~hygroadh.config.SweepAxes` or a mapping of axis name
        to values. Axis order in the output follows
        ``("thickness", "temperature_k", "diffusivity")``.
    progress:
        Optional callback receiving ``(completed, total)`` after each point, so
        a long sweep can report progress to a CLI or a dashboard.
    """
    if isinstance(axes, SweepAxes):
        candidates = {
            "thickness": axes.thickness,
            "temperature_k": axes.temperature_k,
            "diffusivity": axes.diffusivity,
        }
    else:
        unknown = set(axes) - set(AXIS_SETTERS)
        if unknown:
            raise PhysicsError(
                f"unknown sweep axis/axes {sorted(unknown)}; "
                f"available: {sorted(AXIS_SETTERS)}"
            )
        candidates = {name: axes.get(name) for name in AXIS_SETTERS}

    names: list[str] = []
    values: list[np.ndarray] = []
    for name in ("thickness", "temperature_k", "diffusivity"):
        raw = candidates.get(name)
        if raw is None:
            continue
        array = np.atleast_1d(np.asarray(list(raw) if not isinstance(raw, np.ndarray)
                                        else raw, dtype=float))
        if array.size == 0:
            continue
        if np.any(array <= 0.0):
            raise PhysicsError(f"sweep axis {name!r} must contain positive values")
        names.append(name)
        values.append(array)

    if not names:
        raise PhysicsError(
            "no sweep axes given; provide at least one of "
            f"{sorted(AXIS_SETTERS)}"
        )

    shape = tuple(array.size for array in values)
    surfaces = {name: np.full(shape, np.nan) for name in RESPONSE_NAMES}
    records: list[dict[str, object]] = []
    total = int(np.prod(shape))

    # Scalar responses only, so the profile is dead weight at every point.
    scalar_case = case.without_profile()
    for count, index in enumerate(itertools.product(*(range(n) for n in shape)), start=1):
        point = scalar_case
        for axis_position, axis_name in enumerate(names):
            point = AXIS_SETTERS[axis_name](
                point, float(values[axis_position][index[axis_position]])
            )
        result = run(point)
        summary = result.summary()
        for name in RESPONSE_NAMES:
            surfaces[name][index] = float(summary[name])
        records.append(_record(summary, names, values, index))
        if progress is not None:
            progress(count, total)

    return SweepResult(
        base_case=case,
        axis_names=tuple(names),
        axis_values=tuple(values),
        responses=surfaces,
        records=records,
    )


def _record(summary: dict[str, object], names: list[str],
            values: list[np.ndarray], index: tuple[int, ...]) -> dict[str, object]:
    """Build one flat table row, with the swept axes named first."""
    row: dict[str, object] = {}
    for axis_position, axis_name in enumerate(names):
        row[axis_name] = float(values[axis_position][index[axis_position]])
    row.update(summary)
    return row


def response_of(result: SimulationResult, name: str) -> float:
    """Extract a single named response from a finished simulation."""
    summary = result.summary()
    if name not in summary:
        raise KeyError(f"unknown response {name!r}")
    return float(summary[name])
