"""Moisture diffusion solvers.

Three solvers share the :class:`~hygroadh.diffusion.base.DiffusionResult`
interface, in increasing generality and cost:

``analytical``
    Exact plane-sheet series. Constant diffusivity, constant surface
    condition. The verification reference for the other two.
``fd``
    Theta-method finite differences. Adds concentration-dependent
    diffusivity, humidity schedules, and surface resistance.
``langmuir``
    Dual-stage Carter-Kibler transport for non-Fickian, two-stage uptake.
"""

from .base import (
    DiffusionResult,
    Exposure,
    diffusion_length,
    equivalent_sheet_thickness,
)

__all__ = [
    "DiffusionResult",
    "Exposure",
    "diffusion_length",
    "equivalent_sheet_thickness",
]
