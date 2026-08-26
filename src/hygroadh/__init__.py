"""hygroadh --- diffusion-based moisture uptake and adhesion retention modelling.

A computational study of moisture transport in polymer adhesive films: how film
thickness, temperature, and water diffusivity influence normalized moisture
uptake ``M(t)/M_inf`` and the time for moisture to reach a chosen threshold at
the far face of the film.
"""

from .materials import Polymer
from .sorption import HenryIsotherm
from .threshold import MoistureCriterion
from .units import (
    ConfigError,
    ConvergenceError,
    HygroadhError,
    MissingDependencyError,
    PhysicsError,
    to_celsius,
    to_kelvin,
)

__version__ = "0.1.0"

__all__ = [
    "Polymer",
    "HenryIsotherm",
    "MoistureCriterion",
    "HygroadhError",
    "ConfigError",
    "ConvergenceError",
    "MissingDependencyError",
    "PhysicsError",
    "to_kelvin",
    "to_celsius",
    "__version__",
]
