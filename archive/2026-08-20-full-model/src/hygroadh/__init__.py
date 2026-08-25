"""hygroadh --- diffusion-based moisture uptake and adhesion retention modelling.

Predicts how polymer film thickness, temperature, and moisture diffusivity
influence normalized moisture uptake ``M(t)/M_inf`` and a modelled
Adhesion Retention Index for the film/substrate interface.
"""

from .materials import Polymer
from .sorption import PowerLawIsotherm
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
    "PowerLawIsotherm",
    "HygroadhError",
    "ConfigError",
    "ConvergenceError",
    "MissingDependencyError",
    "PhysicsError",
    "to_kelvin",
    "to_celsius",
    "__version__",
]
