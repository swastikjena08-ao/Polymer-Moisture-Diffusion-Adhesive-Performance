"""Physical constants, unit conversions, and argument validation helpers.

Internal convention: SI throughout, with two deliberate exceptions that
match how the experimental literature reports them --- relative humidity
is a fraction in [0, 1], and moisture content is weight percent.
"""

from __future__ import annotations

import numpy as np

#: Universal gas constant, J/(mol*K).
GAS_CONSTANT = 8.314462618

#: Offset between the Celsius and Kelvin scales.
KELVIN_OFFSET = 273.15

SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0


class HygroadhError(Exception):
    """Base class for every error this package raises deliberately."""


class ConfigError(HygroadhError):
    """A configuration file or parameter set is malformed."""


class PhysicsError(HygroadhError):
    """A parameter is outside its physically meaningful range."""


class ConvergenceError(HygroadhError):
    """An iterative solver failed to reach its tolerance."""


class MissingDependencyError(HygroadhError):
    """An optional dependency is needed for the requested operation."""


def to_kelvin(celsius):
    """Convert degrees Celsius to Kelvin.

    Accepts a scalar or an array and preserves the shape, so a sweep axis can
    be written in the units a user thinks in.
    """
    value = np.asarray(celsius, dtype=float) + KELVIN_OFFSET
    return float(value) if value.ndim == 0 else value


def to_celsius(kelvin):
    """Convert Kelvin to degrees Celsius, scalar or array."""
    value = np.asarray(kelvin, dtype=float) - KELVIN_OFFSET
    return float(value) if value.ndim == 0 else value


def hours(seconds: float) -> float:
    """Convert seconds to hours."""
    return float(seconds) / SECONDS_PER_HOUR


def days(seconds: float) -> float:
    """Convert seconds to days."""
    return float(seconds) / SECONDS_PER_DAY


def from_hours(value: float) -> float:
    """Convert hours to seconds."""
    return float(value) * SECONDS_PER_HOUR


def from_days(value: float) -> float:
    """Convert days to seconds."""
    return float(value) * SECONDS_PER_DAY


def require_positive(value: float, name: str) -> float:
    """Return ``value`` as a float, raising if it is not strictly positive."""
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise PhysicsError(f"{name} must be a finite positive number, got {value!r}")
    return v


def require_non_negative(value: float, name: str) -> float:
    """Return ``value`` as a float, raising if it is negative or not finite."""
    v = float(value)
    if not np.isfinite(v) or v < 0.0:
        raise PhysicsError(f"{name} must be a finite non-negative number, got {value!r}")
    return v


def require_finite(value: float, name: str) -> float:
    """Return ``value`` as a float, raising if it is not finite."""
    v = float(value)
    if not np.isfinite(v):
        raise PhysicsError(f"{name} must be finite, got {value!r}")
    return v


def require_fraction(value: float, name: str) -> float:
    """Return ``value`` as a float, raising unless it lies in [0, 1]."""
    v = float(value)
    if not np.isfinite(v) or v < 0.0 or v > 1.0:
        raise PhysicsError(f"{name} must lie in [0, 1], got {value!r}")
    return v


def require_temperature(value: float, name: str = "temperature_k") -> float:
    """Return an absolute temperature, raising if it is not above 0 K.

    Guards against the common mistake of passing degrees Celsius where
    Kelvin is expected: anything below 100 K is rejected, since no
    polymer moisture problem is posed there.
    """
    v = require_positive(value, name)
    if v < 100.0:
        raise PhysicsError(
            f"{name}={v!r} looks like degrees Celsius; absolute temperature in "
            "Kelvin is required (use units.to_kelvin)"
        )
    return v


def arrhenius(value_ref: float, activation_energy: float, temperature_k: float,
              temperature_ref_k: float) -> float:
    """Scale ``value_ref`` from a reference temperature by an Arrhenius law.

    ``value = value_ref * exp(-(Ea/R) * (1/T - 1/T_ref))``

    The reference form is used rather than a pre-exponential factor
    because it is far better conditioned: ``value_ref`` stays in the
    numerical range of the measurement instead of being extrapolated to
    infinite temperature.

    A positive ``activation_energy`` makes the result increase with
    temperature.
    """
    t = require_temperature(temperature_k)
    t_ref = require_temperature(temperature_ref_k, "temperature_ref_k")
    ea = require_finite(activation_energy, "activation_energy")
    ref = require_finite(value_ref, "value_ref")
    exponent = -(ea / GAS_CONSTANT) * (1.0 / t - 1.0 / t_ref)
    return ref * float(np.exp(exponent))
