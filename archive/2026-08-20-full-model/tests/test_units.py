"""Tests for constants, conversions, and the validation helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hygroadh import units
from hygroadh.units import PhysicsError


def test_temperature_conversions_round_trip():
    for celsius in (-40.0, 0.0, 25.0, 85.0, 150.0):
        assert units.to_celsius(units.to_kelvin(celsius)) == pytest.approx(celsius)
    assert units.to_kelvin(0.0) == pytest.approx(273.15)


def test_time_conversions_round_trip():
    assert units.hours(units.from_hours(7.5)) == pytest.approx(7.5)
    assert units.days(units.from_days(30.0)) == pytest.approx(30.0)
    assert units.from_days(1.0) == pytest.approx(24 * units.from_hours(1.0))


def test_arrhenius_returns_the_reference_value_at_the_reference_temperature():
    assert units.arrhenius(5e-13, 50e3, 298.15, 298.15) == pytest.approx(5e-13)


def test_arrhenius_increases_with_temperature_for_positive_activation_energy():
    cold = units.arrhenius(5e-13, 50e3, 298.15, 323.15)
    hot = units.arrhenius(5e-13, 50e3, 348.15, 323.15)
    assert cold < 5e-13 < hot


def test_arrhenius_matches_the_closed_form():
    ea, t, t_ref = 45e3, 333.15, 298.15
    expected = 2.0e-12 * math.exp(-(ea / units.GAS_CONSTANT) * (1 / t - 1 / t_ref))
    assert units.arrhenius(2.0e-12, ea, t, t_ref) == pytest.approx(expected, rel=1e-14)


def test_zero_activation_energy_removes_the_temperature_dependence():
    assert units.arrhenius(1.0, 0.0, 400.0, 250.0) == pytest.approx(1.0)


def test_absolute_temperature_guard_catches_celsius_mistakes():
    """Passing 25 instead of 298.15 is the easiest way to get silently wrong physics."""
    with pytest.raises(PhysicsError, match="Celsius"):
        units.require_temperature(25.0)
    assert units.require_temperature(298.15) == pytest.approx(298.15)


def test_validators_reject_out_of_range_and_non_finite_values():
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(PhysicsError):
            units.require_positive(bad, "x")
    with pytest.raises(PhysicsError):
        units.require_non_negative(-1e-30, "x")
    assert units.require_non_negative(0.0, "x") == 0.0
    for bad in (-0.01, 1.01, math.nan):
        with pytest.raises(PhysicsError):
            units.require_fraction(bad, "rh")
    assert units.require_fraction(1.0, "rh") == 1.0
    with pytest.raises(PhysicsError):
        units.require_finite(np.inf, "x")
