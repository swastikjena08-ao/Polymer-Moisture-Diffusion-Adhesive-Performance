"""Tests for the Henry's-law sorption isotherm."""

from __future__ import annotations

import numpy as np
import pytest

from hygroadh.sorption import HenryIsotherm
from hygroadh.units import PhysicsError

T_REF = 298.15


def test_reference_point_is_reproduced():
    iso = HenryIsotherm(m_ref_pct=2.5, temperature_ref_k=T_REF)
    assert iso.saturation_pct(T_REF, 1.0) == pytest.approx(2.5)


def test_saturation_is_linear_in_humidity():
    """The defining property: doubling the humidity doubles the equilibrium uptake."""
    iso = HenryIsotherm(m_ref_pct=3.0)
    rh = np.linspace(0.0, 1.0, 11)
    values = np.array([iso.saturation_pct(T_REF, r) for r in rh])
    assert values == pytest.approx(3.0 * rh)
    assert iso.saturation_pct(T_REF, 0.5) == pytest.approx(
        0.5 * iso.saturation_pct(T_REF, 1.0)
    )


def test_dry_air_gives_zero_uptake():
    assert HenryIsotherm(m_ref_pct=3.0).saturation_pct(T_REF, 0.0) == 0.0


def test_saturation_increases_monotonically_with_humidity():
    iso = HenryIsotherm(m_ref_pct=2.0)
    values = [iso.saturation_pct(T_REF, r) for r in np.linspace(0, 1, 25)]
    assert np.all(np.diff(values) > 0)


def test_default_isotherm_is_temperature_independent():
    """With no sorption enthalpy, temperature changes the rate but not the endpoint."""
    iso = HenryIsotherm(m_ref_pct=2.0)
    assert iso.saturation_pct(273.15, 0.8) == pytest.approx(
        iso.saturation_pct(353.15, 0.8)
    )


def test_positive_sorption_enthalpy_raises_saturation_with_temperature():
    iso = HenryIsotherm(m_ref_pct=2.0, enthalpy_sorption=8e3, temperature_ref_k=T_REF)
    assert iso.saturation_pct(273.15, 1.0) < iso.saturation_pct(T_REF, 1.0)
    assert iso.saturation_pct(T_REF, 1.0) < iso.saturation_pct(353.15, 1.0)
    assert iso.saturation_pct(T_REF, 1.0) == pytest.approx(2.0)


def test_negative_sorption_enthalpy_lowers_saturation_with_temperature():
    iso = HenryIsotherm(m_ref_pct=2.0, enthalpy_sorption=-8e3, temperature_ref_k=T_REF)
    assert iso.saturation_pct(353.15, 1.0) < iso.saturation_pct(273.15, 1.0)


def test_inverse_round_trips_across_humidity_and_temperature():
    iso = HenryIsotherm(m_ref_pct=2.2, enthalpy_sorption=6e3, temperature_ref_k=T_REF)
    for temperature in (283.15, T_REF, 333.15):
        for rh in (0.05, 0.3, 0.62, 1.0):
            uptake = iso.saturation_pct(temperature, rh)
            assert iso.humidity_for(uptake, temperature) == pytest.approx(rh, rel=1e-12)


def test_inverse_maps_zero_uptake_to_dry_air():
    assert HenryIsotherm(m_ref_pct=2.0).humidity_for(0.0, T_REF) == 0.0


def test_inverse_reports_uptake_beyond_the_saturated_ceiling():
    iso = HenryIsotherm(m_ref_pct=2.0)
    with pytest.raises(ValueError, match="unreachable"):
        iso.humidity_for(5.0, T_REF)
    with pytest.raises(ValueError):
        iso.humidity_for(-0.1, T_REF)


def test_rejects_invalid_parameters_and_humidities():
    with pytest.raises(PhysicsError):
        HenryIsotherm(m_ref_pct=0.0)
    with pytest.raises(PhysicsError):
        HenryIsotherm(m_ref_pct=-1.0)
    with pytest.raises(PhysicsError):
        HenryIsotherm(m_ref_pct=2.0).saturation_pct(T_REF, 1.2)
    with pytest.raises(PhysicsError):
        HenryIsotherm(m_ref_pct=2.0).saturation_pct(T_REF, -0.1)


def test_celsius_mistake_is_caught_even_on_the_zero_humidity_path():
    """The guard must fire before the early return, or a bad temperature slips by."""
    iso = HenryIsotherm(m_ref_pct=2.0)
    with pytest.raises(PhysicsError, match="Celsius"):
        iso.saturation_pct(25.0, 0.0)
    with pytest.raises(PhysicsError, match="Celsius"):
        iso.saturation_pct(25.0, 0.5)
