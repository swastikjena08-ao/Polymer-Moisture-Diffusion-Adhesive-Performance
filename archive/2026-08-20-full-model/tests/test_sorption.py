"""Tests for the equilibrium sorption isotherm."""

from __future__ import annotations

import numpy as np
import pytest

from hygroadh.sorption import PowerLawIsotherm
from hygroadh.units import PhysicsError

T_REF = 298.15


def test_reference_point_is_reproduced():
    iso = PowerLawIsotherm(m_ref_pct=2.5, rh_ref=1.0, temperature_ref_k=T_REF)
    assert iso.saturation_pct(T_REF, 1.0) == pytest.approx(2.5)


def test_unit_exponent_gives_henry_law_linearity():
    iso = PowerLawIsotherm(m_ref_pct=3.0, exponent=1.0)
    rh = np.linspace(0.0, 1.0, 11)
    values = np.array([iso.saturation_pct(T_REF, r) for r in rh])
    assert values == pytest.approx(3.0 * rh)


def test_sub_unit_exponent_gives_a_concave_isotherm():
    """b < 1 is the concave shape epoxies show: uptake outruns humidity at low RH."""
    iso = PowerLawIsotherm(m_ref_pct=3.0, exponent=0.6)
    assert iso.saturation_pct(T_REF, 0.5) > 0.5 * iso.saturation_pct(T_REF, 1.0)


def test_dry_air_gives_zero_uptake():
    iso = PowerLawIsotherm(m_ref_pct=3.0, exponent=0.6)
    assert iso.saturation_pct(T_REF, 0.0) == 0.0


def test_saturation_increases_monotonically_with_humidity():
    iso = PowerLawIsotherm(m_ref_pct=2.0, exponent=0.8)
    values = [iso.saturation_pct(T_REF, r) for r in np.linspace(0, 1, 25)]
    assert np.all(np.diff(values) > 0)


def test_zero_sorption_enthalpy_removes_temperature_dependence():
    iso = PowerLawIsotherm(m_ref_pct=2.0, enthalpy_sorption=0.0)
    assert iso.saturation_pct(273.15, 0.8) == pytest.approx(
        iso.saturation_pct(353.15, 0.8)
    )


def test_positive_sorption_enthalpy_raises_saturation_with_temperature():
    iso = PowerLawIsotherm(m_ref_pct=2.0, enthalpy_sorption=8e3, temperature_ref_k=T_REF)
    assert iso.saturation_pct(273.15, 1.0) < iso.saturation_pct(T_REF, 1.0)
    assert iso.saturation_pct(T_REF, 1.0) < iso.saturation_pct(353.15, 1.0)


def test_inverse_round_trips_across_humidity_and_temperature():
    iso = PowerLawIsotherm(m_ref_pct=2.2, rh_ref=0.85, exponent=0.7,
                           enthalpy_sorption=6e3, temperature_ref_k=T_REF)
    for temperature in (283.15, T_REF, 333.15):
        for rh in (0.05, 0.3, 0.62, 1.0):
            uptake = iso.saturation_pct(temperature, rh)
            assert iso.humidity_for(uptake, temperature) == pytest.approx(rh, rel=1e-12)


def test_inverse_maps_zero_uptake_to_dry_air():
    iso = PowerLawIsotherm(m_ref_pct=2.0)
    assert iso.humidity_for(0.0, T_REF) == 0.0


def test_inverse_reports_uptake_beyond_the_saturated_ceiling():
    iso = PowerLawIsotherm(m_ref_pct=2.0)
    with pytest.raises(ValueError, match="unreachable"):
        iso.humidity_for(5.0, T_REF)
    with pytest.raises(ValueError):
        iso.humidity_for(-0.1, T_REF)


def test_rejects_invalid_parameters():
    with pytest.raises(PhysicsError):
        PowerLawIsotherm(m_ref_pct=0.0)
    with pytest.raises(PhysicsError):
        PowerLawIsotherm(m_ref_pct=2.0, exponent=0.0)
    with pytest.raises(PhysicsError):
        PowerLawIsotherm(m_ref_pct=2.0, rh_ref=1.5)
    with pytest.raises(ValueError):
        PowerLawIsotherm(m_ref_pct=2.0, rh_ref=0.0)
    with pytest.raises(PhysicsError):
        PowerLawIsotherm(m_ref_pct=2.0).saturation_pct(T_REF, 1.2)
