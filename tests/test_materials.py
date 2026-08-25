"""Tests for the polymer material model: Arrhenius moisture transport."""

from __future__ import annotations

import numpy as np
import pytest

from hygroadh.diffusion import analytical
from hygroadh.materials import Polymer
from hygroadh.sorption import HenryIsotherm
from hygroadh.units import PhysicsError

T_REF = 298.15


def _polymer(**kwargs) -> Polymer:
    defaults = dict(
        diffusivity_ref=1.0e-13,
        activation_energy=50.0e3,
        temperature_ref_k=T_REF,
        isotherm=HenryIsotherm(m_ref_pct=2.4, temperature_ref_k=T_REF),
    )
    defaults.update(kwargs)
    return Polymer(**defaults)


def test_diffusivity_at_the_reference_temperature_is_the_reference_value():
    assert _polymer().diffusivity(T_REF) == pytest.approx(1.0e-13)


def test_diffusivity_rises_with_temperature():
    polymer = _polymer()
    values = [polymer.diffusivity(t) for t in (283.15, 298.15, 323.15, 348.15)]
    assert np.all(np.diff(values) > 0)


def test_diffusivity_ratio_follows_the_arrhenius_law():
    polymer = _polymer(activation_energy=50e3)
    ratio = polymer.diffusivity(348.15) / polymer.diffusivity(298.15)
    expected = np.exp(-(50e3 / 8.314462618) * (1 / 348.15 - 1 / 298.15))
    assert ratio == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "activation_energy, expected_ratio",
    [(40e3, 10.1), (50e3, 18.1), (60e3, 32.5)],
)
def test_epoxy_barriers_change_d_by_about_an_order_of_magnitude_over_fifty_kelvin(
    activation_energy, expected_ratio
):
    """Sanity anchor against reported epoxy behaviour, not a tautology.

    Water in epoxy has an activation energy for diffusion of roughly
    40-60 kJ/mol. Across that range a 25 -> 75 degC rise speeds diffusion by 10x
    to 30x, which is why temperature dominates how fast a bondline wets.
    """
    polymer = _polymer(activation_energy=activation_energy)
    ratio = polymer.diffusivity(348.15) / polymer.diffusivity(298.15)
    assert ratio == pytest.approx(expected_ratio, rel=0.02)


def test_zero_activation_energy_removes_the_temperature_dependence():
    polymer = _polymer(activation_energy=0.0)
    assert polymer.diffusivity(273.15) == pytest.approx(polymer.diffusivity(373.15))


def test_with_diffusivity_replaces_only_that_parameter():
    polymer = _polymer()
    swapped = polymer.with_diffusivity(9.9e-13)
    assert swapped.diffusivity_ref == 9.9e-13
    assert swapped.activation_energy == polymer.activation_energy
    assert swapped.isotherm is polymer.isotherm
    assert polymer.diffusivity_ref == 1.0e-13, "the original must be untouched"


def test_saturation_delegates_to_the_isotherm():
    polymer = _polymer()
    assert polymer.saturation_pct(T_REF, 0.5) == pytest.approx(
        polymer.isotherm.saturation_pct(T_REF, 0.5)
    )


def test_rejects_invalid_parameters():
    with pytest.raises(PhysicsError):
        _polymer(diffusivity_ref=0.0)
    with pytest.raises(PhysicsError):
        _polymer(diffusivity_ref=-1e-13)
    with pytest.raises(PhysicsError):
        _polymer(temperature_ref_k=25.0)


# --- the temperature checkpoint --------------------------------------------

def test_temperature_accelerates_uptake_only_through_the_arrhenius_factor():
    """Time-temperature superposition.

    Temperature must enter the *normalized* uptake curve through D and nothing
    else. So the curve at one temperature, replotted against time scaled by the
    diffusivity ratio, has to land exactly on the curve at another. This is a
    much sharper statement than "hotter is faster", and it would fail if a
    temperature dependence leaked into the geometry or the normalization.
    """
    polymer = _polymer()
    thickness = 2.0e-4
    cold, hot = 298.15, 348.15
    ratio = polymer.diffusivity(hot) / polymer.diffusivity(cold)

    t_cold = np.geomspace(1e3, 1e8, 60)
    curve_cold = analytical.solve(
        t_cold, thickness, polymer.diffusivity(cold), 2.0, cold, n_depth=None
    )
    curve_hot = analytical.solve(
        t_cold / ratio, thickness, polymer.diffusivity(hot), 2.0, hot, n_depth=None
    )
    assert curve_hot.uptake_normalized == pytest.approx(
        curve_cold.uptake_normalized, abs=1e-12
    )


def test_half_time_shortens_by_exactly_the_diffusivity_ratio():
    polymer = _polymer()
    thickness = 1.5e-4
    cold, hot = 293.15, 333.15
    t_cold = analytical.half_time(thickness, polymer.diffusivity(cold))
    t_hot = analytical.half_time(thickness, polymer.diffusivity(hot))
    expected = polymer.diffusivity(cold) / polymer.diffusivity(hot)
    assert t_hot / t_cold == pytest.approx(expected, rel=1e-12)
    assert t_hot < t_cold


def test_by_default_temperature_changes_the_rate_but_not_the_endpoint():
    """A direct consequence of the basic isotherm, worth stating explicitly."""
    polymer = _polymer()
    assert polymer.diffusivity(333.15) > polymer.diffusivity(T_REF)
    assert polymer.saturation_pct(333.15, 1.0) == pytest.approx(
        polymer.saturation_pct(T_REF, 1.0)
    )


def test_a_sorption_enthalpy_lets_temperature_move_the_endpoint_too():
    polymer = _polymer(
        isotherm=HenryIsotherm(m_ref_pct=2.4, enthalpy_sorption=8e3,
                               temperature_ref_k=T_REF)
    )
    assert polymer.saturation_pct(333.15, 1.0) > polymer.saturation_pct(T_REF, 1.0)
