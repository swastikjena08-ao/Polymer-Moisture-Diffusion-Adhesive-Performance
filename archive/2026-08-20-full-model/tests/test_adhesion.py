"""Tests for the Adhesion Retention Index model."""

from __future__ import annotations

import numpy as np
import pytest

from hygroadh.adhesion import (
    AdhesionModel,
    interface_is_displaceable,
    wet_work_of_adhesion,
)
from hygroadh.materials import Polymer
from hygroadh.sorption import PowerLawIsotherm
from hygroadh.units import PhysicsError, to_kelvin

TEMPERATURE = to_kelvin(60.0)
SATURATION = 2.0


def _polymer(**kwargs) -> Polymer:
    defaults = dict(
        isotherm=PowerLawIsotherm(m_ref_pct=SATURATION),
        tg_dry_k=to_kelvin(120.0),
        tg_depression_per_pct=15.0,
    )
    defaults.update(kwargs)
    return Polymer(**defaults)


# --- bounds and limiting cases ---------------------------------------------

def test_a_dry_interface_retains_full_adhesion():
    model = AdhesionModel()
    t = np.linspace(0.0, 1e8, 50)
    result = model.evaluate(t, np.zeros_like(t), TEMPERATURE, SATURATION,
                            _polymer().glass_transition_k)
    assert result.index == pytest.approx(1.0, abs=1e-12)
    assert result.damage == pytest.approx(0.0, abs=1e-15)
    # Every factor, including thermal, must sit exactly at unity when dry.
    assert result.thermal == pytest.approx(1.0, abs=1e-12)
    assert result.plasticization == pytest.approx(1.0, abs=1e-12)


def test_index_stays_within_bounds_across_extreme_parameters():
    t = np.linspace(0.0, 1e9, 200)
    phi = np.clip(np.sin(t / 4e7) ** 2, 0.0, 1.0)
    for gain in (0.0, 1.0, 50.0):
        for rate in (0.0, 1e-9, 1e-3):
            for max_loss in (0.0, 0.5, 1.0):
                model = AdhesionModel(
                    plasticization_gain=gain,
                    plasticization_floor=0.0,
                    hydrolysis_rate_ref=rate,
                    hydrolysis_max_loss=max_loss,
                )
                result = model.evaluate(t, phi, TEMPERATURE, SATURATION,
                                        _polymer().glass_transition_k)
                assert np.all(result.index >= 0.0)
                assert np.all(result.index <= 1.0)
                assert np.all(result.damage >= 0.0)
                assert np.all(result.damage <= 1.0)


def test_negative_interface_values_are_clipped_not_propagated():
    """Round-off in a solver can produce phi slightly below zero; a fractional
    exponent would turn that into NaN rather than a harmless zero."""
    model = AdhesionModel(plasticization_exponent=0.5)
    t = np.array([0.0, 1.0, 2.0])
    result = model.evaluate(t, np.array([-1e-16, 0.0, 0.5]), TEMPERATURE, SATURATION)
    assert np.all(np.isfinite(result.index))
    assert result.index[0] == pytest.approx(1.0)


# --- mechanism 1: reversible plasticization --------------------------------

def test_plasticization_falls_monotonically_with_interfacial_moisture():
    model = AdhesionModel()
    phi = np.linspace(0.0, 1.0, 40)
    factor = model.plasticization_factor(phi)
    assert factor[0] == pytest.approx(1.0)
    assert np.all(np.diff(factor) < 0.0)


def test_plasticization_approaches_its_floor_but_never_crosses_it():
    model = AdhesionModel(plasticization_gain=3.0, plasticization_floor=0.25)
    assert model.plasticization_factor(1e3) == pytest.approx(0.25, abs=1e-9)
    assert np.all(model.plasticization_factor(np.linspace(0, 50, 100)) >= 0.25)


def test_zero_gain_disables_plasticization():
    model = AdhesionModel(plasticization_gain=0.0)
    assert model.plasticization_factor(np.linspace(0, 5, 10)) == pytest.approx(1.0)


# --- mechanism 2: thermal softening near the wet Tg ------------------------

def test_thermal_factor_is_a_softening_transition_around_the_wet_tg():
    model = AdhesionModel(tg_offset=20.0, tg_width=8.0)
    tg = 400.0
    # Far below Tg: fully retained. Half-point sits tg_offset below Tg.
    assert model.thermal_factor(tg - 200.0, tg) == pytest.approx(1.0, abs=1e-8)
    assert model.thermal_factor(tg - 20.0, tg) == pytest.approx(0.5)
    assert model.thermal_factor(tg + 50.0, tg) == pytest.approx(0.0, abs=1e-3)
    temps = np.array([300.0, 350.0, 375.0, 385.0, 400.0])
    values = [model.thermal_factor(t, tg) for t in temps]
    assert np.all(np.diff(values) < 0.0)


def test_thermal_factor_does_not_overflow_at_extreme_gaps():
    """The naive logistic raises a RuntimeWarning past about z = 700."""
    model = AdhesionModel(tg_width=1.0)
    assert model.thermal_factor(1e4, 200.0) == pytest.approx(0.0)
    assert model.thermal_factor(200.0, 1e4) == pytest.approx(1.0)


def test_moisture_lowers_tg_and_so_lowers_the_thermal_factor():
    """The coupling that makes moisture and temperature act together, not apart."""
    model = AdhesionModel()
    polymer = _polymer(tg_dry_k=to_kelvin(90.0), tg_depression_per_pct=18.0)
    service = to_kelvin(70.0)
    dry = model.thermal_factor(service, polymer.glass_transition_k(0.0))
    wet = model.thermal_factor(service, polymer.glass_transition_k(2.0))
    assert dry > wet
    assert wet < 0.5 * dry


def test_omitting_the_tg_model_disables_the_thermal_factor():
    model = AdhesionModel()
    t = np.linspace(0.0, 1e7, 20)
    result = model.evaluate(t, np.ones_like(t), TEMPERATURE, SATURATION, None)
    assert result.thermal == pytest.approx(1.0)
    assert np.all(np.isinf(result.glass_transition_k))


# --- mechanism 3: irreversible hydrolysis ----------------------------------

def test_hydrolysis_damage_matches_the_closed_form_for_steady_wetting():
    """At constant phi the trapezoid integral is exact, so this is machine-precise."""
    model = AdhesionModel(hydrolysis_rate_ref=1e-6, hydrolysis_exponent=1.0)
    t = np.linspace(0.0, 5e6, 7)
    rate = model.hydrolysis_rate(TEMPERATURE)
    damage = model.hydrolysis_damage(t, np.ones_like(t), TEMPERATURE)
    assert damage == pytest.approx(1.0 - np.exp(-rate * t), rel=1e-13)


def test_hydrolysis_damage_is_grid_independent_for_steady_wetting():
    model = AdhesionModel(hydrolysis_rate_ref=1e-6)
    coarse = np.array([0.0, 5e6])
    fine = np.linspace(0.0, 5e6, 5000)
    model_coarse = model.hydrolysis_damage(coarse, np.ones_like(coarse), TEMPERATURE)
    model_fine = model.hydrolysis_damage(fine, np.ones_like(fine), TEMPERATURE)
    assert model_coarse[-1] == pytest.approx(model_fine[-1], rel=1e-12)


def test_hydrolysis_damage_never_decreases_even_when_the_film_dries():
    model = AdhesionModel(hydrolysis_rate_ref=1e-6)
    t = np.linspace(0.0, 2e7, 500)
    phi = np.exp(-((t - 5e6) ** 2) / (2 * 2e6**2))  # a wet excursion, then dry
    damage = model.hydrolysis_damage(t, phi, TEMPERATURE)
    assert np.all(np.diff(damage) >= 0.0)


def test_hydrolysis_saturates_rather_than_overshooting_at_extreme_exposure():
    """Total damage is the ceiling, and it must be approached, never exceeded."""
    model = AdhesionModel(hydrolysis_rate_ref=1e-2)
    t = np.array([0.0, 1e12])
    damage = model.hydrolysis_damage(t, np.ones_like(t), TEMPERATURE)
    assert np.all(damage <= 1.0)
    assert damage[-1] == pytest.approx(1.0, abs=1e-9)
    # And the retention factor bottoms out at exactly the configured max loss.
    assert model.hydrolysis_factor(damage)[-1] == pytest.approx(
        1.0 - model.hydrolysis_max_loss
    )


def test_hydrolysis_rate_is_arrhenius_in_temperature():
    model = AdhesionModel(hydrolysis_activation=60e3, temperature_ref_k=298.15)
    assert model.hydrolysis_rate(298.15) == pytest.approx(model.hydrolysis_rate_ref)
    assert model.hydrolysis_rate(338.15) > 5.0 * model.hydrolysis_rate(298.15)


def test_zero_rate_disables_hydrolysis():
    model = AdhesionModel(hydrolysis_rate_ref=0.0)
    t = np.linspace(0.0, 1e9, 20)
    damage = model.hydrolysis_damage(t, np.ones_like(t), TEMPERATURE)
    assert damage == pytest.approx(0.0)


# --- the defining behaviour: irreversibility -------------------------------

def test_drying_recovers_the_reversible_factors_but_not_the_index():
    """A wet-then-dry cycle must leave permanent damage.

    This is the property that makes ARI history-dependent: plasticization and
    thermal softening recover completely once the interface dries, so anything
    still missing from the index is irreversible hydrolysis. Without it, cyclic
    and steady exposure would be indistinguishable.
    """
    model = AdhesionModel(hydrolysis_rate_ref=5e-7, hydrolysis_max_loss=0.5)
    polymer = _polymer()
    t = np.linspace(0.0, 4e7, 800)
    phi = np.where(t < 2e7, 1.0, 0.0)  # saturated, then bone dry

    result = model.evaluate(t, phi, TEMPERATURE, SATURATION,
                            polymer.glass_transition_k)

    # Both reversible mechanisms return to their dry values, which are 1.
    assert result.plasticization[-1] == pytest.approx(1.0)
    assert result.thermal[-1] == pytest.approx(1.0)
    # The index does not recover, and the entire shortfall is hydrolysis.
    assert result.final_index < 0.99
    assert result.final_index == pytest.approx(result.hydrolysis[-1])
    assert result.hydrolysis[-1] < 1.0


def test_longer_wet_exposure_leaves_more_permanent_damage():
    # Rate chosen so damage stays in a sensitive range: at 5e-7 it saturates
    # within the first leg and every exposure returns the same final index.
    model = AdhesionModel(hydrolysis_rate_ref=2e-8)
    polymer = _polymer()
    finals = []
    for wet_duration in (5e6, 1e7, 2e7):
        t = np.linspace(0.0, 4e7, 800)
        phi = np.where(t < wet_duration, 1.0, 0.0)
        finals.append(
            model.evaluate(t, phi, TEMPERATURE, SATURATION,
                           polymer.glass_transition_k).final_index
        )
    assert np.all(np.diff(finals) < 0.0)


# --- reporting -------------------------------------------------------------

def test_index_falls_monotonically_under_monotone_wetting():
    model = AdhesionModel()
    t = np.linspace(0.0, 5e7, 400)
    phi = 1.0 - np.exp(-t / 5e6)
    result = model.evaluate(t, phi, TEMPERATURE, SATURATION,
                            _polymer().glass_transition_k)
    assert np.all(np.diff(result.index) <= 1e-15)


def test_dominant_mechanism_identifies_the_configured_weak_link():
    t = np.linspace(0.0, 2e7, 300)
    phi = np.ones_like(t)
    polymer = _polymer(tg_dry_k=to_kelvin(150.0))

    plasticization_limited = AdhesionModel(
        plasticization_gain=6.0, plasticization_floor=0.05,
        hydrolysis_rate_ref=1e-12,
    ).evaluate(t, phi, TEMPERATURE, SATURATION, polymer.glass_transition_k)
    assert plasticization_limited.dominant_mechanism == "plasticization"

    hydrolysis_limited = AdhesionModel(
        plasticization_gain=0.01, hydrolysis_rate_ref=1e-5,
        hydrolysis_max_loss=0.95,
    ).evaluate(t, phi, TEMPERATURE, SATURATION, polymer.glass_transition_k)
    assert hydrolysis_limited.dominant_mechanism == "hydrolysis"

    thermal_limited = AdhesionModel(
        plasticization_gain=0.01, hydrolysis_rate_ref=1e-12,
    ).evaluate(t, phi, to_kelvin(115.0), SATURATION,
               _polymer(tg_dry_k=to_kelvin(120.0)).glass_transition_k)
    assert thermal_limited.dominant_mechanism == "thermal"


def test_time_to_index_interpolates_and_reports_unreachable_thresholds():
    model = AdhesionModel()
    t = np.linspace(0.0, 5e7, 2000)
    phi = 1.0 - np.exp(-t / 5e6)
    result = model.evaluate(t, phi, TEMPERATURE, SATURATION,
                            _polymer().glass_transition_k)
    crossing = result.time_to_index(0.8)
    assert 0.0 < crossing < 5e7
    assert np.interp(crossing, t, result.index) == pytest.approx(0.8, abs=1e-3)
    assert result.time_to_index(-1.0) == np.inf


def test_factors_dictionary_exposes_every_mechanism():
    model = AdhesionModel()
    t = np.linspace(0.0, 1e7, 10)
    result = model.evaluate(t, np.ones_like(t), TEMPERATURE, SATURATION)
    assert set(result.factors) == {"plasticization", "thermal", "hydrolysis"}
    product = np.ones_like(t)
    for factor in result.factors.values():
        product = product * factor
    assert result.index == pytest.approx(product)


# --- validation ------------------------------------------------------------

def test_rejects_parameters_that_could_push_the_index_out_of_range():
    with pytest.raises(PhysicsError):
        AdhesionModel(plasticization_floor=1.5)
    with pytest.raises(PhysicsError):
        AdhesionModel(hydrolysis_max_loss=-0.1)
    with pytest.raises(PhysicsError):
        AdhesionModel(plasticization_gain=-1.0)
    with pytest.raises(PhysicsError):
        AdhesionModel(tg_width=0.0)
    with pytest.raises(PhysicsError):
        AdhesionModel(hydrolysis_exponent=0.0)


def test_rejects_mismatched_or_unsorted_histories():
    model = AdhesionModel()
    with pytest.raises(PhysicsError, match="same shape"):
        model.evaluate(np.linspace(0, 1, 5), np.ones(4), TEMPERATURE, SATURATION)
    with pytest.raises(PhysicsError, match="ascending"):
        model.evaluate(np.array([0.0, 2.0, 1.0]), np.ones(3), TEMPERATURE, SATURATION)


# --- thermodynamic screening ----------------------------------------------

def test_wet_work_of_adhesion_flags_a_displaceable_interface():
    # A high-energy substrate that water wets better than the polymer does.
    assert wet_work_of_adhesion(0.005, 0.010, 0.050) < 0.0
    assert interface_is_displaceable(0.005, 0.010, 0.050) is True
    # A well-coupled interface stays thermodynamically stable in water.
    assert wet_work_of_adhesion(0.040, 0.030, 0.010) > 0.0
    assert interface_is_displaceable(0.040, 0.030, 0.010) is False


def test_service_temperature_above_the_dry_tg_is_rejected():
    """There is no dry adhesion baseline to take a fraction of once the resin is
    already rubbery when dry, so the model must refuse rather than normalize by
    a near-zero number."""
    model = AdhesionModel()
    polymer = _polymer(tg_dry_k=to_kelvin(40.0))
    t = np.linspace(0.0, 1e6, 10)
    with pytest.raises(PhysicsError, match="dry glass transition"):
        model.evaluate(t, np.ones_like(t), to_kelvin(150.0), SATURATION,
                       polymer.glass_transition_k)
