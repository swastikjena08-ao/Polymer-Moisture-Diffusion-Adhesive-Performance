"""Tests for the simplified Adhesion Retention Index."""

from __future__ import annotations

import numpy as np
import pytest

from hygroadh.adhesion import AdhesionModel
from hygroadh.units import PhysicsError, to_kelvin

TEMPERATURE = to_kelvin(60.0)
SATURATION = 2.0


# --- the knockdown law -----------------------------------------------------

def test_a_dry_bondline_retains_full_adhesion():
    assert AdhesionModel().retention(0.0) == pytest.approx(1.0)


def test_retention_falls_linearly_with_bondline_water():
    model = AdhesionModel(knockdown_per_pct=0.25)
    assert model.retention([0.0, 1.0, 2.0, 3.0]) == pytest.approx([1.0, 0.75, 0.5, 0.25])


def test_retention_is_clipped_at_zero_rather_than_going_negative():
    model = AdhesionModel(knockdown_per_pct=0.25)
    assert model.retention(10.0) == pytest.approx(0.0)
    assert np.all(model.retention(np.linspace(0, 50, 100)) >= 0.0)


def test_retention_never_exceeds_one_even_with_round_off_below_zero():
    """A solver can leave a concentration a hair below zero; that must not
    report better-than-dry adhesion."""
    model = AdhesionModel()
    assert model.retention(-1e-16) == pytest.approx(1.0)
    assert np.all(model.retention(np.array([-1e-12, 0.0, 1.0])) <= 1.0)


def test_zero_knockdown_makes_adhesion_insensitive_to_moisture():
    model = AdhesionModel(knockdown_per_pct=0.0)
    assert model.retention(np.linspace(0, 20, 50)) == pytest.approx(1.0)


def test_retention_is_monotone_decreasing():
    water = np.linspace(0.0, 3.9, 60)
    assert np.all(np.diff(AdhesionModel(knockdown_per_pct=0.25).retention(water)) < 0.0)


# --- the inverse: a threshold is a water budget -----------------------------

def test_moisture_at_index_inverts_the_law_exactly():
    model = AdhesionModel(knockdown_per_pct=0.25)
    for index in (0.0, 0.25, 0.5, 0.8, 1.0):
        water = model.moisture_at_index(index)
        assert model.retention(water) == pytest.approx(index)


def test_an_ari_threshold_reads_as_a_bondline_water_budget():
    """The interpretation worth having: ARI 0.8 at k=0.25 is 0.8 wt% of water."""
    assert AdhesionModel(knockdown_per_pct=0.25).moisture_at_index(0.8) == pytest.approx(0.8)
    assert AdhesionModel(knockdown_per_pct=0.5).moisture_at_index(0.8) == pytest.approx(0.4)


def test_inverse_rejects_an_out_of_range_index_and_a_flat_law():
    model = AdhesionModel(knockdown_per_pct=0.25)
    with pytest.raises(PhysicsError, match=r"\[0, 1\]"):
        model.moisture_at_index(1.5)
    with pytest.raises(PhysicsError, match="no inverse"):
        AdhesionModel(knockdown_per_pct=0.0).moisture_at_index(0.8)


# --- evaluating along a history --------------------------------------------

def test_evaluate_converts_normalized_moisture_to_absolute_water():
    model = AdhesionModel(knockdown_per_pct=0.25)
    t = np.array([0.0, 1.0, 2.0])
    phi = np.array([0.0, 0.5, 1.0])
    result = model.evaluate(t, phi, TEMPERATURE, SATURATION)
    assert result.interface_pct == pytest.approx([0.0, 1.0, 2.0])
    assert result.index == pytest.approx([1.0, 0.75, 0.5])


def test_a_drier_exposure_is_genuinely_less_damaging():
    """Absolute water, not normalized moisture, is what drives the law.

    Both histories saturate --- phi reaches 1 --- but the low-humidity film holds a
    quarter of the water, so it must keep far more of its adhesion. Driving from
    the normalized field would report the two as identical.
    """
    model = AdhesionModel(knockdown_per_pct=0.25)
    t = np.linspace(0.0, 10.0, 20)
    phi = np.ones_like(t)
    wet = model.evaluate(t, phi, TEMPERATURE, 2.0).final_index
    dry = model.evaluate(t, phi, TEMPERATURE, 0.5).final_index
    assert dry > wet
    assert wet == pytest.approx(0.5)
    assert dry == pytest.approx(0.875)


def test_the_index_has_no_memory():
    """The defining limitation of this model, asserted so nobody assumes otherwise.

    Adhesion depends only on how wet the bondline is at that instant. A history
    that wets and then dries ends at exactly 1, however long it spent wet. Real
    hygrothermal ageing is partly irreversible; that is not modelled here.
    """
    model = AdhesionModel(knockdown_per_pct=0.4)
    t = np.linspace(0.0, 100.0, 201)
    brief = np.where((t > 10) & (t < 20), 1.0, 0.0)
    prolonged = np.where(t < 90, 1.0, 0.0)
    brief_final = model.evaluate(t, brief, TEMPERATURE, SATURATION).final_index
    long_final = model.evaluate(t, prolonged, TEMPERATURE, SATURATION).final_index
    assert brief_final == pytest.approx(1.0)
    assert long_final == pytest.approx(1.0)
    assert brief_final == pytest.approx(long_final)


def test_the_index_tracks_moisture_up_and_down():
    model = AdhesionModel(knockdown_per_pct=0.3)
    t = np.linspace(0.0, 20.0, 41)
    phi = np.sin(np.pi * t / 20.0) ** 2
    result = model.evaluate(t, phi, TEMPERATURE, SATURATION)
    # The index is a strictly decreasing function of moisture, so their orderings
    # must be exact mirrors of each other.
    assert np.argmax(result.interface_pct) == np.argmin(result.index)
    assert np.all(np.diff(result.index[np.argsort(result.interface_pct)]) <= 1e-12)


def test_index_stays_within_bounds_over_extreme_inputs():
    t = np.linspace(0.0, 100.0, 200)
    phi = np.clip(np.sin(t / 7.0) ** 2 * 1.4, 0.0, None)
    for knockdown in (0.0, 0.25, 5.0):
        for saturation in (0.0, 0.5, 12.0):
            result = AdhesionModel(knockdown_per_pct=knockdown).evaluate(
                t, phi, TEMPERATURE, saturation
            )
            assert np.all(result.index >= 0.0)
            assert np.all(result.index <= 1.0)


def test_temperature_is_recorded_but_does_not_enter_the_law():
    """Temperature reaches ARI only through the transport that made the history."""
    model = AdhesionModel()
    t = np.array([0.0, 1.0])
    phi = np.array([0.0, 1.0])
    cold = model.evaluate(t, phi, to_kelvin(10.0), SATURATION)
    hot = model.evaluate(t, phi, to_kelvin(90.0), SATURATION)
    assert cold.index == pytest.approx(hot.index)
    assert cold.temperature_k == pytest.approx(to_kelvin(10.0))
    assert hot.temperature_k == pytest.approx(to_kelvin(90.0))


def test_time_to_index_interpolates_and_reports_unreachable_thresholds():
    model = AdhesionModel(knockdown_per_pct=0.25)
    t = np.linspace(0.0, 100.0, 1001)
    phi = 1.0 - np.exp(-t / 10.0)
    result = model.evaluate(t, phi, TEMPERATURE, SATURATION)
    crossing = result.time_to_index(0.75)
    assert 0.0 < crossing < 100.0
    assert np.interp(crossing, t, result.index) == pytest.approx(0.75, abs=1e-3)
    # Saturation here is 2 wt%, so the index bottoms out at 0.5 and 0.4 is
    # unreachable however long the exposure runs.
    assert result.time_to_index(0.4) == np.inf


def test_evaluate_rejects_mismatched_or_unsorted_histories():
    model = AdhesionModel()
    with pytest.raises(PhysicsError, match="same shape"):
        model.evaluate(np.linspace(0, 1, 5), np.ones(4), TEMPERATURE, SATURATION)
    with pytest.raises(PhysicsError, match="ascending"):
        model.evaluate(np.array([0.0, 2.0, 1.0]), np.ones(3), TEMPERATURE, SATURATION)
    with pytest.raises(PhysicsError):
        model.evaluate(np.array([0.0, 1.0]), np.ones(2), 25.0, SATURATION)
    with pytest.raises(PhysicsError):
        model.evaluate(np.array([0.0, 1.0]), np.ones(2), TEMPERATURE, -1.0)


def test_rejects_a_negative_knockdown():
    with pytest.raises(PhysicsError):
        AdhesionModel(knockdown_per_pct=-0.1)
