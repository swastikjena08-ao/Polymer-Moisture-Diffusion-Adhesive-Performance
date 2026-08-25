"""Tests for the moisture-penetration threshold criterion."""

from __future__ import annotations

import numpy as np
import pytest

from hygroadh.threshold import BASES, MoistureCriterion
from hygroadh.units import PhysicsError

SATURATION = 2.0


# --- construction ---------------------------------------------------------

def test_defaults_are_a_half_of_equilibrium():
    criterion = MoistureCriterion()
    assert criterion.value == pytest.approx(0.5)
    assert criterion.basis == "normalized"


def test_rejects_an_unknown_basis_and_a_non_positive_value():
    with pytest.raises(PhysicsError, match="basis must be"):
        MoistureCriterion(0.5, "percent")
    for bad in (0.0, -0.1):
        with pytest.raises(PhysicsError):
            MoistureCriterion(bad)


def test_a_normalized_threshold_of_one_is_rejected():
    """The far face approaches equilibrium asymptotically and never arrives."""
    with pytest.raises(PhysicsError, match="never crossed"):
        MoistureCriterion(1.0, "normalized")
    with pytest.raises(PhysicsError):
        MoistureCriterion(1.5, "normalized")
    # On the absolute basis the same number is perfectly legitimate.
    assert MoistureCriterion(1.5, "wt_pct").value == pytest.approx(1.5)


def test_both_bases_are_advertised():
    assert set(BASES) == {"normalized", "wt_pct"}


# --- converting between the two bases -------------------------------------

def test_a_normalized_threshold_ignores_saturation():
    criterion = MoistureCriterion(0.4, "normalized")
    assert criterion.normalized_level(2.0) == pytest.approx(0.4)
    assert criterion.normalized_level(0.5) == pytest.approx(0.4)
    assert criterion.moisture_pct(2.0) == pytest.approx(0.8)


def test_an_absolute_threshold_scales_against_saturation():
    criterion = MoistureCriterion(1.0, "wt_pct")
    assert criterion.normalized_level(2.0) == pytest.approx(0.5)
    assert criterion.normalized_level(4.0) == pytest.approx(0.25)
    assert criterion.moisture_pct(2.0) == pytest.approx(1.0)


def test_the_two_bases_agree_when_set_to_the_same_physical_level():
    fraction = MoistureCriterion(0.5, "normalized")
    absolute = MoistureCriterion(0.5 * SATURATION, "wt_pct")
    assert fraction.normalized_level(SATURATION) == pytest.approx(
        absolute.normalized_level(SATURATION)
    )


# --- reachability ---------------------------------------------------------

def test_a_normalized_threshold_is_always_reachable():
    for value in (0.01, 0.5, 0.99):
        criterion = MoistureCriterion(value, "normalized")
        assert criterion.is_reachable(SATURATION)
        assert criterion.unreachable_reason(SATURATION) is None


def test_an_absolute_threshold_above_the_ceiling_is_unreachable():
    criterion = MoistureCriterion(3.0, "wt_pct")
    assert not criterion.is_reachable(2.0)
    reason = criterion.unreachable_reason(2.0)
    assert reason is not None
    assert "2 wt%" in reason and "can never" in reason
    # And it becomes reachable once the film can hold that much.
    assert criterion.is_reachable(4.0)


def test_zero_saturation_makes_any_absolute_threshold_unreachable():
    criterion = MoistureCriterion(0.1, "wt_pct")
    assert not criterion.is_reachable(0.0)
    assert np.isinf(criterion.normalized_level(0.0))


# --- evaluating against a history -----------------------------------------

def _history(n=2001, tau=10.0, span=100.0):
    time = np.linspace(0.0, span, n)
    return time, 1.0 - np.exp(-time / tau)


def test_time_to_threshold_interpolates_the_crossing():
    time, far_face = _history()
    criterion = MoistureCriterion(0.5, "normalized")
    crossing = criterion.time_to_threshold(time, far_face, SATURATION)
    # 1 - exp(-t/tau) = 0.5 at t = tau ln 2.
    assert crossing == pytest.approx(10.0 * np.log(2.0), rel=1e-3)


def test_a_higher_threshold_takes_longer():
    time, far_face = _history()
    times = [
        MoistureCriterion(v, "normalized").time_to_threshold(time, far_face, SATURATION)
        for v in (0.2, 0.4, 0.6, 0.8)
    ]
    assert np.all(np.diff(times) > 0.0)


def test_an_unreached_threshold_is_infinite_not_an_error():
    """A window too short to cross the threshold is a legitimate answer."""
    time, far_face = _history(span=1.0, tau=10.0)
    criterion = MoistureCriterion(0.9, "normalized")
    assert criterion.time_to_threshold(time, far_face, SATURATION) == np.inf


def test_an_unreachable_absolute_threshold_is_infinite_without_scanning():
    time, far_face = _history()
    criterion = MoistureCriterion(5.0, "wt_pct")
    assert criterion.time_to_threshold(time, far_face, SATURATION) == np.inf


def test_the_absolute_basis_responds_to_saturation():
    """The same criterion is met sooner in a wetter film."""
    time, far_face = _history()
    criterion = MoistureCriterion(1.0, "wt_pct")
    wet = criterion.time_to_threshold(time, far_face, 4.0)
    dry = criterion.time_to_threshold(time, far_face, 2.0)
    assert wet < dry


# --- description ----------------------------------------------------------

def test_descriptions_state_the_criterion_in_both_unit_systems():
    fraction = MoistureCriterion(0.5, "normalized").describe(SATURATION)
    assert "50%" in fraction and "1 wt%" in fraction
    absolute = MoistureCriterion(1.0, "wt_pct").describe(SATURATION)
    assert "1 wt%" in absolute and "50%" in absolute


def test_descriptions_work_without_a_saturation_value():
    assert "50%" in MoistureCriterion(0.5, "normalized").describe()
    assert "1.5 wt%" in MoistureCriterion(1.5, "wt_pct").describe()
