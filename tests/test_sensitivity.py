"""Tests for local elasticities and Morris screening."""

from __future__ import annotations

import numpy as np
import pytest

from hygroadh import sensitivity
from hygroadh.threshold import MoistureCriterion
from hygroadh.materials import Polymer
from hygroadh.simulate import Case, ExposureCondition
from hygroadh.sorption import HenryIsotherm
from hygroadh.units import PhysicsError, from_days, to_kelvin


def _case(**kwargs) -> Case:
    polymer = kwargs.pop("polymer", None) or Polymer(
        diffusivity_ref=2.0e-13,
        isotherm=HenryIsotherm(m_ref_pct=2.4),
    )
    condition = kwargs.pop("condition", None) or ExposureCondition(
        temperature_k=to_kelvin(60.0), duration=from_days(60),
        relative_humidity=0.85, n_times=2000,
    )
    defaults = dict(thickness=200e-6, polymer=polymer, condition=condition)
    defaults.update(kwargs)
    return Case(**defaults)


# --- the headline verification ---------------------------------------------

def test_elasticities_recover_the_analytical_scaling_laws():
    """Independent verification of the whole chain.

    In the diffusion-controlled regime the interface wets on a timescale
    ``L**2/D``, so the time for ARI to cross a threshold must scale as ``L**2``
    and as ``1/D``. Those exponents are +2 and -1, and they emerge here from a
    numerical derivative of the full coupled model --- transport, isotherm, Tg,
    and damage integration included. Getting them right is strong evidence the
    pieces are wired together correctly.
    """
    case = _case()
    result = sensitivity.local_elasticities(case, "time_to_threshold")
    assert result.elasticities["thickness"] == pytest.approx(2.0, abs=0.02)
    assert result.elasticities["diffusivity"] == pytest.approx(-1.0, abs=0.02)
    # Temperature acts through Arrhenius D, so its log-log slope is large.
    assert result.elasticities["temperature_k"] < -5.0


def test_thickness_and_diffusivity_have_equal_and_opposite_influence():
    """Both act only through Fo = D t / l**2, so their exponents must be 2 and -1."""
    result = sensitivity.local_elasticities(
        _case(), "time_to_threshold"
    )
    assert result.elasticities["thickness"] == pytest.approx(
        -2.0 * result.elasticities["diffusivity"], rel=0.02
    )


# --- the saturation degeneracy, documented on purpose ----------------------

def test_end_of_run_responses_are_blind_to_every_factor_once_saturated():
    """A real degeneracy, pinned so nobody 'fixes' it into a wrong answer.

    Given long enough, every film reaches M/M_inf = 1 whatever its thickness,
    temperature, or diffusivity. All three set how *fast* moisture arrives, not
    how much arrives eventually, so any end-of-run response is insensitive to all
    of them. This is why time_to_threshold is the default response.
    """
    result = sensitivity.local_elasticities(_case(), "final_interface_normalized")
    for factor in sensitivity.FACTORS:
        assert result.elasticities[factor] == pytest.approx(0.0, abs=1e-9)


def test_default_response_is_the_one_sensitive_to_all_three_factors():
    result = sensitivity.local_elasticities(_case())
    assert result.response == "time_to_threshold"
    assert all(abs(value) > 1e-6 for value in result.elasticities.values())


# --- reporting -------------------------------------------------------------

def test_ranking_orders_by_absolute_influence():
    result = sensitivity.local_elasticities(_case(), "time_to_threshold")
    magnitudes = [abs(value) for _, value in result.ranking]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert result.most_influential == result.ranking[0][0]


def test_elasticities_are_insensitive_to_the_step_size():
    """A well-converged derivative must not depend on how it was taken."""
    case = _case()
    coarse = sensitivity.local_elasticities(case, relative_step=0.05)
    fine = sensitivity.local_elasticities(case, relative_step=0.01)
    for factor in sensitivity.FACTORS:
        assert coarse.elasticities[factor] == pytest.approx(
            fine.elasticities[factor], rel=0.05
        )


# --- validation ------------------------------------------------------------

def test_unreachable_response_is_reported_with_a_usable_message():
    case = _case(
        condition=ExposureCondition(temperature_k=to_kelvin(5.0), duration=from_days(1),
                                    relative_humidity=0.3, n_times=100),
        criterion=MoistureCriterion(2.0, "wt_pct"),
    )
    with pytest.raises(PhysicsError, match="never reached"):
        sensitivity.local_elasticities(case, "time_to_threshold")


def test_rejects_unknown_responses_factors_and_steps():
    case = _case()
    with pytest.raises(PhysicsError, match="unknown response"):
        sensitivity.local_elasticities(case, "vibes")
    with pytest.raises(PhysicsError, match="unknown factor"):
        sensitivity.local_elasticities(case, factors=("humidity",))
    with pytest.raises(PhysicsError):
        sensitivity.local_elasticities(case, relative_step=0.0)
    with pytest.raises(PhysicsError, match="well below"):
        sensitivity.local_elasticities(case, relative_step=0.9)


# --- default ranges --------------------------------------------------------

def test_default_ranges_are_geometric_for_scale_factors_and_additive_for_temperature():
    case = _case()
    ranges = sensitivity.default_ranges(case, factor_span=3.0, temperature_span=25.0)
    low, high = ranges["thickness"]
    assert low == pytest.approx(case.thickness / 3.0)
    assert high == pytest.approx(case.thickness * 3.0)
    low, high = ranges["diffusivity"]
    assert low == pytest.approx(case.polymer.diffusivity_ref / 3.0)
    assert high == pytest.approx(case.polymer.diffusivity_ref * 3.0)
    low, high = ranges["temperature_k"]
    assert low == pytest.approx(case.condition.temperature_k - 25.0)


def test_default_temperature_range_stays_physically_meaningful():
    """The lower bound is clamped: nothing useful lives below 200 K."""
    case = _case()
    ranges = sensitivity.default_ranges(case, temperature_span=500.0)
    assert ranges["temperature_k"][0] == pytest.approx(200.0)
    assert ranges["temperature_k"][1] > case.condition.temperature_k


def test_default_ranges_reject_a_non_positive_span():
    case = _case()
    with pytest.raises(PhysicsError):
        sensitivity.default_ranges(case, factor_span=0.0)
    with pytest.raises(PhysicsError):
        sensitivity.default_ranges(case, temperature_span=0.0)


# --- Morris screening ------------------------------------------------------

def test_morris_reports_every_factor_and_the_expected_evaluation_count():
    result = sensitivity.morris_screening(_case(), n_trajectories=5)
    assert set(result.factors) == set(sensitivity.FACTORS)
    assert set(result.mu_star) == set(sensitivity.FACTORS)
    assert set(result.sigma) == set(sensitivity.FACTORS)
    # One base evaluation plus one per factor, per trajectory.
    assert result.n_evaluations == 5 * (len(sensitivity.FACTORS) + 1)


def test_morris_mean_absolute_effect_dominates_the_signed_mean():
    """mu_star >= |mean| always, since averaging magnitudes cannot cancel."""
    result = sensitivity.morris_screening(_case(), n_trajectories=6)
    for factor in result.factors:
        assert result.mu_star[factor] >= abs(result.mean[factor]) - 1e-12


def test_morris_is_reproducible_for_a_fixed_seed_and_varies_otherwise():
    first = sensitivity.morris_screening(_case(), n_trajectories=4, seed=7)
    again = sensitivity.morris_screening(_case(), n_trajectories=4, seed=7)
    other = sensitivity.morris_screening(_case(), n_trajectories=4, seed=8)
    assert first.mu_star == pytest.approx(again.mu_star)
    assert first.mu_star != pytest.approx(other.mu_star)


def test_morris_finds_thickness_influential_for_the_threshold_time():
    result = sensitivity.morris_screening(_case(), n_trajectories=8)
    assert result.mu_star["thickness"] > 0.0
    assert result.most_influential in sensitivity.FACTORS
    assert result.ranking[0][1] >= result.ranking[-1][1]


def test_morris_ignores_a_factor_it_is_not_given_a_range_for():
    case = _case()
    ranges = {"thickness": (5e-5, 5e-4)}
    result = sensitivity.morris_screening(case, ranges, n_trajectories=3)
    assert result.factors == ("thickness",)
    assert result.n_evaluations == 3 * 2


def test_morris_flags_a_factor_whose_effect_depends_on_where_you_are():
    """Interaction detection: sigma large relative to mu_star.

    Thickness and diffusivity enter only as L**2/D, so the effect of one depends
    strongly on the other; the screening should say so rather than implying a
    single elasticity describes them everywhere.
    """
    result = sensitivity.morris_screening(_case(), n_trajectories=8)
    assert result.interacts("thickness")
    assert isinstance(result.interacts("temperature_k"), bool)


def test_morris_rejects_bad_ranges_and_settings():
    case = _case()
    with pytest.raises(PhysicsError, match="at least one"):
        sensitivity.morris_screening(case, {})
    with pytest.raises(PhysicsError, match="increasing"):
        sensitivity.morris_screening(case, {"thickness": (4e-4, 1e-4)})
    with pytest.raises(PhysicsError, match="increasing"):
        sensitivity.morris_screening(case, {"thickness": (0.0, 1e-4)})
    with pytest.raises(PhysicsError):
        sensitivity.morris_screening(case, n_trajectories=0)
    with pytest.raises(PhysicsError):
        sensitivity.morris_screening(case, n_levels=1)
