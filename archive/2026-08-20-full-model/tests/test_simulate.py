"""Tests for case orchestration: solver selection, wiring, and derived scalars."""

from __future__ import annotations

import numpy as np
import pytest

from hygroadh.adhesion import AdhesionModel
from hygroadh.diffusion.fd import Schedule
from hygroadh.materials import Polymer
from hygroadh.simulate import Case, ExposureCondition, run, surface_condition
from hygroadh.sorption import PowerLawIsotherm
from hygroadh.units import PhysicsError, from_days, to_kelvin


def _polymer(**kwargs) -> Polymer:
    defaults = dict(
        diffusivity_ref=2.0e-13,
        activation_energy=50.0e3,
        isotherm=PowerLawIsotherm(m_ref_pct=2.4, exponent=0.8),
        tg_dry_k=to_kelvin(120.0),
    )
    defaults.update(kwargs)
    return Polymer(**defaults)


def _case(**kwargs) -> Case:
    condition = kwargs.pop("condition", None) or ExposureCondition(
        temperature_k=to_kelvin(60.0), duration=from_days(60), relative_humidity=0.85
    )
    defaults = dict(thickness=200e-6, condition=condition, polymer=_polymer())
    defaults.update(kwargs)
    return Case(**defaults)


# --- time grids ------------------------------------------------------------

def test_time_grid_starts_at_zero_and_ends_at_the_duration():
    for spacing in ("log", "linear"):
        condition = ExposureCondition(
            temperature_k=300.0, duration=1e6, n_times=50, time_spacing=spacing
        )
        grid = condition.time_grid()
        assert grid[0] == 0.0
        assert grid[-1] == pytest.approx(1e6)
        assert grid.size == 50
        assert np.all(np.diff(grid) > 0)


def test_spacing_defaults_to_log_but_switches_to_linear_for_a_schedule():
    """Log spacing resolves the early square-root region; cycling needs uniform."""
    steady = ExposureCondition(temperature_k=300.0, duration=1e6)
    assert steady.resolved_spacing == "log"
    cycled = ExposureCondition(
        temperature_k=300.0, duration=1e6,
        humidity_schedule=Schedule.cycle(0.9, 0.1, 1e5, 10),
    )
    assert cycled.resolved_spacing == "linear"


# --- solver selection ------------------------------------------------------

def test_plain_cases_use_the_exact_analytical_solution():
    result = run(_case())
    assert result.transport.solver == "analytical"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"polymer": Polymer(diffusivity_ref=2e-13, conc_dependence=1.0,
                            isotherm=PowerLawIsotherm(2.4), tg_dry_k=to_kelvin(120))},
        {"condition": ExposureCondition(temperature_k=to_kelvin(60.0),
                                        duration=from_days(10),
                                        surface_transfer=1e-9)},
        {"condition": ExposureCondition(
            temperature_k=to_kelvin(60.0), duration=from_days(10),
            humidity_schedule=Schedule.cycle(0.9, 0.1, from_days(2), 5))},
    ],
)
def test_features_beyond_the_analytical_solution_select_finite_volumes(kwargs):
    case = _case(**kwargs)
    assert case.requires_finite_volume
    assert case.resolved_solver == "fd"
    assert run(case).transport.solver == "finite_volume"


def test_requesting_the_analytical_solver_for_an_unsupported_case_is_refused():
    """Silently substituting different physics would be worse than refusing."""
    case = _case(solver="analytical", condition=ExposureCondition(
        temperature_k=to_kelvin(60.0), duration=from_days(10), surface_transfer=1e-9))
    with pytest.raises(PhysicsError, match="analytical solution assumes"):
        case.resolved_solver


def test_the_two_solvers_agree_on_a_case_both_can_handle():
    exact = run(_case(solver="analytical", n_cells=160))
    numeric = run(_case(solver="fd", n_cells=160))
    assert numeric.uptake_normalized == pytest.approx(exact.uptake_normalized, abs=2e-3)
    assert numeric.index == pytest.approx(exact.index, abs=5e-3)


# --- temperature resolution ------------------------------------------------

def test_temperature_resolves_diffusivity_and_saturation_through_the_polymer():
    case = _case()
    result = run(case)
    assert result.diffusivity == pytest.approx(
        case.polymer.diffusivity(case.condition.temperature_k)
    )
    assert result.saturation_pct == pytest.approx(
        case.polymer.saturation_pct(case.condition.temperature_k,
                                    case.condition.relative_humidity)
    )


def test_uptake_is_normalized_against_the_nominal_humidity_not_liquid_water():
    """M/M_inf -> 1 means equilibrium with the exposure RH, not with immersion."""
    result = run(_case())
    assert result.uptake_normalized[-1] == pytest.approx(1.0, abs=1e-6)
    assert result.saturation_pct < result.case.polymer.saturation_pct(
        result.case.condition.temperature_k, 1.0
    )


# --- humidity schedules ----------------------------------------------------

def test_a_constant_schedule_at_the_nominal_humidity_is_a_unit_surface_value():
    polymer = _polymer()
    condition = ExposureCondition(
        temperature_k=to_kelvin(60.0), duration=from_days(10), relative_humidity=0.85,
        humidity_schedule=Schedule.constant(0.85),
    )
    saturation = polymer.saturation_pct(condition.temperature_k, 0.85)
    surface = surface_condition(polymer, condition, saturation)
    assert surface(0.0) == pytest.approx(1.0)
    assert surface(1e6) == pytest.approx(1.0)


def test_the_isotherm_is_applied_to_the_schedule_not_assumed_linear():
    """With exponent 0.8, half the humidity gives more than half the uptake."""
    polymer = _polymer(isotherm=PowerLawIsotherm(m_ref_pct=2.4, exponent=0.8))
    condition = ExposureCondition(
        temperature_k=to_kelvin(60.0), duration=from_days(10), relative_humidity=1.0,
        humidity_schedule=Schedule.constant(0.5),
    )
    saturation = polymer.saturation_pct(condition.temperature_k, 1.0)
    surface = surface_condition(polymer, condition, saturation)
    assert surface(0.0) == pytest.approx(0.5**0.8)
    assert surface(0.0) > 0.5


def test_a_drying_leg_lets_uptake_fall_but_never_the_index_recover_fully():
    polymer = _polymer()
    wet = from_days(20)
    condition = ExposureCondition(
        temperature_k=to_kelvin(70.0), duration=from_days(60), relative_humidity=0.9,
        humidity_schedule=Schedule(
            np.array([0.0, wet, wet * 1.001, from_days(60)]),
            np.array([0.9, 0.9, 0.0, 0.0]),
        ),
        n_times=400,
    )
    result = run(_case(condition=condition, thickness=100e-6))
    peak = int(np.argmax(result.uptake_normalized))
    assert 0 < peak < result.time.size - 1
    assert result.uptake_normalized[-1] < 0.2 * result.uptake_normalized[peak]
    # Reversible mechanisms recover; the index does not.
    assert result.adhesion.plasticization[-1] == pytest.approx(1.0, abs=1e-3)
    assert result.final_index < 1.0


# --- derived scalars -------------------------------------------------------

def test_half_uptake_time_matches_the_analytical_half_time():
    from hygroadh.diffusion import analytical

    case = _case(condition=ExposureCondition(
        temperature_k=to_kelvin(60.0), duration=from_days(60), relative_humidity=0.85,
        n_times=3000))
    result = run(case)
    expected = analytical.half_time(case.thickness, result.diffusivity)
    assert result.time_to_half_uptake == pytest.approx(expected, rel=2e-3)


def test_unreached_ari_threshold_is_reported_as_infinite_not_raised():
    """A cold thick film legitimately never crosses the threshold."""
    case = _case(
        thickness=2e-3,
        condition=ExposureCondition(temperature_k=to_kelvin(5.0), duration=from_days(1),
                                    relative_humidity=0.4),
        adhesion=AdhesionModel(plasticization_gain=0.05, hydrolysis_rate_ref=1e-14),
    )
    assert run(case).time_to_ari_threshold == np.inf


def test_summary_reports_every_scalar_a_report_needs():
    summary = run(_case()).summary()
    for key in (
        "thickness_um", "temperature_c", "diffusivity_m2_s", "saturation_pct",
        "time_to_half_uptake_s", "final_ari", "time_to_ari_threshold_s",
        "dominant_mechanism", "solver",
    ):
        assert key in summary, f"missing {key}"
    assert summary["thickness_um"] == pytest.approx(200.0)
    assert summary["temperature_c"] == pytest.approx(60.0)


# --- case variation helpers used by sweeps ---------------------------------

def test_variation_helpers_change_one_axis_and_leave_the_rest_alone():
    case = _case()
    thicker = case.with_thickness(400e-6)
    hotter = case.with_temperature(to_kelvin(80.0))
    faster = case.with_diffusivity(1e-12)
    assert thicker.thickness == 400e-6
    assert thicker.condition.temperature_k == case.condition.temperature_k
    assert hotter.condition.temperature_k == to_kelvin(80.0)
    assert hotter.thickness == case.thickness
    assert faster.polymer.diffusivity_ref == 1e-12
    assert faster.thickness == case.thickness
    assert case.thickness == 200e-6, "the original must be untouched"


# --- validation ------------------------------------------------------------

def test_rejects_invalid_cases_and_conditions():
    with pytest.raises(PhysicsError):
        _case(thickness=0.0)
    with pytest.raises(PhysicsError):
        _case(solver="magic")
    with pytest.raises(PhysicsError):
        _case(ari_threshold=1.5)
    with pytest.raises(PhysicsError, match="greater than zero"):
        ExposureCondition(temperature_k=300.0, duration=1e6, relative_humidity=0.0)
    with pytest.raises(PhysicsError):
        ExposureCondition(temperature_k=300.0, duration=-1.0)
    with pytest.raises(PhysicsError):
        ExposureCondition(temperature_k=300.0, duration=1e6, n_times=1)
    with pytest.raises(PhysicsError):
        ExposureCondition(temperature_k=300.0, duration=1e6, exposure="edgewise")
