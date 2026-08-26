"""Tests for parametric sweeps over the design space."""

from __future__ import annotations

import numpy as np
import pytest

from hygroadh import sweep
from hygroadh.threshold import MoistureCriterion
from hygroadh.config import SweepAxes
from hygroadh.materials import Polymer
from hygroadh.simulate import Case, ExposureCondition
from hygroadh.sorption import HenryIsotherm
from hygroadh.units import PhysicsError, from_days, to_celsius, to_kelvin


def _case(**kwargs) -> Case:
    polymer = Polymer(
        diffusivity_ref=2.0e-13,
        isotherm=HenryIsotherm(m_ref_pct=2.4),
    )
    condition = kwargs.pop("condition", None) or ExposureCondition(
        temperature_k=to_kelvin(60.0), duration=from_days(60),
        relative_humidity=0.85, n_times=600,
    )
    defaults = dict(thickness=200e-6, polymer=polymer, condition=condition)
    defaults.update(kwargs)
    return Case(**defaults)


def test_grid_shape_and_axis_order_are_canonical():
    result = sweep.run_sweep(_case(), {
        "diffusivity": [1e-13, 2e-13],
        "thickness": [1e-4, 2e-4, 4e-4],
    })
    # Reported in canonical order regardless of the order they were given.
    assert result.axis_names == ("thickness", "diffusivity")
    assert result.shape == (3, 2)
    assert result.surface("time_to_threshold_s").shape == (3, 2)
    assert len(result.records) == 6


def test_sweep_accepts_a_sweep_axes_object():
    axes = SweepAxes(thickness=np.array([1e-4, 2e-4]),
                     temperature_k=to_kelvin(np.array([40.0, 60.0])))
    result = sweep.run_sweep(_case(), axes)
    assert result.axis_names == ("thickness", "temperature_k")
    assert result.shape == (2, 2)


def test_every_response_is_populated_at_every_point():
    result = sweep.run_sweep(_case(), {"thickness": [1e-4, 2e-4]})
    for name in sweep.RESPONSE_NAMES:
        surface = result.surface(name)
        assert not np.any(np.isnan(surface)), f"{name} has unfilled points"


# --- the trends the framework exists to predict ----------------------------

def test_time_to_threshold_scales_with_thickness_squared():
    """Diffusion-controlled: doubling thickness quadruples the time to threshold.

    Exact, not approximate. The index depends only on the instantaneous bondline
    moisture, which is a function of Fo = D t / l**2 alone, so the threshold is
    always crossed at one fixed Fourier number.
    """
    thickness = np.array([1e-4, 2e-4, 4e-4])
    result = sweep.run_sweep(
        _case(), {"thickness": thickness}
    )
    times = result.surface("time_to_threshold_s")
    assert np.all(np.isfinite(times))
    ratios = times[1:] / times[:-1]
    assert ratios == pytest.approx([4.0, 4.0], rel=0.05)


def test_time_to_threshold_is_inversely_proportional_to_diffusivity():
    diffusivity = np.array([1e-13, 2e-13, 4e-13])
    result = sweep.run_sweep(
        _case(), {"diffusivity": diffusivity}
    )
    times = result.surface("time_to_threshold_s")
    ratios = times[:-1] / times[1:]
    assert ratios == pytest.approx([2.0, 2.0], rel=0.05)


def test_raising_temperature_shortens_the_time_to_threshold():
    result = sweep.run_sweep(_case(), {
        "temperature_k": to_kelvin(np.array([25.0, 40.0, 60.0, 80.0]))
    })
    times = result.surface("time_to_threshold_s")
    assert np.all(np.diff(times) < 0.0)


def test_thickness_and_temperature_map_is_monotone_in_both_directions():
    """The headline dashboard surface: thickness protects, temperature attacks."""
    result = sweep.run_sweep(_case(), {
        "thickness": np.array([50, 100, 200, 400]) * 1e-6,
        "temperature_k": to_kelvin(np.array([25.0, 45.0, 65.0])),
    })
    times = result.surface("time_to_threshold_s")
    assert np.all(np.diff(times, axis=0) > 0.0), "thicker must survive longer"
    assert np.all(np.diff(times, axis=1) < 0.0), "hotter must fail sooner"


# --- unreachable thresholds ------------------------------------------------

def test_unreachable_thresholds_are_infinite_and_counted_not_raised():
    # Equilibrium uptake at 30% RH is 0.72 wt%, so a 2 wt% threshold at the far
    # face can never be met however long the film is exposed.
    case = _case(
        condition=ExposureCondition(temperature_k=to_kelvin(5.0), duration=from_days(1),
                                    relative_humidity=0.3, n_times=100),
        criterion=MoistureCriterion(2.0, "wt_pct"),
    )
    result = sweep.run_sweep(case, {"thickness": [1e-3, 2e-3]})
    times = result.surface("time_to_threshold_s")
    assert np.all(np.isinf(times))
    assert result.finite_fraction("time_to_threshold_s") == 0.0
    # The always-finite responses must still be usable.
    assert np.all(np.isfinite(result.surface("final_interface_normalized")))


# --- records and presentation ---------------------------------------------

def test_records_carry_the_swept_axes_alongside_the_summary():
    result = sweep.run_sweep(_case(), {"thickness": [1e-4, 2e-4]})
    row = result.records[0]
    assert "thickness" in row
    assert row["thickness"] == pytest.approx(1e-4)
    assert "time_to_threshold_s" in row and "final_interface_pct" in row
    assert row["thickness_um"] == pytest.approx(100.0)


def test_axis_display_values_convert_to_readable_units():
    assert sweep.axis_display_values("thickness", np.array([1e-4])) == pytest.approx([100.0])
    assert sweep.axis_display_values(
        "temperature_k", np.array([to_kelvin(60.0)])
    ) == pytest.approx([60.0])
    assert to_celsius(to_kelvin(60.0)) == pytest.approx(60.0)


def test_axis_lookup_reports_unknown_names():
    result = sweep.run_sweep(_case(), {"thickness": [1e-4, 2e-4]})
    assert result.axis("thickness") == pytest.approx([1e-4, 2e-4])
    with pytest.raises(KeyError):
        result.axis("temperature_k")
    with pytest.raises(KeyError):
        result.surface("nonsense")


def test_progress_callback_is_called_once_per_point():
    seen = []
    sweep.run_sweep(_case(), {"thickness": [1e-4, 2e-4, 4e-4]},
                    progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


# --- validation ------------------------------------------------------------

def test_rejects_unknown_empty_or_non_positive_axes():
    case = _case()
    with pytest.raises(PhysicsError, match="unknown sweep axis"):
        sweep.run_sweep(case, {"humidity": [0.5]})
    with pytest.raises(PhysicsError, match="no sweep axes"):
        sweep.run_sweep(case, {})
    with pytest.raises(PhysicsError, match="no sweep axes"):
        sweep.run_sweep(case, SweepAxes())
    with pytest.raises(PhysicsError, match="positive"):
        sweep.run_sweep(case, {"thickness": [1e-4, 0.0]})
