"""Verification of the analytical Fickian solution against known results."""

from __future__ import annotations

import math

import numpy as np
import pytest

from _compat import trapezoid

from hygroadh.diffusion import analytical
from hygroadh.diffusion.base import equivalent_sheet_thickness


def test_uptake_starts_at_zero_and_saturates():
    assert analytical.fickian_uptake(0.0) == 0.0
    assert analytical.fickian_uptake(5.0) == pytest.approx(1.0, abs=1e-12)


def test_uptake_is_monotone_increasing():
    fo = np.geomspace(1e-8, 5.0, 4000)
    uptake = analytical.fickian_uptake(fo)
    assert np.all(np.diff(uptake) >= -1e-14)


def test_short_time_uptake_matches_square_root_law():
    """At small Fo the sheet behaves as two semi-infinite bodies: M/M_inf = 4 sqrt(Fo/pi)."""
    fo = np.array([1e-8, 1e-7, 1e-6, 1e-5])
    expected = 4.0 * np.sqrt(fo / math.pi)
    assert analytical.fickian_uptake(fo) == pytest.approx(expected, rel=1e-9)


def test_the_two_series_agree_across_the_crossover():
    """The short-time and eigenfunction series must overlap, or the curve has a step."""
    fo = analytical.SHORT_TIME_CROSSOVER
    below = analytical._uptake_short(np.array([fo]))[0]
    above = analytical._uptake_eigen(np.array([fo]))[0]
    assert below == pytest.approx(above, abs=1e-12)


def test_half_uptake_fourier_number_matches_the_published_constant():
    assert analytical.fourier_number_for_uptake(0.5) == pytest.approx(
        analytical.FOURIER_HALF_UPTAKE, rel=1e-6
    )


def test_fourier_number_for_uptake_inverts_the_uptake_curve():
    for fraction in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        fo = analytical.fourier_number_for_uptake(fraction)
        assert analytical.fickian_uptake(fo) == pytest.approx(fraction, rel=1e-9)


def test_half_time_scales_with_thickness_squared_and_inverse_diffusivity():
    base = analytical.half_time(1e-4, 1e-12)
    assert analytical.half_time(2e-4, 1e-12) == pytest.approx(4.0 * base)
    assert analytical.half_time(1e-4, 2e-12) == pytest.approx(0.5 * base)


def test_diffusivity_from_half_time_round_trips():
    thickness, diffusivity = 2.5e-4, 3.7e-13
    t_half = analytical.half_time(thickness, diffusivity)
    assert analytical.diffusivity_from_half_time(thickness, t_half) == pytest.approx(
        diffusivity, rel=1e-12
    )


def test_one_sided_film_matches_a_double_thickness_free_film():
    """The no-flux substrate plane and a symmetry mid-plane impose the same condition."""
    t = np.geomspace(1.0, 1e7, 60)
    one = analytical.solve(t, 1e-4, 1e-12, 2.0, 298.15, exposure="one_sided")
    two = analytical.solve(t, 2e-4, 1e-12, 2.0, 298.15, exposure="two_sided")
    assert one.uptake_normalized == pytest.approx(two.uptake_normalized, abs=1e-12)
    assert one.interface_normalized == pytest.approx(two.interface_normalized, abs=1e-12)


def test_profile_is_zero_initially_and_uniform_at_saturation():
    xi = np.linspace(-0.5, 0.5, 21)
    assert analytical.fickian_profile([0.0], xi)[0] == pytest.approx(0.0, abs=1e-12)
    assert analytical.fickian_profile([10.0], xi)[0] == pytest.approx(1.0, abs=1e-9)


def test_profile_holds_the_surface_at_saturation_at_all_times():
    fo = np.geomspace(1e-6, 2.0, 40)
    surface = analytical.fickian_profile(fo, [0.5])[:, 0]
    assert surface == pytest.approx(1.0, abs=1e-9)


def test_profile_decreases_monotonically_with_depth():
    """Water enters at the face, so concentration must fall toward the no-flux plane."""
    depth_from_face = np.linspace(0.0, 0.5, 30)
    for fo in (1e-4, 1e-2, 0.05, 0.2, 1.0):
        profile = analytical.fickian_profile([fo], 0.5 - depth_from_face)[0]
        assert np.all(np.diff(profile) <= 1e-12)


def test_profile_series_agree_wherever_both_converge():
    """Compared at identical Fo, so this measures the series and not the curve's slope."""
    xi = np.linspace(-0.5, 0.5, 21)
    for fo in (0.02, 0.03, analytical.SHORT_TIME_CROSSOVER, 0.08, 0.2, 0.3):
        fo_arr = np.array([fo])
        short = analytical._profile_short(fo_arr, xi)[0]
        eigen = analytical._profile_eigen(fo_arr, xi)[0]
        assert short == pytest.approx(eigen, abs=1e-13), f"mismatch at Fo={fo}"


def test_interface_lags_behind_the_mean_uptake():
    """The whole point of tracking the interface: it wets later than the film average."""
    fo = np.geomspace(1e-5, 1.0, 50)
    interface = analytical.fickian_interface(fo)
    mean = analytical.fickian_uptake(fo)
    assert np.all(interface <= mean + 1e-12)
    assert np.any(interface < 0.5 * mean)


def test_uptake_integrates_the_profile():
    """Mass balance on the exact solution: the mean of C/C_sat is M/M_inf."""
    xi = np.linspace(-0.5, 0.5, 20001)
    for fo in (1e-3, 0.05, 0.3, 1.0):
        profile = analytical.fickian_profile([fo], xi)[0]
        mean = trapezoid(profile, xi)
        assert mean == pytest.approx(analytical.fickian_uptake(fo), abs=2e-5)


def test_solve_populates_depth_grid_and_absolute_uptake():
    t = np.linspace(0.0, 1e6, 25)
    result = analytical.solve(t, 1e-4, 1e-12, 3.0, 323.15, n_depth=15)
    assert result.depth.shape == (15,)
    assert result.profile_normalized.shape == (25, 15)
    assert result.depth[0] == 0.0
    assert result.depth[-1] == pytest.approx(1e-4)
    assert result.uptake_pct == pytest.approx(result.uptake_normalized * 3.0)
    assert result.solver == "analytical"


def test_fourier_number_property_is_consistent_with_the_geometry():
    t = np.linspace(0.0, 1e6, 11)
    result = analytical.solve(t, 1e-4, 1e-12, 2.0, 298.15)
    sheet = equivalent_sheet_thickness(1e-4, "one_sided")
    assert result.fourier_number == pytest.approx(1e-12 * t / sheet**2)


def test_time_to_uptake_interpolates_and_reports_unreachable_targets():
    # Log spacing: the interesting part of the curve is at short times, and a
    # uniform grid coarse enough to reach saturation resolves it badly.
    t = np.concatenate([[0.0], np.geomspace(1.0, 1e7, 4000)])
    result = analytical.solve(t, 1e-4, 1e-12, 2.0, 298.15, n_depth=None)
    predicted = result.time_to_uptake(0.5)
    assert predicted == pytest.approx(analytical.half_time(1e-4, 1e-12), rel=1e-3)
    assert result.time_to_uptake(1.5) == math.inf


def test_solve_rejects_bad_inputs():
    with pytest.raises(ValueError):
        analytical.solve([0.0, -1.0], 1e-4, 1e-12, 2.0, 298.15)
    with pytest.raises(ValueError):
        analytical.solve([10.0, 1.0], 1e-4, 1e-12, 2.0, 298.15)
    with pytest.raises(ValueError):
        analytical.fickian_uptake(-0.1)
    with pytest.raises(ValueError):
        analytical.fickian_profile([1.0], [0.9])
