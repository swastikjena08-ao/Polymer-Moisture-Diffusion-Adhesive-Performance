"""Verification of the finite-volume solver against the exact solution."""

from __future__ import annotations

import numpy as np
import pytest

from _compat import trapezoid

from hygroadh.diffusion import analytical, fd
from hygroadh.diffusion.base import diffusion_length
from hygroadh.units import ConvergenceError, PhysicsError

THICKNESS = 2.0e-4
DIFFUSIVITY = 1.0e-12
TEMPERATURE = 323.15


def _times(n=60, end_fourier=1.2, thickness=THICKNESS, exposure="one_sided"):
    """Output times spanning up to a given Fourier number, log-spaced."""
    sheet = 2.0 * thickness if exposure == "one_sided" else thickness
    t_end = end_fourier * sheet**2 / DIFFUSIVITY
    return np.concatenate([[0.0], np.geomspace(t_end * 1e-4, t_end, n)])


def test_initial_uptake_is_exactly_zero():
    """The cell-centred grid must not manufacture mass from the surface jump."""
    t = _times()
    result = fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE)
    assert result.uptake_normalized[0] == 0.0
    assert result.interface_normalized[0] == 0.0


def test_matches_the_analytical_uptake_curve():
    t = _times()
    numeric = fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_cells=160)
    exact = analytical.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_depth=None)
    error = np.max(np.abs(numeric.uptake_normalized - exact.uptake_normalized))
    assert error < 1e-3, f"max uptake error {error:.2e}"


def test_matches_the_analytical_interface_history():
    """The adhesion-driving quantity must be right, not just the gravimetric average."""
    t = _times()
    numeric = fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_cells=160)
    exact = analytical.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_depth=None)
    error = np.max(np.abs(numeric.interface_normalized - exact.interface_normalized))
    assert error < 2e-3, f"max interface error {error:.2e}"


def test_spatial_convergence_is_second_order():
    """Refining the grid must reduce the error at second order.

    The norm deliberately excludes the earliest samples. At very small Fo the
    diffusion front is thinner than a single cell -- 0.44 cells at Fo = 1.2e-4
    on the coarsest grid here -- so no fixed grid resolves it and those points
    are not in the asymptotic range. Including them masks the true order
    (measured 0.85 instead of 2.0) while saying nothing about the scheme.
    """
    t = _times(n=25, end_fourier=0.3)
    exact = analytical.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_depth=None)
    resolved = exact.fourier_number >= 0.005
    errors = []
    # Hold the *time* step fixed in absolute terms across every grid, so this
    # isolates spatial error. Leaving the adaptive rule on would let temporal
    # error plateau at ~1.6e-4 and mask the spatial order once the grid is fine.
    fixed_dt = 20.0
    for n_cells in (20, 40, 80, 160):
        dz = (THICKNESS) / n_cells
        numeric = fd.solve(
            t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_cells=n_cells,
            cfl=fixed_dt * DIFFUSIVITY / dz**2, step_growth=0.0,
        )
        deviation = np.abs(numeric.uptake_normalized - exact.uptake_normalized)
        errors.append(np.max(deviation[resolved]))
    errors = np.array(errors)
    assert np.all(np.diff(errors) < 0.0), f"error not decreasing: {errors}"
    orders = np.log2(errors[:-1] / errors[1:])
    assert np.min(orders) > 1.8, f"observed convergence orders {orders}"


def test_conserves_mass_against_the_integrated_surface_flux():
    """Absorbed mass must equal the time integral of the flux entering the face.

    Crank-Nicolson's time integration *is* the trapezoid rule, so with the
    output grid finer than the internal step and Rannacher startup disabled,
    this quadrature is the scheme's own and the balance must close to machine
    precision. Using backward Euler instead leaves a 2.5% gap that is purely a
    quadrature mismatch, not a leak -- which is why the tolerance here is
    1e-9 rather than something forgiving.
    """
    t = np.linspace(0.0, 0.3 * (2 * THICKNESS) ** 2 / DIFFUSIVITY, 3000)
    n_cells = 60
    result = fd.solve(
        t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_cells=n_cells,
        theta=0.5, rannacher_steps=0,
    )
    span = diffusion_length(THICKNESS, "one_sided")
    dz = span / n_cells
    conductance = 2.0 * DIFFUSIVITY / dz  # Dirichlet half-cell conductance
    flux = conductance * (1.0 - result.profile_normalized[:, 0])
    absorbed = trapezoid(flux, t)
    gained = result.uptake_normalized[-1] * span
    assert absorbed == pytest.approx(gained, rel=1e-9)


def test_equilibrium_is_a_fixed_point():
    """Starting at the surface value, nothing may drift."""
    t = _times()
    result = fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, initial=1.0,
                      surface=1.0)
    assert result.uptake_normalized == pytest.approx(1.0, abs=1e-12)


def test_solution_stays_within_physical_bounds():
    t = _times()
    result = fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_cells=40)
    assert np.all(result.profile_normalized >= -1e-12)
    assert np.all(result.profile_normalized <= 1.0 + 1e-12)
    assert np.all(np.diff(result.uptake_normalized) >= -1e-12)


def test_one_sided_and_two_sided_geometries_agree():
    t = _times()
    one = fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, exposure="one_sided")
    two = fd.solve(t, 2 * THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE,
                   exposure="two_sided")
    assert one.uptake_normalized == pytest.approx(two.uptake_normalized, abs=1e-12)


def test_profile_decreases_from_the_wetted_face():
    t = _times(n=20)
    result = fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_cells=40)
    for row in result.profile_normalized[1:]:
        assert np.all(np.diff(row) <= 1e-12)


def test_surface_resistance_slows_uptake_and_recovers_dirichlet_when_large():
    t = _times()
    ideal = fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_cells=40)
    resistive = fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_cells=40,
                         surface_transfer=DIFFUSIVITY / THICKNESS)
    assert np.all(resistive.uptake_normalized <= ideal.uptake_normalized + 1e-12)
    assert resistive.uptake_normalized[len(t) // 2] < ideal.uptake_normalized[len(t) // 2]

    fast = fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_cells=40,
                    surface_transfer=1e6 * DIFFUSIVITY / THICKNESS)
    assert fast.uptake_normalized == pytest.approx(ideal.uptake_normalized, abs=1e-6)


def test_constant_schedule_equals_a_constant_float():
    t = _times(n=20)
    plain = fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_cells=30,
                     surface=0.7)
    scheduled = fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_cells=30,
                         surface=fd.Schedule.constant(0.7))
    assert plain.uptake_normalized == pytest.approx(scheduled.uptake_normalized)


def test_drying_leg_reverses_uptake():
    """A wet-then-dry schedule must absorb and then desorb."""
    sheet = 2 * THICKNESS
    t_wet = 0.4 * sheet**2 / DIFFUSIVITY
    schedule = fd.Schedule(np.array([0.0, t_wet, t_wet * 1.001, 3 * t_wet]),
                           np.array([1.0, 1.0, 0.0, 0.0]))
    t = np.linspace(0.0, 3 * t_wet, 400)
    result = fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_cells=40,
                      surface=schedule)
    peak = int(np.argmax(result.uptake_normalized))
    assert 0 < peak < len(t) - 1
    assert result.uptake_normalized[-1] < 0.1 * result.uptake_normalized[peak]


def test_cycle_schedule_alternates_between_its_levels():
    schedule = fd.Schedule.cycle(high=0.9, low=0.1, period=100.0, n_cycles=3)
    assert schedule(0.0) == pytest.approx(0.9)
    assert schedule(40.0) == pytest.approx(0.9)
    assert schedule(90.0) == pytest.approx(0.1)
    assert schedule.maximum == pytest.approx(0.9)
    assert schedule(1e6) == pytest.approx(0.1)  # held after the last knot


def test_tridiagonal_solver_matches_a_dense_solve():
    rng = np.random.default_rng(11)
    n = 25
    lower = np.concatenate([[0.0], rng.uniform(-1, -0.2, n - 1)])
    upper = np.concatenate([rng.uniform(-1, -0.2, n - 1), [0.0]])
    diag = 4.0 + rng.uniform(0, 1, n)
    rhs = rng.normal(size=n)
    dense = np.diag(diag) + np.diag(upper[:-1], 1) + np.diag(lower[1:], -1)
    assert fd.solve_tridiagonal(lower, diag, upper, rhs) == pytest.approx(
        np.linalg.solve(dense, rhs)
    )


def test_tridiagonal_solver_rejects_a_singular_pivot():
    with pytest.raises(ConvergenceError):
        fd.solve_tridiagonal([0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0])


def test_backward_euler_and_crank_nicolson_agree_when_refined():
    """With the step actually refined, the two time schemes must converge together.

    ``step_growth=0`` is required here: with the adaptive rule on, ``cfl`` stops
    governing the step at late times and the first-order and second-order
    schemes stay visibly apart.
    """
    t = _times(n=30, end_fourier=0.5)
    common = dict(n_cells=40, cfl=0.25, step_growth=0.0)
    be = fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, theta=1.0, **common)
    cn = fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, theta=0.5, **common)
    assert be.uptake_normalized == pytest.approx(cn.uptake_normalized, abs=2e-3)


def test_growing_time_step_stays_accurate_over_a_long_exposure():
    """The adaptive step is what makes long runs affordable; it must not cost accuracy.

    A fixed step sized to dz**2/D needs ~1.4 million steps for a multi-month
    exposure. Letting it grow with elapsed time reduces that to a few hundred,
    and this pins the accuracy that buys.
    """
    sheet = 2 * THICKNESS
    t = np.concatenate([[0.0], np.geomspace(1.0, 20.0 * sheet**2 / DIFFUSIVITY, 200)])
    exact = analytical.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_depth=None)
    adaptive = fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_cells=80)
    # Restricted to the resolved window for the same reason as the convergence
    # test: this grid reaches down to Fo = 6e-6, where the front is a small
    # fraction of a cell and the error is spatial, not temporal.
    resolved = exact.fourier_number >= 0.005
    error = np.max(np.abs(adaptive.uptake_normalized - exact.uptake_normalized)[resolved])
    assert error < 1e-3, f"max uptake error {error:.2e}"
    interface_error = np.max(
        np.abs(adaptive.interface_normalized - exact.interface_normalized)[resolved]
    )
    assert interface_error < 1e-3, f"max interface error {interface_error:.2e}"


def test_step_growth_can_be_disabled_and_agrees_with_the_adaptive_result():
    t = _times(n=25, end_fourier=0.3)
    adaptive = fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_cells=40)
    fixed = fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_cells=40,
                     step_growth=0.0)
    assert adaptive.uptake_normalized == pytest.approx(
        fixed.uptake_normalized, abs=5e-4
    )


def test_rejects_a_negative_step_growth():
    with pytest.raises(PhysicsError):
        fd.solve(_times(n=5), THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE,
                 step_growth=-0.1)


def test_rejects_invalid_configuration():
    t = _times(n=5)
    with pytest.raises(PhysicsError):
        fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, n_cells=1)
    with pytest.raises(PhysicsError):
        fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, theta=1.5)
    with pytest.raises(PhysicsError):
        fd.solve(t, THICKNESS, DIFFUSIVITY, 2.0, TEMPERATURE, exposure="sideways")
    with pytest.raises(PhysicsError):
        fd.Schedule(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
