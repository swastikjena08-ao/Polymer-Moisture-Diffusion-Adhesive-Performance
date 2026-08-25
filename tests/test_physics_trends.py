"""The qualitative claims this study exists to make.

Every other test file checks a component. This one checks the *physics the tool
is for*, so a refactor that quietly breaks the science fails loudly instead of
returning plausible-looking numbers. If a change makes a test here fail, the
change is wrong until argued otherwise.

The research question these pin down:

    How do film thickness, temperature, and water diffusivity influence the time
    for moisture to penetrate a polymer adhesive film?
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hygroadh.materials import Polymer
from hygroadh.simulate import Case, ExposureCondition, run
from hygroadh.sorption import HenryIsotherm
from hygroadh.threshold import MoistureCriterion
from hygroadh.units import from_days, to_kelvin


def _case(thickness=200e-6, diffusivity=1e-13, temperature_c=60.0,
          humidity=0.85, threshold=0.5, basis="normalized", exposure="one_sided",
          n_times=4000, duration_days=200.0, enthalpy=0.0,
          activation_energy=50e3) -> Case:
    polymer = Polymer(
        diffusivity_ref=diffusivity,
        activation_energy=activation_energy,
        isotherm=HenryIsotherm(m_ref_pct=2.4, enthalpy_sorption=enthalpy),
    )
    return Case(
        thickness=thickness,
        polymer=polymer,
        criterion=MoistureCriterion(threshold, basis),
        condition=ExposureCondition(
            temperature_k=to_kelvin(temperature_c),
            duration=from_days(duration_days),
            relative_humidity=humidity,
            exposure=exposure,
            n_times=n_times,
        ),
    )


def _at_fourier(result, target):
    """Index of the sample closest to a given Fourier number."""
    return int(np.argmin(np.abs(result.transport.fourier_number - target)))


def _crossing_fourier(result):
    """The Fourier number at which the threshold is crossed."""
    seconds = result.time_to_threshold
    sheet = 2 * result.case.thickness if result.case.condition.exposure == "one_sided" \
        else result.case.thickness
    return result.diffusivity * seconds / sheet**2


# --- 1. normalized transport is a function of the Fourier number alone -----

def test_normalized_transport_collapses_onto_one_curve_in_fourier_number():
    """Thickness and diffusivity enter only through Fo = D t / l**2.

    Scaling D with the square of thickness leaves the whole normalized history
    unchanged, for the film average and for the far face alike. This is why
    absolute time, not normalized uptake, is what the study reports.
    """
    reference = None
    for thickness, diffusivity in ((100e-6, 1e-13), (200e-6, 4e-13), (400e-6, 16e-13)):
        result = run(_case(thickness=thickness, diffusivity=diffusivity))
        if reference is None:
            reference = result
            continue
        assert result.uptake_normalized == pytest.approx(
            reference.uptake_normalized, abs=1e-9
        )
        assert result.interface_normalized == pytest.approx(
            reference.interface_normalized, abs=1e-9
        )


def test_the_threshold_is_crossed_at_one_fixed_fourier_number():
    """The reason the scaling laws below are exact rather than approximate.

    A normalized threshold is a level on the far-face curve, and that curve is a
    function of Fo alone --- so the crossing happens at the same Fo whatever the
    thickness, diffusivity, or temperature. Every result in Sections 5, 6 and 7
    of the dashboard follows from this one fact.
    """
    values = [
        _crossing_fourier(run(_case(**kwargs)))
        for kwargs in (
            {},
            {"thickness": 50e-6},
            {"thickness": 800e-6},
            {"diffusivity": 1e-14},
            {"diffusivity": 1e-11},
            {"temperature_c": 20.0},
            {"temperature_c": 80.0},
        )
    ]
    assert values == pytest.approx([values[0]] * len(values), rel=1e-3)


# --- 2. thickness: t ~ L^2 ------------------------------------------------

def test_penetration_time_scales_as_thickness_squared():
    """Doubling the film thickness quadruples the penetration time."""
    times = [
        run(_case(thickness=t)).time_to_threshold
        for t in (50e-6, 100e-6, 200e-6, 400e-6, 800e-6)
    ]
    assert np.all(np.isfinite(times))
    ratios = [times[i + 1] / times[i] for i in range(len(times) - 1)]
    assert ratios == pytest.approx([4.0] * len(ratios), rel=0.01)


def test_penetration_time_matches_the_characteristic_diffusion_time_shape():
    """t_threshold is a fixed multiple of l**2/D, which is the L**2/D relationship
    the study quotes --- recovered here rather than assumed."""
    ratios = []
    for thickness in (100e-6, 200e-6, 400e-6):
        for diffusivity in (5e-14, 1e-13, 4e-13):
            result = run(_case(thickness=thickness, diffusivity=diffusivity))
            sheet = 2 * thickness
            characteristic = sheet**2 / result.diffusivity
            ratios.append(result.time_to_threshold / characteristic)
    assert ratios == pytest.approx([ratios[0]] * len(ratios), rel=1e-3)


# --- 3. diffusivity: t ~ 1/D ---------------------------------------------

def test_penetration_time_is_inversely_proportional_to_diffusivity():
    times = [
        run(_case(diffusivity=d)).time_to_threshold
        for d in (1e-14, 1e-13, 1e-12, 1e-11)
    ]
    ratios = [times[i] / times[i + 1] for i in range(len(times) - 1)]
    assert ratios == pytest.approx([10.0] * len(ratios), rel=0.01)


# --- 4. temperature: through the Arrhenius factor -------------------------

def test_a_hotter_film_is_penetrated_sooner():
    times = [
        run(_case(temperature_c=t)).time_to_threshold
        for t in (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)
    ]
    assert np.all(np.isfinite(times))
    assert np.all(np.diff(times) < 0.0), f"times {times}"


def test_penetration_time_tracks_the_arrhenius_diffusivity_exactly():
    """Temperature acts on transport only through D(T).

    So the ratio of penetration times between any two temperatures must equal the
    inverse ratio of their diffusivities. This is a much sharper statement than
    "hotter is faster", and it would fail if a temperature dependence leaked into
    the geometry or the normalization.
    """
    cold, hot = run(_case(temperature_c=25.0)), run(_case(temperature_c=75.0))
    assert hot.time_to_threshold / cold.time_to_threshold == pytest.approx(
        cold.diffusivity / hot.diffusivity, rel=1e-3
    )


def test_a_zero_activation_energy_removes_the_temperature_effect():
    times = [
        run(_case(temperature_c=t, activation_energy=0.0)).time_to_threshold
        for t in (20.0, 50.0, 80.0)
    ]
    assert times == pytest.approx([times[0]] * 3, rel=1e-9)


def test_a_larger_activation_energy_makes_temperature_matter_more():
    spans = []
    for activation_energy in (20e3, 50e3, 80e3):
        cold = run(_case(temperature_c=20.0, activation_energy=activation_energy))
        hot = run(_case(temperature_c=80.0, activation_energy=activation_energy))
        spans.append(cold.time_to_threshold / hot.time_to_threshold)
    assert np.all(np.diff(spans) > 0.0), f"cold/hot time ratio by Ea: {spans}"


# --- 5. what temperature does NOT do under these assumptions -------------

def test_by_default_temperature_does_not_change_the_equilibrium_uptake():
    """With Henry's law and no sorption enthalpy, temperature sets the rate only.

    Asserted so the assumption stays visible: the same film ends up equally wet
    at 20 degC and 80 degC, it just gets there sooner when hot.
    """
    cold, hot = run(_case(temperature_c=20.0)), run(_case(temperature_c=80.0))
    assert hot.saturation_pct == pytest.approx(cold.saturation_pct)
    assert hot.interface_pct[-1] == pytest.approx(cold.interface_pct[-1], rel=1e-6)


def test_a_sorption_enthalpy_lets_temperature_move_the_equilibrium_too():
    cold = run(_case(temperature_c=20.0, enthalpy=8e3))
    hot = run(_case(temperature_c=80.0, enthalpy=8e3))
    assert hot.saturation_pct > cold.saturation_pct


# --- 6. humidity ---------------------------------------------------------

def test_humidity_sets_the_equilibrium_uptake_proportionally():
    """Henry's law: equilibrium uptake is linear in relative humidity."""
    values = [run(_case(humidity=rh)).saturation_pct for rh in (0.25, 0.5, 0.75, 1.0)]
    assert values == pytest.approx([0.6, 1.2, 1.8, 2.4], rel=1e-9)


def test_a_normalized_threshold_is_reached_at_the_same_time_at_any_humidity():
    """Because the criterion is a fraction of the film's own equilibrium.

    Humidity scales the whole concentration field, so a *relative* threshold is
    crossed at the same instant. Choosing the absolute basis is what makes
    humidity matter.
    """
    times = [
        run(_case(humidity=rh, threshold=0.5, basis="normalized")).time_to_threshold
        for rh in (0.3, 0.6, 0.9)
    ]
    assert times == pytest.approx([times[0]] * 3, rel=1e-6)


def test_an_absolute_threshold_is_reached_sooner_at_higher_humidity():
    times = [
        run(_case(humidity=rh, threshold=0.6, basis="wt_pct")).time_to_threshold
        for rh in (0.4, 0.6, 0.8, 1.0)
    ]
    assert np.all(np.isfinite(times))
    assert np.all(np.diff(times) < 0.0), f"times {times}"


def test_an_absolute_threshold_above_the_equilibrium_uptake_is_never_reached():
    """Analytic condition, not a numerical accident: the film simply cannot hold
    that much water at this humidity."""
    dry = run(_case(humidity=0.2, threshold=0.6, basis="wt_pct"))
    assert dry.saturation_pct == pytest.approx(0.48)
    assert not dry.threshold_reachable
    assert dry.time_to_threshold == np.inf

    wet = run(_case(humidity=0.4, threshold=0.6, basis="wt_pct"))
    assert wet.saturation_pct == pytest.approx(0.96)
    assert wet.threshold_reachable
    assert np.isfinite(wet.time_to_threshold)


# --- 7. why the far face, and why the geometry matters -------------------

def test_the_far_face_always_lags_the_film_average():
    result = run(_case())
    assert np.all(result.interface_normalized <= result.uptake_normalized + 1e-12)
    early = _at_fourier(result, 0.02)
    assert result.interface_normalized[early] < 0.5 * result.uptake_normalized[early]


def test_reading_the_threshold_off_the_film_average_understates_the_time():
    """Why the study tracks the far face, stated as a numerical difference.

    The film average crosses any given level earlier than the far face does, so
    using it would report moisture arriving at the bondline sooner than it does.
    """
    result = run(_case())
    criterion = result.case.criterion
    honest = result.time_to_threshold
    naive = criterion.time_to_threshold(
        result.time, result.uptake_normalized, result.saturation_pct
    )
    assert naive < honest
    assert naive < 0.6 * honest


def test_a_free_film_is_penetrated_four_times_sooner_than_a_bonded_one():
    """Wetted on both faces, the diffusion length halves, so the time falls by four."""
    bonded = run(_case(exposure="one_sided")).time_to_threshold
    free = run(_case(exposure="two_sided")).time_to_threshold
    assert free / bonded == pytest.approx(0.25, rel=0.01)


def test_a_one_sided_film_matches_a_double_thickness_free_film():
    one = run(_case(thickness=200e-6, exposure="one_sided"))
    two = run(_case(thickness=400e-6, exposure="two_sided"))
    assert one.time_to_threshold == pytest.approx(two.time_to_threshold, rel=1e-6)


# --- 8. the model's boundaries, asserted so they cannot be forgotten -----

def test_nothing_accumulates_so_a_dried_film_returns_to_its_initial_state():
    """The honest boundary of this transport-only model.

    Moisture leaves as readily as it arrived, and no damage variable records that
    it was ever there. A film wet for sixty days and then dried is
    indistinguishable from one never exposed. Any real irreversible degradation
    --- hydrolysis, permanent plasticization --- is outside this scope, and this
    test exists so a future change cannot quietly imply otherwise.
    """
    from hygroadh.diffusion.fd import Schedule

    case = _case(duration_days=120.0, n_times=1500)
    wet_end = from_days(60.0)
    condition = replace(
        case.condition,
        humidity_schedule=Schedule(
            np.array([0.0, wet_end, wet_end * 1.001, from_days(120.0)]),
            np.array([0.85, 0.85, 0.0, 0.0]),
        ),
        time_spacing="linear",
    )
    result = run(replace(case, condition=condition))

    assert result.interface_normalized.max() > 0.9, "the far face must actually wet"
    assert result.uptake_normalized[-1] < 0.02, "the film must actually dry out"
    assert result.interface_normalized[-1] < 0.02


def test_the_solution_stays_within_physical_bounds():
    result = run(_case())
    assert np.all(result.uptake_normalized >= -1e-12)
    assert np.all(result.uptake_normalized <= 1.0 + 1e-12)
    assert np.all(result.interface_normalized >= -1e-12)
    assert np.all(result.interface_normalized <= 1.0 + 1e-12)
    assert np.all(np.diff(result.uptake_normalized) >= -1e-12), "monotone under steady exposure"
