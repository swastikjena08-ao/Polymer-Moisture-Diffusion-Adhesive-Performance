"""The qualitative claims this framework exists to make.

Every other test file checks a component. This one checks the *physics the tool
is for*, so a refactor that quietly breaks the science fails loudly instead of
returning plausible-looking numbers. If a change makes a test here fail, the
change is wrong until argued otherwise.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hygroadh.adhesion import AdhesionModel
from hygroadh.materials import Polymer
from hygroadh.simulate import Case, ExposureCondition, run
from hygroadh.sorption import PowerLawIsotherm
from hygroadh.units import from_days, to_kelvin

# Two adhesion regimes, because the thickness scaling of service life is
# qualitatively different in each and both are physically real.
# Gains are per weight percent of interfacial water; see hygroadh.adhesion.
WETTING_LIMITED = dict(
    plasticization_gain=0.8, plasticization_floor=0.30, hydrolysis_rate_ref=1e-12
)
HYDROLYSIS_LIMITED = dict(
    plasticization_gain=0.15, plasticization_floor=0.55, hydrolysis_rate_ref=7e-8
)


def _case(thickness=200e-6, diffusivity=1e-13, temperature_c=60.0,
          threshold=0.7, adhesion=None, exposure="one_sided", n_times=4000,
          duration_days=400.0) -> Case:
    polymer = Polymer(
        diffusivity_ref=diffusivity,
        activation_energy=50e3,
        isotherm=PowerLawIsotherm(m_ref_pct=2.4, exponent=0.8, enthalpy_sorption=6e3),
        tg_dry_k=to_kelvin(140.0),
        tg_depression_per_pct=12.0,
    )
    return Case(
        thickness=thickness,
        polymer=polymer,
        adhesion=AdhesionModel(**{**HYDROLYSIS_LIMITED, **(adhesion or {})}),
        ari_threshold=threshold,
        condition=ExposureCondition(
            temperature_k=to_kelvin(temperature_c),
            duration=from_days(duration_days),
            relative_humidity=0.85,
            exposure=exposure,
            n_times=n_times,
        ),
    )


def _at_fourier(result, target):
    """Index of the sample closest to a given Fourier number."""
    return int(np.argmin(np.abs(result.transport.fourier_number - target)))


# --- 1. normalized uptake is a function of the Fourier number alone --------

def test_normalized_uptake_collapses_onto_one_curve_in_fourier_number():
    """Thickness and diffusivity are interchangeable for the gravimetric curve.

    They enter only through Fo = D t / l**2, so scaling D with the square of
    thickness leaves the whole normalized uptake history unchanged. This is the
    reason a gravimetric test alone cannot tell you how thickness affects
    adhesion.
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


# --- 2. the discriminating test: ARI does not collapse ---------------------

def test_ari_is_not_determined_by_normalized_uptake():
    """The central claim, and the one that justifies the whole framework.

    At a fixed Fourier number, three films of different thickness have the same
    normalized uptake *and* the same normalized bondline moisture --- yet
    different ARI. The thicker film reached that state over four times as much
    absolute time, and irreversible hydrolysis accumulated throughout. So ARI
    depends on absolute time, not only on how wet the film is, and no
    normalization of the uptake curve can substitute for it.

    A model that drove adhesion off M/M_inf would report all three as identical.
    """
    uptakes, indices = [], []
    for thickness in (100e-6, 200e-6, 400e-6):
        result = run(_case(thickness=thickness))
        index = _at_fourier(result, 0.3)
        uptakes.append(float(result.uptake_normalized[index]))
        indices.append(float(result.index[index]))

    # Same state of wetness...
    assert uptakes == pytest.approx([uptakes[0]] * 3, abs=1e-3)
    # ...materially different retained adhesion, decreasing with thickness.
    assert np.all(np.diff(indices) < -0.02), f"ARI values {indices}"


def test_at_matched_uptake_the_thicker_film_has_lost_more_adhesion():
    """The same statement read off the uptake curve instead of the Fo axis."""
    times, indices = [], []
    for thickness in (100e-6, 200e-6, 400e-6):
        result = run(_case(thickness=thickness))
        index = int(np.argmin(np.abs(result.uptake_normalized - 0.9)))
        times.append(float(result.time[index]))
        indices.append(float(result.index[index]))
    # Reaching 90% uptake takes ~4x longer for each doubling of thickness...
    assert times[1] / times[0] == pytest.approx(4.0, rel=0.05)
    assert times[2] / times[1] == pytest.approx(4.0, rel=0.05)
    # ...and that extra time is spent accumulating irreversible damage.
    assert np.all(np.diff(indices) < 0.0), f"ARI at matched uptake: {indices}"


# --- 3. the three design variables move service life the right way --------

def test_a_thicker_film_survives_longer():
    times = [
        run(_case(thickness=t)).time_to_ari_threshold
        for t in (50e-6, 100e-6, 200e-6, 400e-6)
    ]
    assert np.all(np.isfinite(times))
    assert np.all(np.diff(times) > 0.0), f"times {times}"


def test_a_hotter_exposure_fails_sooner():
    times = [
        run(_case(temperature_c=t)).time_to_ari_threshold
        for t in (25.0, 40.0, 60.0, 80.0)
    ]
    assert np.all(np.diff(times) < 0.0), f"times {times}"


def test_a_more_permeable_polymer_fails_sooner():
    times = [
        run(_case(diffusivity=d)).time_to_ari_threshold
        for d in (5e-14, 1e-13, 2e-13, 4e-13)
    ]
    assert np.all(np.diff(times) < 0.0), f"times {times}"


# --- 4. the thickness scaling depends on which mechanism binds ------------

def test_thickness_scaling_is_quadratic_when_wetting_is_the_bottleneck():
    """If adhesion collapses as soon as the bondline wets, service life is the
    wetting time, which scales as L**2/D."""
    times = [
        run(_case(thickness=t, threshold=0.8, adhesion=WETTING_LIMITED)).time_to_ari_threshold
        for t in (100e-6, 200e-6, 400e-6)
    ]
    assert times[1] / times[0] == pytest.approx(4.0, rel=0.03)
    assert times[2] / times[1] == pytest.approx(4.0, rel=0.03)


def test_thickness_buys_much_less_when_hydrolysis_is_the_bottleneck():
    """A design lesson the framework should be able to deliver.

    When slow chemical attack sets the life rather than wetting, doubling the
    film thickness no longer quadruples the life --- it barely helps, because the
    clock starts running as soon as any water arrives at the interface. Adding
    thickness is the wrong remedy in this regime; a better coupling agent is the
    right one.
    """
    times = [
        run(_case(thickness=t, adhesion=HYDROLYSIS_LIMITED)).time_to_ari_threshold
        for t in (100e-6, 200e-6, 400e-6)
    ]
    ratios = [times[1] / times[0], times[2] / times[1]]
    assert all(ratio > 1.0 for ratio in ratios), "thickness must still help a little"
    assert all(ratio < 2.5 for ratio in ratios), f"expected weak scaling, got {ratios}"


def test_the_binding_mechanism_is_reported_correctly_in_each_regime():
    wetting = run(_case(threshold=0.8, adhesion=WETTING_LIMITED))
    hydrolysis = run(_case(adhesion=HYDROLYSIS_LIMITED))
    assert wetting.dominant_mechanism == "plasticization"
    assert hydrolysis.dominant_mechanism == "hydrolysis"


# --- 5. geometry ----------------------------------------------------------

def test_a_free_film_fails_sooner_than_a_bonded_one_of_the_same_thickness():
    """Wetted on both faces, the mid-plane is only half a thickness from air."""
    bonded = run(_case(exposure="one_sided")).time_to_ari_threshold
    free = run(_case(exposure="two_sided")).time_to_ari_threshold
    assert free < bonded


def test_the_bondline_always_lags_the_film_average():
    result = run(_case())
    assert np.all(result.interface_normalized <= result.uptake_normalized + 1e-12)
    early = _at_fourier(result, 0.02)
    assert result.interface_normalized[early] < 0.5 * result.uptake_normalized[early]


def test_driving_adhesion_off_the_film_average_would_overstate_early_damage():
    """Why the framework tracks the interface, stated as a numerical difference."""
    result = run(_case())
    model = result.case.adhesion
    honest = result.adhesion
    naive = model.evaluate(
        result.time,
        result.uptake_normalized,          # the wrong driver, deliberately
        result.case.condition.temperature_k,
        result.saturation_pct,
        result.case.polymer.glass_transition_k,
    )
    early = _at_fourier(result, 0.02)
    assert naive.index[early] < honest.index[early] - 0.01
    assert naive.time_to_index(0.9) < honest.time_to_index(0.9)


# --- 6. irreversibility and history dependence ---------------------------

def test_temperature_acts_on_adhesion_by_a_route_independent_of_diffusion():
    """Raising temperature at frozen transport still costs adhesion.

    With the activation energy for diffusion set to zero, D no longer depends on
    temperature, so the entire uptake history is temperature-independent. ARI
    must still fall with temperature, through the wet-Tg gap and the hydrolysis
    rate. This separates the two channels the design deliberately keeps apart.
    """
    results = []
    for temperature_c in (30.0, 60.0, 85.0):
        case = _case(temperature_c=temperature_c)
        # Freeze both temperature channels in transport: D via the activation
        # energy, and saturation via the sorption enthalpy on the isotherm.
        frozen = replace(
            case.polymer,
            activation_energy=0.0,
            isotherm=replace(case.polymer.isotherm, enthalpy_sorption=0.0),
        )
        results.append(run(replace(case, polymer=frozen)))

    reference = results[0]
    for result in results[1:]:
        assert result.uptake_normalized == pytest.approx(
            reference.uptake_normalized, abs=1e-9
        ), "transport must be frozen for this test to isolate the other channel"
    finals = [result.final_index for result in results]
    assert np.all(np.diff(finals) < 0.0), f"final ARI {finals}"


def test_a_dry_film_retains_full_adhesion_indefinitely():
    """No spontaneous decay: with essentially no water there is nothing to degrade.

    This is the test that catches driving degradation off *normalized* moisture.
    Normalized bondline moisture reaches 1 here just as it does at 90% RH,
    because it is normalized against this exposure's own saturation --- which is
    5e-5 wt%. Only a model driven by absolute water content gets this right.
    """
    case = _case()
    condition = replace(case.condition, relative_humidity=1e-6)
    result = run(replace(case, condition=condition))
    assert result.interface_normalized[-1] == pytest.approx(1.0, abs=1e-3), (
        "normalized moisture should still saturate; that is the point"
    )
    assert result.saturation_pct < 1e-3
    assert result.final_index > 0.995


def test_lowering_humidity_lengthens_service_life():
    """Humidity must be a real design variable, not a normalization constant."""
    finals, times = [], []
    for humidity in (0.2, 0.4, 0.6, 0.8, 0.95):
        case = _case()
        result = run(replace(case, condition=replace(
            case.condition, relative_humidity=humidity)))
        finals.append(result.final_index)
        times.append(result.time_to_ari_threshold)
    assert np.all(np.diff(finals) < 0.0), f"final ARI by humidity: {finals}"
    assert np.all(np.diff(times) < 0.0), f"t_ARI by humidity: {times}"
    # And the effect must be large enough to matter for a design decision.
    assert times[0] > 3.0 * times[-1]


def test_drying_recovers_only_the_reversible_part_of_the_loss():
    """Cyclic and steady exposure must be distinguishable."""
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

    assert result.uptake_normalized[-1] < 0.05, "the film must actually dry out"
    assert result.adhesion.plasticization[-1] == pytest.approx(1.0, abs=1e-3)
    assert result.adhesion.thermal[-1] == pytest.approx(1.0, abs=1e-3)
    # What remains missing is exactly the irreversible part.
    assert result.final_index < 0.95
    assert result.final_index == pytest.approx(result.adhesion.hydrolysis[-1], abs=1e-3)


def _cycled(duty, n_cycles, days_total=120.0):
    """A wet/dry cycled run with hydrolysis slow enough to stay unsaturated.

    The default hydrolysis rate drives damage to 1 within this exposure, which
    pins ARI at its floor and hides any difference between schedules. These
    tests need the sensitive part of the curve.
    """
    from hygroadh.diffusion.fd import Schedule

    case = _case(duration_days=days_total, n_times=3000,
                 adhesion=dict(hydrolysis_rate_ref=4e-9))
    condition = replace(
        case.condition,
        humidity_schedule=Schedule.cycle(
            high=0.85, low=0.0, period=from_days(days_total / n_cycles),
            n_cycles=n_cycles, duty=duty, ramp=from_days(0.25),
        ),
        time_spacing="linear",
    )
    return run(replace(case, condition=condition))


def test_more_time_spent_wet_causes_more_irreversible_damage():
    """Duty fraction is the variable that matters in cyclic exposure."""
    damages, finals = [], []
    for duty in (0.2, 0.4, 0.6, 0.8):
        result = _cycled(duty, n_cycles=4)
        damages.append(float(result.adhesion.damage[-1]))
        finals.append(result.final_index)
    assert np.all(np.diff(damages) > 0.0), f"damage by duty: {damages}"
    assert np.all(np.diff(finals) < 0.0), f"final ARI by duty: {finals}"


def test_chopping_the_same_wet_time_into_more_cycles_barely_matters():
    """A design insight worth pinning: total wet time dominates, not cycle count.

    At a fixed duty fraction the accumulated damage moves by only a couple of
    percent as the same wet time is split into 1 to 16 cycles. It rises slightly,
    because short dry legs do not fully dry the bondline, so the time-averaged
    interfacial water content is a little higher. The practical consequence is
    that an accelerated test can compress cycles without much distorting the
    damage it produces --- provided the duty fraction is preserved.
    """
    damages = [float(_cycled(0.5, n).adhesion.damage[-1]) for n in (1, 2, 4, 8, 16)]
    spread = (max(damages) - min(damages)) / float(np.mean(damages))
    assert spread < 0.10, f"expected weak dependence on cycle count, got {damages}"
    assert np.all(np.diff(damages) > 0.0), (
        "incomplete drying should make frequent cycling marginally worse"
    )


def test_the_shipped_cycling_example_visibly_ratchets():
    """The worked cycling config must keep demonstrating what it is there to show.

    Its whole purpose is the irreversible mechanism: damage should step up on
    each wet leg and hold through each dry one, ending well short of saturation.
    A faster hydrolysis rate saturates the damage in the first cycle and the
    example silently stops illustrating anything.
    """
    from hygroadh.config import load_configuration

    result = run(load_configuration("configs/humidity_cycling.yaml").case)
    damage = result.adhesion.damage
    per_cycle = [
        float(np.interp(day * 86400.0, result.time, damage))
        for day in (8, 16, 24, 32, 40)
    ]
    assert np.all(np.diff(per_cycle) > 0.02), f"damage should step up: {per_cycle}"
    assert per_cycle[-1] < 0.9, "damage must stay off its ceiling to remain legible"
    # The film ends dry, so anything ARI has lost is irreversible by construction.
    assert result.uptake_normalized[-1] < 0.15
    assert result.adhesion.plasticization[-1] > 0.95
    assert result.final_index < 0.85
