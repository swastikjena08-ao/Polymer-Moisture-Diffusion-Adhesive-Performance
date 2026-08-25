"""Tests for configuration loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hygroadh.config import build_configuration, load_configuration, load_raw
from hygroadh.units import KELVIN_OFFSET, ConfigError, from_days

MINIMAL = {
    "film": {"thickness_um": 100},
    "polymer": {
        "diffusivity_ref": 5e-13,
        "tg_dry_c": 100,
        "isotherm": {"m_ref_pct": 2.0},
    },
    "exposure": {"temperature_c": 40, "duration_days": 30},
}


def _with(section: str, **updates) -> dict:
    """A copy of MINIMAL with one section updated."""
    import copy

    data = copy.deepcopy(MINIMAL)
    data.setdefault(section, {}).update(updates)
    return data


# --- happy path ------------------------------------------------------------

def test_minimal_config_builds_a_usable_case():
    configuration = build_configuration(MINIMAL)
    case = configuration.case
    assert case.thickness == pytest.approx(100e-6)
    assert case.condition.temperature_k == pytest.approx(40 + KELVIN_OFFSET)
    assert case.condition.duration == pytest.approx(from_days(30))
    assert case.polymer.tg_dry_k == pytest.approx(100 + KELVIN_OFFSET)
    assert configuration.sweep.is_empty


def test_shipped_configs_all_load():
    """The examples must stay in step with the loader."""
    for path in sorted(Path("configs").glob("*.yaml")):
        configuration = load_configuration(path)
        assert configuration.case.name
        assert configuration.case.thickness > 0


def test_the_worked_example_defines_a_sweep():
    configuration = load_configuration("configs/epoxy_on_steel.yaml")
    axes = configuration.sweep
    assert not axes.is_empty
    assert axes.thickness is not None and axes.thickness.size == 5
    assert axes.temperature_k is not None and axes.temperature_k.size == 5
    assert axes.thickness[0] == pytest.approx(50e-6)


def test_the_cycling_example_builds_a_humidity_schedule():
    configuration = load_configuration("configs/humidity_cycling.yaml")
    condition = configuration.case.condition
    assert condition.humidity_schedule is not None
    assert condition.humidity_schedule.maximum == pytest.approx(0.95)
    assert condition.resolved_spacing == "linear"
    # A schedule means the analytical solution cannot be used.
    assert configuration.case.requires_finite_volume
    assert configuration.case.resolved_solver == "fd"


# --- unit-friendly keys ----------------------------------------------------

@pytest.mark.parametrize(
    "key, value, expected",
    [("thickness", 2e-4, 2e-4), ("thickness_mm", 0.2, 2e-4), ("thickness_um", 200, 2e-4)],
)
def test_thickness_accepts_metres_millimetres_or_micrometres(key, value, expected):
    data = {**MINIMAL, "film": {key: value}}
    assert build_configuration(data).case.thickness == pytest.approx(expected)


@pytest.mark.parametrize(
    "key, value, expected",
    [("duration", 3600.0, 3600.0), ("duration_hours", 2, 7200.0),
     ("duration_days", 1, 86400.0)],
)
def test_duration_accepts_seconds_hours_or_days(key, value, expected):
    data = {**MINIMAL, "exposure": {"temperature_c": 40, key: value}}
    assert build_configuration(data).case.condition.duration == pytest.approx(expected)


def test_temperature_accepts_celsius_or_kelvin():
    celsius = build_configuration(_with("exposure", temperature_c=60))
    data = {**MINIMAL, "exposure": {"temperature_k": 333.15, "duration_days": 30}}
    kelvin = build_configuration(data)
    assert celsius.case.condition.temperature_k == pytest.approx(
        kelvin.case.condition.temperature_k
    )


def test_giving_both_temperature_units_is_an_error():
    data = {**MINIMAL, "exposure": {"temperature_c": 60, "temperature_k": 333.15,
                                    "duration_days": 30}}
    with pytest.raises(ConfigError, match="exactly one"):
        build_configuration(data)


def test_giving_two_thickness_units_is_an_error():
    data = {**MINIMAL, "film": {"thickness_um": 200, "thickness_mm": 0.2}}
    with pytest.raises(ConfigError, match="exactly one"):
        build_configuration(data)


def test_giving_two_duration_units_is_an_error():
    data = {**MINIMAL, "exposure": {"temperature_c": 40, "duration_days": 1,
                                    "duration_hours": 24}}
    with pytest.raises(ConfigError, match="exactly one"):
        build_configuration(data)


# --- strictness ------------------------------------------------------------

def test_unknown_keys_are_rejected_rather_than_ignored():
    """A silently ignored typo would leave the run on a default value."""
    data = _with("polymer", tg_depresion_per_pct=15)
    with pytest.raises(ConfigError, match="unknown key"):
        build_configuration(data)


@pytest.mark.parametrize("section", ["film", "polymer", "exposure"])
def test_required_sections_are_reported_by_name(section):
    data = {key: value for key, value in MINIMAL.items() if key != section}
    with pytest.raises(ConfigError, match=section):
        build_configuration(data)


def test_missing_required_keys_are_reported_with_their_path():
    data = {**MINIMAL, "polymer": {"tg_dry_c": 100, "isotherm": {"m_ref_pct": 2.0}}}
    with pytest.raises(ConfigError, match="polymer.diffusivity_ref"):
        build_configuration(data)
    data = {**MINIMAL, "polymer": {"diffusivity_ref": 5e-13, "tg_dry_c": 100}}
    with pytest.raises(ConfigError, match="polymer.isotherm"):
        build_configuration(data)


def test_non_numeric_values_are_reported_with_their_path():
    with pytest.raises(ConfigError, match="film.thickness_um"):
        build_configuration({**MINIMAL, "film": {"thickness_um": "thick"}})


def test_physically_invalid_values_surface_as_config_errors_naming_the_section():
    """A bad value must say which section to edit, not just what is wrong."""
    with pytest.raises(ConfigError, match="thickness"):
        build_configuration({**MINIMAL, "film": {"thickness_um": -5}})
    with pytest.raises(ConfigError, match=r"exposure: relative_humidity"):
        build_configuration(_with("exposure", relative_humidity=2.0))
    with pytest.raises(ConfigError, match=r"polymer: .*tg_floor_k"):
        build_configuration(_with("polymer", tg_dry_c=20, tg_floor_c=80))
    with pytest.raises(ConfigError, match=r"polymer\.isotherm: .*m_ref_pct"):
        build_configuration({**MINIMAL, "polymer": {
            "diffusivity_ref": 5e-13, "tg_dry_c": 100,
            "isotherm": {"m_ref_pct": -1.0}}})
    with pytest.raises(ConfigError, match=r"adhesion: "):
        build_configuration({**MINIMAL, "adhesion": {"plasticization_floor": 1.5}})


def test_a_section_that_is_not_a_mapping_is_reported():
    with pytest.raises(ConfigError, match="must be a mapping"):
        build_configuration({**MINIMAL, "polymer": [1, 2, 3]})


def test_non_whole_integer_counts_are_rejected():
    with pytest.raises(ConfigError, match="whole number"):
        build_configuration(_with("exposure", n_times=12.5))


# --- humidity schedules ----------------------------------------------------

def test_knot_schedule_is_built_with_time_units():
    data = _with("exposure", humidity_schedule={
        "type": "knots", "time_unit": "d",
        "times": [0, 5, 5.01, 10], "relative_humidity": [0.9, 0.9, 0.1, 0.1],
    })
    schedule = build_configuration(data).case.condition.humidity_schedule
    assert schedule is not None
    assert schedule(0.0) == pytest.approx(0.9)
    assert schedule(from_days(10)) == pytest.approx(0.1)


def test_cycle_schedule_alternates():
    data = _with("exposure", humidity_schedule={
        "type": "cycle", "time_unit": "d", "high": 0.9, "low": 0.2,
        "period": 4, "n_cycles": 3,
    })
    schedule = build_configuration(data).case.condition.humidity_schedule
    assert schedule(0.0) == pytest.approx(0.9)
    assert schedule(from_days(3)) == pytest.approx(0.2)


def test_bad_schedules_are_reported_clearly():
    with pytest.raises(ConfigError, match="must be 'knots' or 'cycle'"):
        build_configuration(_with("exposure", humidity_schedule={"type": "sine"}))
    with pytest.raises(ConfigError, match="time_unit"):
        build_configuration(_with("exposure", humidity_schedule={
            "type": "cycle", "time_unit": "fortnights", "high": 0.9, "low": 0.1,
            "period": 4}))
    with pytest.raises(ConfigError, match="needs both"):
        build_configuration(_with("exposure", humidity_schedule={
            "type": "knots", "times": [0, 1]}))
    with pytest.raises(ConfigError, match="ascending"):
        build_configuration(_with("exposure", humidity_schedule={
            "type": "knots", "times": [5, 0], "relative_humidity": [0.5, 0.5]}))


# --- sweep section ---------------------------------------------------------

def test_sweep_axes_are_converted_to_si():
    data = {**MINIMAL, "sweep": {"thickness_um": [50, 100], "temperature_c": [25, 45]}}
    axes = build_configuration(data).sweep
    assert axes.thickness == pytest.approx([50e-6, 100e-6])
    assert axes.temperature_k == pytest.approx([25 + KELVIN_OFFSET, 45 + KELVIN_OFFSET])


def test_sweep_diffusivity_scale_multiplies_the_polymer_value():
    data = {**MINIMAL, "sweep": {"diffusivity_scale": [0.5, 1.0, 2.0]}}
    axes = build_configuration(data).sweep
    assert axes.diffusivity == pytest.approx([2.5e-13, 5e-13, 1e-12])


def test_sweep_rejects_scalar_thickness_and_conflicting_keys():
    with pytest.raises(ConfigError, match="not a sweep axis"):
        build_configuration({**MINIMAL, "sweep": {"thickness": 1e-4}})
    with pytest.raises(ConfigError, match="both temperature_c and temperature_k"):
        build_configuration({**MINIMAL, "sweep": {"temperature_c": [20],
                                                  "temperature_k": [300]}})
    with pytest.raises(ConfigError, match="both diffusivity"):
        build_configuration({**MINIMAL, "sweep": {"diffusivity": [1e-13],
                                                  "diffusivity_scale": [1.0]}})


def test_sweep_rejects_non_numeric_and_empty_lists():
    with pytest.raises(ConfigError, match="only numbers"):
        build_configuration({**MINIMAL, "sweep": {"thickness_um": [50, "thick"]}})
    with pytest.raises(ConfigError, match="non-empty"):
        build_configuration({**MINIMAL, "sweep": {"thickness_um": []}})


# --- file loading ----------------------------------------------------------

def test_json_configs_load_identically_to_yaml(tmp_path):
    path = tmp_path / "case.json"
    path.write_text(json.dumps(MINIMAL))
    assert load_configuration(path).case.thickness == pytest.approx(100e-6)


def test_missing_files_and_unknown_extensions_are_reported(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_raw(tmp_path / "absent.yaml")
    other = tmp_path / "case.txt"
    other.write_text("nope")
    with pytest.raises(ConfigError, match="unsupported config extension"):
        load_raw(other)


def test_malformed_files_are_reported_as_config_errors(tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_raw(bad_json)
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("a:\n  - b\n c: broken\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_raw(bad_yaml)


def test_an_empty_file_is_reported_as_a_missing_section(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    with pytest.raises(ConfigError, match="film section is required"):
        load_configuration(path)
