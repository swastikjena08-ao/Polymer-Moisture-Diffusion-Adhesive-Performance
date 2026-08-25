"""Tests for CSV and JSON output."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from hygroadh import io_data, sweep
from hygroadh.config import load_configuration
from hygroadh.simulate import run
from hygroadh.units import to_kelvin


@pytest.fixture(scope="module")
def result():
    return run(load_configuration("configs/epoxy_on_steel.yaml").case)


def test_history_csv_has_a_row_per_sample_and_the_declared_columns(tmp_path, result):
    path = io_data.write_history_csv(tmp_path / "history.csv", result)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == list(io_data.HISTORY_COLUMNS)
    assert len(rows) == result.time.size
    assert float(rows[0]["time_s"]) == 0.0
    assert float(rows[-1]["ari"]) == pytest.approx(result.final_index, rel=1e-6)


def test_history_csv_round_trips_through_the_reader(tmp_path, result):
    path = io_data.write_history_csv(tmp_path / "history.csv", result)
    time, uptake = io_data.read_uptake_csv(path)
    assert time == pytest.approx(result.time, rel=1e-8)
    assert uptake == pytest.approx(result.uptake_normalized, rel=1e-8, abs=1e-12)


def test_history_csv_creates_missing_directories(tmp_path, result):
    path = io_data.write_history_csv(tmp_path / "deep" / "nested" / "h.csv", result)
    assert path.is_file()


def test_profile_csv_is_tidy_with_one_row_per_time_and_depth(tmp_path, result):
    path = io_data.write_profile_csv(tmp_path / "profile.csv", result)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    depth_count = result.transport.depth.size
    assert len(rows) == result.time.size * depth_count
    assert set(rows[0]) == {"time_s", "time_days", "depth_um", "c_over_csat",
                            "moisture_pct"}
    # Moisture in wt% must be the normalized value times saturation.
    row = rows[len(rows) // 2]
    assert float(row["moisture_pct"]) == pytest.approx(
        float(row["c_over_csat"]) * result.saturation_pct, rel=1e-6
    )


def test_profile_csv_refuses_a_result_that_has_no_profile(tmp_path):
    case = load_configuration("configs/epoxy_on_steel.yaml").case
    scalar_only = run(case.without_profile())
    with pytest.raises(ValueError, match="store_profile=False"):
        io_data.write_profile_csv(tmp_path / "p.csv", scalar_only)


def test_sweep_csv_has_a_row_per_grid_point_with_axes_first(tmp_path):
    case = load_configuration("configs/epoxy_on_steel.yaml").case
    result = sweep.run_sweep(case, {
        "thickness": [1e-4, 2e-4], "temperature_k": to_kelvin(np.array([30.0, 50.0])),
    })
    path = io_data.write_sweep_csv(tmp_path / "sweep.csv", result)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    header = list(rows[0].keys())
    assert header[0] == "thickness" and header[1] == "temperature_k"
    assert "final_ari" in header and "dominant_mechanism" in header


def test_summary_json_replaces_infinities_with_a_readable_marker(tmp_path):
    path = io_data.write_summary_json(tmp_path / "s.json", {
        "time_to_ari_threshold_s": math.inf,
        "final_ari": 0.5,
        "nested": {"also": -math.inf, "list": [1.0, math.nan]},
        "count": np.int64(4),
    })
    # Strict parsing: would raise on a bare Infinity token.
    payload = json.loads(path.read_text())
    assert payload["time_to_ari_threshold_s"] == "not reached"
    assert payload["nested"]["also"] == "not reached"
    assert payload["nested"]["list"][1] == "not reached"
    assert payload["final_ari"] == 0.5
    assert payload["count"] == 4


def test_summary_json_of_a_real_run_is_strictly_parseable(tmp_path, result):
    path = io_data.write_summary_json(tmp_path / "s.json", result.summary())
    assert json.loads(path.read_text())["dominant_mechanism"] in (
        "plasticization", "thermal", "hydrolysis"
    )


def test_reader_reports_missing_files_columns_and_short_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        io_data.read_uptake_csv(tmp_path / "absent.csv")

    wrong = tmp_path / "wrong.csv"
    wrong.write_text("a,b\n1,2\n3,4\n")
    with pytest.raises(ValueError, match="missing column"):
        io_data.read_uptake_csv(wrong)

    short = tmp_path / "short.csv"
    short.write_text("time_s,uptake_normalized\n0,0\n")
    with pytest.raises(ValueError, match="at least two data rows"):
        io_data.read_uptake_csv(short)

    bad = tmp_path / "bad.csv"
    bad.write_text("time_s,uptake_normalized\n0,0\n1,notanumber\n")
    with pytest.raises(ValueError, match="line 3"):
        io_data.read_uptake_csv(bad)


def test_reader_accepts_custom_column_names(tmp_path):
    path = tmp_path / "gravimetric.csv"
    path.write_text("hours,mass_gain\n0,0\n1,0.4\n2,0.7\n")
    time, uptake = io_data.read_uptake_csv(path, "hours", "mass_gain")
    assert time == pytest.approx([0.0, 1.0, 2.0])
    assert uptake == pytest.approx([0.0, 0.4, 0.7])
