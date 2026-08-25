"""End-to-end tests for the command-line interface."""

from __future__ import annotations

import json

import pytest

from hygroadh import cli, dashboard

CONFIG = "configs/epoxy_on_steel.yaml"
BASELINE = "configs/baseline.yaml"
CYCLING = "configs/humidity_cycling.yaml"


def test_info_prints_resolved_parameters(capsys):
    assert cli.main(["info", CONFIG]) == 0
    out = capsys.readouterr().out
    assert "epoxy_on_steel" in out
    assert "D at temperature" in out
    assert "wet Tg at saturation" in out
    assert "sweep axes" in out


def test_run_prints_a_summary_and_the_standing_note(capsys):
    assert cli.main(["run", CONFIG]) == 0
    out = capsys.readouterr().out
    assert "final ARI" in out
    assert "binding mechanism" in out
    assert "time to ARI 0.70" in out
    # The caveat must travel with the numbers.
    assert "phenomenological model" in out


def test_run_writes_the_expected_artifacts(tmp_path, capsys):
    assert cli.main(["run", CONFIG, "--outdir", str(tmp_path)]) == 0
    capsys.readouterr()
    for name in ("history.csv", "summary.json", "profile.csv"):
        assert (tmp_path / name).is_file(), f"missing {name}"
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["name"] == "epoxy_on_steel"
    assert 0.0 <= summary["final_ari"] <= 1.0


def test_run_handles_the_cycling_config_through_finite_volumes(capsys):
    assert cli.main(["run", CYCLING]) == 0
    out = capsys.readouterr().out
    assert "finite_volume" in out


def test_sweep_reports_the_grid_and_writes_a_table(tmp_path, capsys):
    assert cli.main(["sweep", CONFIG, "--outdir", str(tmp_path), "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "swept axes: thickness, temperature_k" in out
    assert "shape (5, 5)" in out
    assert "fastest failure" in out
    table = (tmp_path / "sweep.csv").read_text().splitlines()
    assert len(table) == 26, "header plus 25 grid points"


def test_sweep_without_a_sweep_section_explains_how_to_add_one(capsys):
    assert cli.main(["sweep", BASELINE]) == 2
    err = capsys.readouterr().err
    assert "no sweep section" in err
    assert "thickness_um:" in err, "the message should show a usable example"


def test_sweep_reports_progress_unless_quiet(capsys):
    cli.main(["sweep", CONFIG])
    assert "points" in capsys.readouterr().err
    cli.main(["sweep", CONFIG, "--quiet"])
    assert "points" not in capsys.readouterr().err


def test_sensitivity_ranks_the_design_variables(capsys):
    assert cli.main(["sensitivity", CONFIG]) == 0
    out = capsys.readouterr().out
    assert "local elasticities" in out
    assert "thickness" in out and "diffusivity" in out
    assert "most influential" in out


def test_sensitivity_can_add_morris_screening(capsys):
    assert cli.main(["sensitivity", CONFIG, "--morris", "--trajectories", "4"]) == 0
    out = capsys.readouterr().out
    assert "Morris screening, 4 trajectories" in out
    assert "mu*" in out and "interacts" in out


def test_sensitivity_accepts_an_alternative_response(capsys):
    assert cli.main(["sensitivity", CONFIG, "--response", "final_ari"]) == 0
    assert "final_ari" in capsys.readouterr().out


def test_sensitivity_rejects_an_unknown_response():
    with pytest.raises(SystemExit):
        cli.main(["sensitivity", CONFIG, "--response", "vibes"])


def test_missing_config_exits_with_an_error(capsys):
    assert cli.main(["run", "configs/does-not-exist.yaml"]) == 2
    assert "error:" in capsys.readouterr().err


def test_invalid_config_exits_with_a_located_error(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "film:\n  thickness_um: 100\n"
        "polymer:\n  diffusivity_ref: 5e-13\n  tg_dry_c: 100\n"
        "  isotherm:\n    m_ref_pct: 2.0\n"
        "exposure:\n  temperature_c: 40\n  duration_days: 30\n"
        "  relative_humidity: 4.0\n"
    )
    assert cli.main(["run", str(bad)]) == 2
    assert "exposure: relative_humidity" in capsys.readouterr().err


def test_version_and_help_are_available(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert "hygroadh" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["--help"])


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        cli.main([])


def test_serve_passes_config_defaults_through(monkeypatch, capsys):
    captured = {}

    def fake_serve(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(dashboard, "serve", fake_serve)
    assert cli.main(["serve", CONFIG, "--port", "9999", "--no-browser"]) == 0
    assert captured["port"] == 9999
    assert captured["open_browser"] is False
    assert captured["defaults"]["thickness_um"] == pytest.approx(200.0)


def test_serve_warns_about_settings_it_cannot_show(monkeypatch, capsys):
    monkeypatch.setattr(dashboard, "serve", lambda **_: None)
    assert cli.main(["serve", CYCLING, "--no-browser"]) == 0
    err = capsys.readouterr().err
    assert "no control for" in err
    assert "humidity schedule" in err


def test_serve_without_a_config_uses_built_in_defaults(monkeypatch):
    captured = {}
    monkeypatch.setattr(dashboard, "serve", lambda **kwargs: captured.update(kwargs))
    assert cli.main(["serve", "--no-browser"]) == 0
    assert captured["defaults"] is None
