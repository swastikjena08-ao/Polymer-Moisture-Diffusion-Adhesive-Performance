"""Tests for the dashboard server: payload structure and real HTTP requests."""

from __future__ import annotations

import json
import math
import re
import threading
import urllib.error
import urllib.request
from dataclasses import replace

import numpy as np
import pytest

from hygroadh import dashboard
from hygroadh.config import load_configuration
from hygroadh.units import KELVIN_OFFSET, HygroadhError


@pytest.fixture(scope="module")
def server():
    """A dashboard bound to an ephemeral port, so tests never collide."""
    bound = dashboard.build_dashboard(host="127.0.0.1", port=0)
    thread = threading.Thread(target=bound.serve_forever, daemon=True)
    thread.start()
    try:
        yield bound
    finally:
        bound.shutdown()
        bound.close()
        thread.join(timeout=5)


def _get(server, path):
    with urllib.request.urlopen(server.url.rstrip("/") + path, timeout=20) as response:
        return response.status, response.read(), dict(response.headers)


def _post(server, path, payload):
    request = urllib.request.Request(
        server.url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read()
            kind = response.headers.get("Content-Type", "")
            if "json" in kind:
                return response.status, json.loads(body), response.headers
            return response.status, body.decode(), response.headers
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read()), error.headers


def _post_bytes(server, path, body):
    request = urllib.request.Request(
        server.url.rstrip("/") + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, json.loads(response.read()), response.headers
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read()), error.headers


# --- building a case from parameters -------------------------------------

def test_default_params_build_a_valid_case():
    case = dashboard.case_from_params({})
    assert case.thickness == pytest.approx(200e-6)
    assert case.condition.temperature_k == pytest.approx(60.0 + KELVIN_OFFSET)
    assert case.condition.relative_humidity == pytest.approx(0.85)
    assert case.polymer.activation_energy == pytest.approx(50e3), "kJ/mol -> J/mol"
    assert case.polymer.temperature_ref_k == pytest.approx(25.0 + KELVIN_OFFSET)
    assert case.criterion.basis == "normalized"


def test_humidity_is_taken_as_a_percentage():
    case = dashboard.case_from_params({"relative_humidity_pct": 40.0})
    assert case.condition.relative_humidity == pytest.approx(0.40)


def test_humidity_outside_zero_to_one_hundred_percent_is_rejected():
    for bad in (0.0, -5.0, 120.0):
        with pytest.raises(HygroadhError, match="relative humidity"):
            dashboard.case_from_params({"relative_humidity_pct": bad})


def test_activation_energy_is_read_in_kilojoules():
    case = dashboard.case_from_params({"ea_kj": 65.0})
    assert case.polymer.activation_energy == pytest.approx(65e3)


def test_unknown_parameters_are_rejected_rather_than_ignored():
    """A stale or misspelled control name must fail, not silently use a default."""
    with pytest.raises(HygroadhError, match="unknown parameter"):
        dashboard.case_from_params({"knockdown_per_pct": 0.25})
    with pytest.raises(HygroadhError, match="thicknes_um"):
        dashboard.case_from_params({"thicknes_um": 200.0})


def test_non_numeric_and_non_finite_parameters_are_rejected():
    with pytest.raises(HygroadhError, match="thickness_um"):
        dashboard.case_from_params({"thickness_um": "thick"})
    with pytest.raises(HygroadhError, match="finite"):
        dashboard.case_from_params({"thickness_um": float("inf")})


def test_invalid_choices_are_reported_by_name():
    for key, value in (("exposure", "sideways"), ("threshold_basis", "percent"),
                       ("duration_mode", "guess")):
        with pytest.raises(HygroadhError, match=key):
            dashboard.case_from_params({key: value})


# --- the automatic simulation window ------------------------------------

def test_the_automatic_window_scales_with_the_diffusion_time():
    """A fixed default window would be absurd: l**2/D spans nine decades here."""
    thin = dashboard.case_from_params({"thickness_um": 50.0}).condition.duration
    thick = dashboard.case_from_params({"thickness_um": 400.0}).condition.duration
    assert thick / thin == pytest.approx(64.0, rel=1e-6)
    fast = dashboard.case_from_params({"d_ref": 1e-12}).condition.duration
    slow = dashboard.case_from_params({"d_ref": 1e-13}).condition.duration
    assert slow / fast == pytest.approx(10.0, rel=1e-6)


def test_a_manual_window_is_used_verbatim():
    case = dashboard.case_from_params({"duration_mode": "manual",
                                       "duration_hours": 12.0})
    assert case.condition.duration == pytest.approx(12.0 * 3600.0)


def test_the_automatic_window_reaches_near_equilibrium():
    payload = dashboard.simulate_payload({})
    assert payload["history"]["uptake_normalized"][-1] > 0.999
    assert payload["study_window"] == "auto"


def test_studies_resize_their_window_per_point_when_automatic():
    """Otherwise thick or slow points report 'not reached' and answer nothing."""
    payload = dashboard.simulate_payload({})
    for study in ("thickness_study", "diffusivity_study"):
        times = payload[study]["time_to_threshold_s"]
        assert all(t is not None for t in times), f"{study} has unreached points"


def test_a_manual_window_may_legitimately_leave_points_unreached():
    payload = dashboard.simulate_payload({
        "duration_mode": "manual", "duration_hours": 1.0,
    })
    assert payload["study_window"] == "manual"
    times = payload["thickness_study"]["time_to_threshold_s"]
    assert any(t is None for t in times), "a 1 h window cannot wet an 800 um film"


# --- payload structure --------------------------------------------------

def test_payload_carries_every_section_the_page_reads():
    payload = dashboard.simulate_payload({})
    for key in ("inputs", "derived", "history", "profile", "temperature",
                "temperature_study", "thickness_study", "diffusivity_study",
                "sensitivity", "study_window", "elapsed_ms"):
        assert key in payload, f"payload missing {key}"
    for key in ("thickness_um", "temperature_c", "temperature_k",
                "relative_humidity_pct", "d_ref", "t_ref_c", "t_ref_k", "ea_kj",
                "threshold_basis", "threshold_value", "exposure", "duration_hours"):
        assert key in payload["inputs"], f"inputs missing {key}"
    for key in ("diffusivity", "saturation_pct", "threshold_pct",
                "threshold_normalized", "threshold_reachable", "time_to_threshold_s",
                "time_to_half_uptake_s", "characteristic_time_s", "solver"):
        assert key in payload["derived"], f"derived missing {key}"


def test_the_inputs_echo_makes_a_run_reproducible():
    """Every number needed to repeat the run must come back with the results."""
    params = {"thickness_um": 321.0, "temperature_c": 47.0, "ea_kj": 44.0,
              "d_ref": 3e-13, "t_ref_c": 20.0, "relative_humidity_pct": 61.0}
    inputs = dashboard.simulate_payload(params)["inputs"]
    assert inputs["thickness_um"] == pytest.approx(321.0)
    assert inputs["temperature_c"] == pytest.approx(47.0)
    assert inputs["temperature_k"] == pytest.approx(47.0 + KELVIN_OFFSET)
    assert inputs["ea_kj"] == pytest.approx(44.0)
    assert inputs["d_ref"] == pytest.approx(3e-13)
    assert inputs["t_ref_k"] == pytest.approx(20.0 + KELVIN_OFFSET)
    assert inputs["relative_humidity_pct"] == pytest.approx(61.0)


def test_the_diffusivity_reported_is_the_arrhenius_value_at_the_input_temperature():
    payload = dashboard.simulate_payload({"temperature_c": 60.0, "d_ref": 1e-13,
                                          "t_ref_c": 25.0, "ea_kj": 50.0})
    expected = 1e-13 * math.exp(
        -(50e3 / 8.314462618) * (1 / (60 + KELVIN_OFFSET) - 1 / (25 + KELVIN_OFFSET))
    )
    assert payload["derived"]["diffusivity"] == pytest.approx(expected, rel=1e-9)


def test_payload_is_strictly_json_serializable():
    """No Infinity or NaN tokens, which JSON.parse rejects outright."""
    payload = dashboard.simulate_payload({"duration_mode": "manual",
                                          "duration_hours": 0.05})
    json.dumps(payload, allow_nan=False)  # raises if any nan/inf survived


def test_an_unreachable_threshold_becomes_null_and_is_explained():
    payload = dashboard.simulate_payload({
        "threshold_basis": "wt_pct", "threshold_value": 3.0,
        "relative_humidity_pct": 50.0,
    })
    derived = payload["derived"]
    assert derived["threshold_reachable"] is False
    assert derived["time_to_threshold_s"] is None
    assert "can never be reached" in derived["threshold_note"]


def test_json_safe_maps_non_finite_values_to_null():
    cleaned = dashboard.json_safe({
        "a": math.inf, "b": -math.inf, "c": math.nan, "d": 1.5,
        "e": np.array([1.0, np.inf]), "f": np.int64(3), "g": np.True_,
    })
    assert cleaned["a"] is None and cleaned["b"] is None and cleaned["c"] is None
    assert cleaned["d"] == 1.5 and cleaned["e"] == [1.0, None]
    assert cleaned["f"] == 3 and cleaned["g"] is True


# --- profiles -----------------------------------------------------------

def test_profile_snapshots_are_labelled_and_ordered_in_time():
    profile = dashboard.simulate_payload({})["profile"]
    assert [s["label"] for s in profile["snapshots"]] == list(dashboard.SNAPSHOT_LABELS)
    times = [s["time_s"] for s in profile["snapshots"]]
    assert np.all(np.diff(times) > 0.0)
    # Later snapshots are wetter everywhere through the thickness.
    for earlier, later in zip(profile["snapshots"], profile["snapshots"][1:]):
        assert np.all(
            np.array(later["profile"]) >= np.array(earlier["profile"]) - 1e-9
        )


def test_profile_falls_from_the_wetted_face_to_the_far_face():
    profile = dashboard.simulate_payload({})["profile"]
    assert profile["depth_um"][0] == pytest.approx(0.0)
    assert profile["depth_um"][-1] == pytest.approx(200.0, rel=0.02)
    for snapshot in profile["snapshots"]:
        assert np.all(np.diff(snapshot["profile"]) <= 1e-9)


def test_slider_frames_cover_the_whole_window():
    profile = dashboard.simulate_payload({})["profile"]
    frames = profile["frames"]
    assert len(frames) == len(profile["frame_time_s"])
    assert len(frames) <= dashboard.PROFILE_SLIDER_FRAMES + 2
    assert profile["frame_time_s"][0] == pytest.approx(0.0)
    assert profile["frame_time_s"][-1] == pytest.approx(
        dashboard.simulate_payload({})["history"]["time_s"][-1]
    )
    assert len(frames[0]) == len(profile["depth_um"])


# --- the studies --------------------------------------------------------

def test_the_thickness_study_matches_its_own_l_squared_reference():
    """The reference is anchored, not fitted, so agreement is a real check."""
    study = dashboard.simulate_payload({})["thickness_study"]
    simulated = np.array(study["time_to_threshold_s"])
    reference = np.array(study["l_squared_reference_s"])
    assert simulated == pytest.approx(reference, rel=0.01)


def test_the_study_sample_points_are_centred_on_the_current_value():
    """A sweep must straddle the operating point, not drift away from it."""
    for thickness in (75.0, 200.0, 640.0):
        study = dashboard.simulate_payload({"thickness_um": thickness})["thickness_study"]
        points = study["thickness_um"]
        assert len(points) == dashboard.STUDY_POINTS
        assert points[len(points) // 2] == pytest.approx(thickness)
        assert points[0] == pytest.approx(thickness / dashboard.THICKNESS_STUDY_SPAN)
        assert points[-1] == pytest.approx(thickness * dashboard.THICKNESS_STUDY_SPAN)

    for d_ref in (1e-14, 1e-13, 5e-12):
        study = dashboard.simulate_payload({"d_ref": d_ref})["diffusivity_study"]
        points = study["d_ref"]
        assert points[len(points) // 2] == pytest.approx(d_ref)
        assert points[0] == pytest.approx(d_ref / dashboard.DIFFUSIVITY_STUDY_SPAN)


def test_the_default_thickness_sweep_reproduces_the_conventional_series():
    """At 200 um a factor-4 span lands exactly on 50/100/200/400/800 um."""
    study = dashboard.simulate_payload({})["thickness_study"]
    assert study["thickness_um"] == pytest.approx([50.0, 100.0, 200.0, 400.0, 800.0])


def test_study_points_are_geometric_and_symmetric():
    points = dashboard.study_points(200.0, 4.0, 5)
    assert points == pytest.approx([50.0, 100.0, 200.0, 400.0, 800.0])
    ratios = points[1:] / points[:-1]
    assert ratios == pytest.approx([ratios[0]] * len(ratios))


def test_the_diffusivity_study_is_inversely_proportional_to_d():
    study = dashboard.simulate_payload({})["diffusivity_study"]
    times = np.array(study["time_to_threshold_s"])
    assert np.all(np.diff(times) < 0.0)
    ratios = times[:-1] / times[1:]
    assert ratios == pytest.approx([ratios[0]] * len(ratios), rel=0.01)


def test_the_diffusivity_study_reports_both_d_ref_and_the_effective_d():
    """Sweeping D_ref while labelling the axis "D" would be wrong by the
    Arrhenius factor -- 8.3x at the default 60 degC."""
    payload = dashboard.simulate_payload({})
    study = payload["diffusivity_study"]
    factor = payload["derived"]["diffusivity"] / payload["inputs"]["d_ref"]
    assert factor > 5.0, "the default case should have a substantial Arrhenius factor"
    effective = np.array(study["diffusivity_at_temperature"])
    assert effective == pytest.approx(np.array(study["d_ref"]) * factor, rel=1e-9)


def test_equilibrium_uptake_is_an_input_and_scales_the_concentration_field():
    """Previously hardcoded at 2.4 wt% with no control; now settable."""
    for value in (1.0, 2.4, 6.0):
        payload = dashboard.simulate_payload({"m_ref_pct": value,
                                              "relative_humidity_pct": 100.0})
        assert payload["inputs"]["m_ref_pct"] == pytest.approx(value)
        assert payload["derived"]["saturation_pct"] == pytest.approx(value)


def test_equilibrium_uptake_does_not_change_a_normalized_threshold_time():
    """It scales the whole field, so a *relative* criterion is unaffected --- which
    is the property that makes the normalized basis a pure transport measure."""
    times = [
        dashboard.simulate_payload({"m_ref_pct": m})["derived"]["time_to_threshold_s"]
        for m in (1.0, 2.4, 6.0)
    ]
    assert times == pytest.approx([times[0]] * 3, rel=1e-6)


def test_equilibrium_uptake_does_change_an_absolute_threshold_time():
    times = [
        dashboard.simulate_payload({
            "m_ref_pct": m, "threshold_basis": "wt_pct", "threshold_value": 0.8,
        })["derived"]["time_to_threshold_s"]
        for m in (1.5, 3.0, 6.0)
    ]
    assert all(t is not None for t in times)
    assert np.all(np.diff(times) < 0.0), "a wetter film meets an absolute level sooner"


def test_the_temperature_study_spans_the_requested_range_in_ten_degree_steps():
    payload = dashboard.simulate_payload({"study_temp_min_c": 20.0,
                                          "study_temp_max_c": 80.0})
    rows = payload["temperature_study"]
    assert [r["temperature_c"] for r in rows] == pytest.approx(
        [20, 30, 40, 50, 60, 70, 80]
    )
    for row in rows:
        assert row["temperature_k"] == pytest.approx(row["temperature_c"] + KELVIN_OFFSET)
    diffusivity = [r["diffusivity"] for r in rows]
    times = [r["time_to_threshold_s"] for r in rows]
    assert np.all(np.diff(diffusivity) > 0.0), "D must rise with temperature"
    assert np.all(np.diff(times) < 0.0), "penetration time must fall"


def test_the_temperature_curves_carry_both_the_fine_and_the_swept_grids():
    temperature = dashboard.simulate_payload({})["temperature"]
    assert len(temperature["curve_temperature_c"]) == dashboard.TEMPERATURE_CURVE_POINTS
    assert len(temperature["penetration_temperature_c"]) == \
        dashboard.TEMPERATURE_PENETRATION_POINTS
    assert np.all(np.diff(temperature["curve_diffusivity"]) > 0.0)


def test_an_inverted_study_range_is_rejected():
    with pytest.raises(HygroadhError, match="must increase"):
        dashboard.simulate_payload({"study_temp_min_c": 80.0, "study_temp_max_c": 20.0})


def test_sensitivity_recovers_the_scaling_exponents():
    sensitivity = dashboard.simulate_payload({})["sensitivity"]
    assert sensitivity["response"] == "time_to_threshold"
    assert sensitivity["elasticities"]["thickness"] == pytest.approx(2.0, abs=0.02)
    assert sensitivity["elasticities"]["diffusivity"] == pytest.approx(-1.0, abs=0.02)
    assert sensitivity["elasticities"]["temperature_k"] < -5.0


def test_sensitivity_reports_an_explanation_instead_of_failing():
    payload = dashboard.simulate_payload({
        "threshold_basis": "wt_pct", "threshold_value": 3.0,
        "relative_humidity_pct": 50.0,
    })
    assert "error" in payload["sensitivity"]


# --- CSV export ---------------------------------------------------------

def test_summary_export_labels_inputs_and_calculated_values():
    text = dashboard.export_summary_csv({})
    assert "quantity,value,unit,kind" in text
    assert ",input" in text and ",calculated" in text
    assert "thickness_um" in text and "diffusivity" in text
    assert "m_ref_pct" in text, "equilibrium uptake is an input and must be recorded"
    assert "elasticity_thickness" in text
    # The provenance notes must travel with the numbers.
    assert "not measured adhesive strength" in text
    assert "model parameters" in text


def test_history_export_has_a_row_per_sample():
    text = dashboard.export_history_csv({})
    lines = text.strip().splitlines()
    payload = dashboard.simulate_payload({})
    assert len(lines) == len(payload["history"]["time_s"]) + 1
    assert lines[0].startswith("time_s,time_hours")


def test_studies_export_carries_all_three_sweeps():
    text = dashboard.export_studies_csv({})
    assert text.count("temperature,temperature") == 7
    assert text.count("thickness,thickness") == dashboard.STUDY_POINTS
    assert text.count("diffusivity,d_ref") == dashboard.STUDY_POINTS


def test_exports_render_unreachable_times_as_words_not_blanks():
    text = dashboard.export_summary_csv({
        "threshold_basis": "wt_pct", "threshold_value": 3.0,
        "relative_humidity_pct": 50.0,
    })
    assert "not reached" in text


# --- the page -----------------------------------------------------------

def test_page_injects_defaults_and_keeps_no_placeholder():
    page = dashboard.render_page().decode()
    assert "window.__HYGROADH_DEFAULTS__" not in page
    assert '"thickness_um": 200' in page
    assert "<title>" in page


def test_page_is_self_contained():
    """No external scripts, styles, fonts, or images to fetch."""
    page = dashboard.render_page().decode().lower()
    for forbidden in ("<link", "@import", "<script src", 'src="http', "src='http",
                      'href="http', "href='http", "url(http", "//cdn",
                      'fetch("http', "@font-face"):
        assert forbidden not in page, f"page references {forbidden!r}"
    assert 'fetch("/api/simulate"' in page


def test_page_states_the_research_question_and_the_scope_limit():
    page = " ".join(dashboard.render_page().decode().split())
    assert "How do polymer-film thickness, temperature, and water diffusivity" in page
    assert "predicts moisture transport, not measured bond strength" in page


def test_page_carries_the_assumptions_section():
    page = " ".join(dashboard.render_page().decode().split())
    for assumption in ("One-dimensional diffusion", "Fickian diffusion",
                       "Arrhenius temperature dependence", "No hydrolysis model",
                       "No plasticization model", "No glass-transition model",
                       "No experimentally measured adhesion-strength prediction"):
        assert assumption in page, f"missing assumption: {assumption}"


def test_the_scope_statement_survives_without_the_limitations_section():
    """The limitations panel was removed by request; the scope claim must not go
    with it. It has to stay somewhere a reader cannot miss, or the dashboard
    starts reading as a bond-strength prediction."""
    page = " ".join(dashboard.render_page().decode().split())
    assert "predicts moisture transport, not measured bond strength" in page
    assert "proxy for potential performance change" in page


def test_page_has_all_ten_numbered_sections():
    page = dashboard.render_page().decode()
    for number in range(1, 11):
        assert f'id="s{number}"' in page, f"missing section {number}"
    assert 'id="s11"' not in page, "section numbering must stay contiguous"
    for kind in ("summary", "history", "studies"):
        assert f'data-export="{kind}"' in page


def test_render_page_accepts_overridden_defaults():
    assert "777" in dashboard.render_page({"thickness_um": 777.0}).decode()


# --- page/server contract ----------------------------------------------
#
# The page is hand-written HTML+JS with no build step and no way to run it here,
# so the realistic failure mode is the page and the payload drifting apart: a
# renamed field leaves a blank chart with no error anywhere.

PAGE = dashboard.PAGE_PATH.read_text()


def test_every_control_maps_to_a_real_parameter():
    keys = set(re.findall(r'key: "([a-z_]+)"', PAGE))
    assert keys, "no controls found; the page parser is wrong"
    assert keys - set(dashboard.DEFAULT_PARAMS) == set()


def test_every_parameter_has_a_control_on_the_page():
    """Otherwise a parameter is silently unreachable from the dashboard."""
    keys = set(re.findall(r'key: "([a-z_]+)"', PAGE))
    assert set(dashboard.DEFAULT_PARAMS) - keys == set()


def test_every_svg_and_element_the_script_targets_exists():
    targets = (set(re.findall(r'getElementById\("([a-z-]+)"\)', PAGE))
               | set(re.findall(r'chart\("([a-z-]+)"', PAGE))
               | set(re.findall(r'data-svg="([a-z-]+)"', PAGE)))
    for target in targets:
        assert f'id="{target}"' in PAGE, f"script targets missing element #{target}"


def test_the_page_brackets_balance():
    """A crude but effective guard against a truncated or mis-edited script."""
    script = PAGE[PAGE.index('<script>\n"use strict";'):]
    for opener, closer in (("{", "}"), ("(", ")"), ("[", "]")):
        assert script.count(opener) == script.count(closer), (
            f"unbalanced {opener}{closer} in the page script"
        )


# --- config round-trip -------------------------------------------------

def test_params_from_case_round_trips_through_case_from_params():
    configuration = load_configuration("configs/epoxy_on_steel.yaml")
    params = dashboard.params_from_case(configuration.case)
    rebuilt = dashboard.case_from_params(params)
    original = configuration.case
    assert rebuilt.thickness == pytest.approx(original.thickness)
    assert rebuilt.condition.temperature_k == pytest.approx(
        original.condition.temperature_k
    )
    assert rebuilt.condition.relative_humidity == pytest.approx(
        original.condition.relative_humidity
    )
    assert rebuilt.polymer.diffusivity_ref == pytest.approx(
        original.polymer.diffusivity_ref, rel=1e-9
    )
    assert rebuilt.polymer.activation_energy == pytest.approx(
        original.polymer.activation_energy
    )
    assert rebuilt.criterion.basis == original.criterion.basis
    assert rebuilt.criterion.value == pytest.approx(original.criterion.value)
    assert rebuilt.condition.duration == pytest.approx(
        original.condition.duration, rel=1e-6
    )


def test_unsupported_features_are_named_rather_than_silently_dropped():
    configuration = load_configuration("configs/humidity_cycling.yaml")
    missing = dashboard.unsupported_dashboard_features(configuration.case)
    assert "humidity schedule" in missing


def test_no_unsupported_features_for_a_plain_case():
    configuration = load_configuration("configs/epoxy_on_steel.yaml")
    assert dashboard.unsupported_dashboard_features(configuration.case) == []


# --- HTTP behaviour ----------------------------------------------------

def test_root_serves_the_page(server):
    status, body, headers = _get(server, "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"Polymer Moisture Diffusion" in body


def test_health_and_defaults_endpoints(server):
    status, body, _ = _get(server, "/api/health")
    assert status == 200 and json.loads(body)["status"] == "ok"
    status, body, _ = _get(server, "/api/defaults")
    assert status == 200
    assert json.loads(body)["thickness_um"] == pytest.approx(200.0)


def test_simulate_endpoint_returns_a_full_payload(server):
    status, payload, _ = _post(server, "/api/simulate", {"thickness_um": 120.0})
    assert status == 200
    assert payload["inputs"]["thickness_um"] == pytest.approx(120.0)
    assert payload["elapsed_ms"] > 0.0


def test_simulate_endpoint_accepts_an_empty_body(server):
    status, payload, _ = _post(server, "/api/simulate", {})
    assert status == 200
    assert payload["inputs"]["thickness_um"] == pytest.approx(200.0)


@pytest.mark.parametrize("kind", ["summary", "history", "studies"])
def test_export_endpoints_return_a_downloadable_csv(server, kind):
    status, body, headers = _post(server, "/api/export/" + kind, {})
    assert status == 200
    assert headers["Content-Type"].startswith("text/csv")
    assert "attachment" in headers["Content-Disposition"]
    assert kind in headers["Content-Disposition"]
    assert len(body.splitlines()) > 3


def test_an_unknown_export_is_reported(server):
    status, payload, _ = _post(server, "/api/export/everything", {})
    assert status == 404
    assert "unknown export" in payload["error"]


def test_invalid_parameters_return_a_readable_error_not_a_crash(server):
    for params, needle in (
        ({"thickness_um": -5.0}, "thickness"),
        ({"relative_humidity_pct": 300.0}, "relative humidity"),
        ({"exposure": "sideways"}, "exposure"),
        ({"d_ref": 0.0}, "diffusivity"),
        ({"threshold_basis": "normalized", "threshold_value": 1.0}, "never crossed"),
    ):
        status, payload, _ = _post(server, "/api/simulate", params)
        assert status == 400, f"{params} should be rejected"
        assert needle in payload["error"], f"{params}: {payload['error']}"


def test_oversized_request_body_is_rejected(server):
    status, payload, _ = _post_bytes(
        server,
        "/api/simulate",
        b"{" + b"x" * dashboard.MAX_REQUEST_BYTES + b"}",
    )

    assert status == 413
    assert "request body exceeds" in payload["error"]


def test_export_endpoints_also_validate(server):
    status, payload, _ = _post(server, "/api/export/summary", {"thickness_um": -1.0})
    assert status == 400
    assert "thickness" in payload["error"]


def test_unknown_paths_and_methods_are_rejected_cleanly(server):
    status, payload, _ = _post(server, "/api/nope", {})
    assert status == 404
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(server, "/api/nope")
    assert excinfo.value.code == 404


def test_server_survives_a_bad_request_and_keeps_serving(server):
    _post(server, "/api/simulate", {"thickness_um": -1.0})
    status, _, _ = _post(server, "/api/simulate", {})
    assert status == 200, "server must stay healthy after a rejected request"


def test_concurrent_requests_are_served(server):
    """Threading matters: a slider drag can overlap with an in-flight request."""
    results: list[int] = []
    lock = threading.Lock()

    def hit(thickness):
        status, _, _ = _post(server, "/api/simulate", {"thickness_um": thickness})
        with lock:
            results.append(status)

    threads = [threading.Thread(target=hit, args=(80.0 + 20 * i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert results == [200] * 6


# --- binding ----------------------------------------------------------

def test_build_dashboard_reports_the_port_it_actually_bound():
    bound = dashboard.build_dashboard(port=0)
    try:
        assert bound.port > 0
        assert str(bound.port) in bound.url
        assert bound.url.startswith("http://localhost:")
    finally:
        bound.close()


def test_build_dashboard_walks_forward_past_a_taken_port():
    first = dashboard.build_dashboard(port=0)
    try:
        second = dashboard.build_dashboard(port=first.port, max_attempts=5)
        try:
            assert second.port > first.port
        finally:
            second.close()
    finally:
        first.close()


def test_build_dashboard_gives_up_with_a_clear_message():
    first = dashboard.build_dashboard(port=0)
    try:
        with pytest.raises(HygroadhError, match="could not bind"):
            dashboard.build_dashboard(port=first.port, max_attempts=1)
    finally:
        first.close()


def test_dashboard_works_as_a_context_manager():
    with dashboard.build_dashboard(port=0) as bound:
        assert bound.port > 0


def test_there_is_no_run_button_and_no_dead_machinery_behind_it():
    """The button was removed as redundant: the model is deterministic, so
    re-running unchanged inputs can only reproduce the same numbers. This also
    guards against the visible-feedback scaffolding being left behind."""
    for orphan in ('id="run"', "Run simulation", "setRunning", "pendingForce",
                   "MIN_BUTTON_FEEDBACK_MS", "button.primary", 'id="runinfo"'):
        assert orphan not in PAGE, f"leftover from the removed button: {orphan}"
    # Reset survives -- it does something a re-run cannot.
    assert 'id="reset"' in PAGE


def test_typed_values_apply_without_needing_a_button():
    """Removing the button would regress typing if the boxes still only committed
    on blur. They must apply on `input`, and must not clamp mid-keystroke: typing
    "350" passes through "3" and "35", and rewriting the field under the user's
    fingers is worse than the original bug."""
    assert 'num.addEventListener("input", applyLive)' in PAGE
    assert 'num.addEventListener("blur", commit)' in PAGE
    assert '"Enter"' in PAGE
    live = PAGE[PAGE.index("const applyLive = function ()"):PAGE.index("const commit = function ()")]
    assert 'classList.add("invalid")' in live, "out-of-range input must be marked"
    assert "num.value =" not in live, "applyLive must never rewrite the field"
    assert "Math.min(def.max" in PAGE[PAGE.index("const commit = function ()"):], \
        "clamping belongs in commit, on blur"


def test_a_failed_request_offers_a_retry_since_there_is_no_run_button():
    assert "function bannerWithRetry(" in PAGE
    assert 'id="retry"' in PAGE
    assert "bannerWithRetry(" in PAGE[PAGE.index("} catch (err) {"):]


def test_an_in_flight_request_still_never_drops_the_latest_input():
    body = PAGE[PAGE.index("async function refresh()"):]
    assert "if (pending) { pending = false; refresh(); }" in body, (
        "a coalesced request must re-run immediately, not on a fresh debounce"
    )


def test_the_page_is_a_two_column_shell_with_controls_on_the_left():
    """Layout contract: inputs in a sidebar, every result section in the main column.

    Asserted structurally rather than visually, because there is no browser here.
    What it catches is a section accidentally left outside <main> --- which would
    render underneath the sidebar instead of beside it.
    """
    assert '<div class="layout">' in PAGE
    aside = PAGE[PAGE.index('<aside id="s1">'):PAGE.index("</aside>")]
    main = PAGE[PAGE.index("<main>"):PAGE.index("</main>")]

    # The controls host and the reset button belong to the sidebar.
    for needle in ('id="inputs"', 'id="reset"', 'id="autonote"'):
        assert needle in aside, f"{needle} should be in the sidebar"
        assert needle not in main, f"{needle} must not also be in the main column"

    # Every numbered results section belongs to the main column.
    for number in range(2, 11):
        assert f'id="s{number}"' in main, f"section {number} is not inside <main>"


def test_the_sidebar_is_sticky_and_collapses_on_narrow_screens():
    style = PAGE[PAGE.index("<style>"):PAGE.index("</style>")]
    assert "position: sticky" in style
    assert "overflow-y: auto" in style, "a tall control column must scroll on its own"
    assert "@media (max-width: 1080px)" in style
    collapsed = style[style.index("@media (max-width: 1080px)"):]
    assert "grid-template-columns: 1fr" in collapsed
    assert "position: static" in collapsed


def test_every_control_lands_in_a_named_group_that_the_sidebar_renders():
    """An unrecognised group would silently drop its controls off the page."""
    groups = set(re.findall(r'group: "([A-Za-z ]+)"', PAGE))
    assert groups, "no control groups found; the page parser is wrong"
    order = re.search(r"GROUP_ORDER = \[(.*?)\]", PAGE, re.S).group(1)
    for group in groups:
        assert f'"{group}"' in order, f"group {group!r} is missing from GROUP_ORDER"
    keys = re.findall(r'key: "([a-z_]+)"', PAGE)
    grouped = re.findall(r'group: "[A-Za-z ]+", key: "([a-z_]+)"', PAGE)
    assert set(keys) == set(grouped), "every control must declare a group"


# --- WSL compatibility --------------------------------------------------

def test_wsl_is_detected_from_the_environment(monkeypatch):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    assert dashboard.is_wsl() is True
    monkeypatch.delenv("WSL_DISTRO_NAME")
    monkeypatch.setenv("WSL_INTEROP", "/run/WSL/8_interop")
    assert dashboard.is_wsl() is True


def test_wsl_is_detected_from_the_kernel_string(monkeypatch, tmp_path):
    """The environment variables are missing under some init setups, so the
    kernel release string is checked too."""
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    fake = tmp_path / "version"
    real_path = dashboard.Path
    monkeypatch.setattr(
        dashboard, "Path",
        lambda p: fake if p == "/proc/version" else real_path(p),
    )

    fake.write_text("Linux version 5.15.153.1-microsoft-standard-WSL2 (root@build)")
    assert dashboard.is_wsl() is True
    fake.write_text("Linux version 6.8.0-136-generic (buildd@lcy02-amd64-082)")
    assert dashboard.is_wsl() is False


def test_a_missing_proc_version_is_not_wsl(monkeypatch):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)

    class Missing:
        def read_text(self):
            raise OSError("no such file")

    real_path = dashboard.Path
    monkeypatch.setattr(
        dashboard, "Path",
        lambda p: Missing() if p == "/proc/version" else real_path(p),
    )
    assert dashboard.is_wsl() is False


def test_loopback_advertises_only_the_local_url():
    urls = dashboard.access_urls("127.0.0.1", 8765)
    assert urls == [("http://localhost:8765/", "from this machine")]


def test_binding_all_interfaces_advertises_a_reachable_address():
    """The point of --host 0.0.0.0 under WSL is an address Windows can reach, so
    the routable IP has to be printed, not just localhost."""
    urls = dashboard.access_urls("0.0.0.0", 8765)
    assert urls[0] == ("http://localhost:8765/", "from this machine")
    assert len(urls) == 2
    url, note = urls[1]
    assert url.endswith(":8765/") and url.startswith("http://")
    assert "Windows" in note
    assert dashboard.local_ip() in url


def test_an_explicit_host_is_advertised_verbatim():
    assert dashboard.access_urls("192.168.1.50", 9000) == [
        ("http://192.168.1.50:9000/", "as bound")
    ]


def test_local_ip_returns_something_usable():
    address = dashboard.local_ip()
    assert address is None or address.count(".") == 3


def test_no_browser_is_opened_unless_asked(monkeypatch):
    """Under WSL there is usually no Linux browser, and printing the URL is more
    reliable than launching something. Off by default."""
    import inspect

    signature = inspect.signature(dashboard.serve)
    assert signature.parameters["open_browser"].default is False


def test_wsl_browser_launch_prefers_the_windows_helpers(monkeypatch):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    launched: list[list[str]] = []
    monkeypatch.setattr(dashboard.shutil, "which",
                        lambda name: "/usr/bin/" + name if name == "wslview" else None)
    monkeypatch.setattr(dashboard.subprocess, "Popen",
                        lambda cmd, **kw: launched.append(cmd))
    assert dashboard.open_in_browser("http://localhost:8765/") is True
    assert launched == [["wslview", "http://localhost:8765/"]]


def test_wsl_browser_launch_falls_back_to_explorer(monkeypatch):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    launched: list[list[str]] = []
    monkeypatch.setattr(dashboard.shutil, "which",
                        lambda name: "/mnt/c/explorer.exe" if name == "explorer.exe" else None)
    monkeypatch.setattr(dashboard.subprocess, "Popen",
                        lambda cmd, **kw: launched.append(cmd))
    assert dashboard.open_in_browser("http://localhost:8765/") is True
    assert launched == [["explorer.exe", "http://localhost:8765/"]]


def test_wsl_browser_launch_reports_failure_when_nothing_is_available(monkeypatch):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    monkeypatch.setattr(dashboard.shutil, "which", lambda name: None)
    assert dashboard.open_in_browser("http://localhost:8765/") is False


# --- typeset mathematics ------------------------------------------------

def test_equations_are_mathml_not_plain_text():
    """The governing equations are typeset, not monospaced ASCII.

    MathML is used rather than a JavaScript typesetter because every current
    browser lays it out natively from a system math font, so the page needs no
    external script, stylesheet, or webfont and stays self-contained.
    """
    assert PAGE.count("<math") >= 4
    assert PAGE.count('display="block"') >= 4
    # The old ASCII renderings must be gone.
    for ascii_form in ("D_ref &middot;", "t_diffusion ~", "&part;&sup2;C/&part;x&sup2;",
                       "L&sup2;/D"):
        assert ascii_form not in PAGE, f"plain-text equation left behind: {ascii_form}"


def test_every_mathml_block_is_well_formed_xml():
    """A malformed block renders as run-together characters with no error."""
    import xml.etree.ElementTree as ET

    blocks = re.findall(r"<math.*?</math>", PAGE, re.S)
    assert len(blocks) >= 4
    for index, block in enumerate(blocks, 1):
        try:
            ET.fromstring(block)
        except ET.ParseError as error:  # pragma: no cover - guards a real failure
            raise AssertionError(f"MathML block {index} is malformed: {error}")


def test_the_governing_equations_are_all_present():
    page = dashboard.render_page().decode()
    # Fick's second law: a second derivative in x against a first in t.
    assert page.count("&#x2202;") >= 4, "the partial-derivative operator"
    assert "<msup><mo>&#x2202;</mo><mn>2</mn></msup>" in page, "second order in x"
    # Arrhenius: an exponential of -Ea/R times a reciprocal-temperature difference.
    assert "<mi>exp</mi>" in page
    assert "<msub><mi>E</mi><mi>a</mi></msub>" in page
    assert "<msub><mi>T</mi><mi>ref</mi></msub>" in page
    # The characteristic time and the Fourier number.
    assert "<msub><mi>t</mi><mi>diffusion</mi></msub>" in page
    assert "<mi>Fo</mi>" in page


def test_the_fourier_number_the_ui_quotes_is_actually_defined():
    """The profile snapshots are labelled with Fo values, so the page has to say
    what Fo is and which length scale it uses."""
    page = " ".join(dashboard.render_page().decode().split())
    assert "Fourier number" in page
    assert "diffusion length" in page


def test_inline_math_avoids_stacked_fractions():
    """A stacked fraction in running text inflates the line height; inline math
    should use solidus form."""
    for block in re.findall(r'<math class="im">.*?</math>', PAGE, re.S):
        assert "<mfrac" not in block, f"stacked fraction used inline: {block}"


def test_subscripts_render_as_the_letters_they_are_meant_to_be():
    """Guards a real bug: &#8337;&#8340; are subscript 'e' and 'schwa', so the
    far-face legend read C/C-e-schwa instead of C/C-sat."""
    assert "&#8337;" not in PAGE and "&#8340;" not in PAGE
    # sat = subscript s, a, t
    assert "&#8347;&#8336;&#8348;" in PAGE or "ₛₐₜ" in PAGE


def test_typeset_math_does_not_break_the_self_contained_guarantee():
    """MathML needs no webfont; a JavaScript typesetter would have needed one."""
    page = dashboard.render_page().decode().lower()
    for forbidden in ("<link", "@import", "<script src", "@font-face", "mathjax",
                      "katex"):
        assert forbidden not in page, f"typesetting pulled in {forbidden!r}"


# --- animation ----------------------------------------------------------

def test_the_payload_marks_the_frame_where_the_threshold_is_crossed():
    """Computed server-side so the animation and the reported time cannot
    disagree about where the criterion is met."""
    payload = dashboard.simulate_payload({})
    profile = payload["profile"]
    index = profile["threshold_frame"]
    assert isinstance(index, int) and 0 < index < len(profile["frames"])

    level = payload["derived"]["threshold_normalized"]
    far_face = [frame[-1] for frame in profile["frames"]]
    assert far_face[index] >= level, "the marked frame must be at or above"
    assert far_face[index - 1] < level, "and the one before it must be below"
    assert all(value < level for value in far_face[:index]), "it must be the first"


def test_the_threshold_frame_sits_just_after_the_interpolated_time():
    """The frame is a sample, the reported time is interpolated between samples,
    so the frame lands at or after it -- never before."""
    payload = dashboard.simulate_payload({})
    index = payload["profile"]["threshold_frame"]
    frame_time = payload["profile"]["frame_time_s"][index]
    reported = payload["derived"]["time_to_threshold_s"]
    assert frame_time >= reported
    assert frame_time < reported * 1.5, "and not wildly after it"


def test_there_is_no_threshold_frame_when_the_threshold_is_unreachable():
    payload = dashboard.simulate_payload({
        "threshold_basis": "wt_pct", "threshold_value": 3.0,
        "relative_humidity_pct": 50.0,
    })
    assert payload["profile"]["threshold_frame"] is None
    assert payload["derived"]["time_to_threshold_s"] is None


def test_the_threshold_frame_moves_with_the_threshold():
    frames = [
        dashboard.simulate_payload({"threshold_value": value})["profile"]["threshold_frame"]
        for value in (0.2, 0.4, 0.6, 0.8)
    ]
    assert all(isinstance(f, int) for f in frames)
    assert frames == sorted(frames), "a higher threshold is crossed later"
    assert frames[0] < frames[-1]


def test_the_page_has_both_animation_checkboxes():
    assert 'id="anim-threshold"' in PAGE
    assert 'id="anim-end"' in PAGE
    assert PAGE.count('type="checkbox"') >= 2
    assert "Animate until the threshold" in PAGE
    assert "Animate to the end" in PAGE


def test_the_two_animation_modes_are_mutually_exclusive():
    """You cannot both halt at the threshold and continue past it, so checking
    one has to clear the other."""
    block = PAGE[PAGE.index('getElementById("anim-threshold").addEventListener'):]
    threshold_handler = block[:block.index('getElementById("anim-end").addEventListener')]
    assert 'getElementById("anim-end").checked = false' in threshold_handler
    end_handler = block[block.index('getElementById("anim-end").addEventListener'):]
    assert 'getElementById("anim-threshold").checked = false' in end_handler


def test_playback_uses_animation_frames_and_keeps_real_time():
    """requestAnimationFrame rather than setInterval, so it pauses with the tab;
    and the step loop consumes accumulated time so a slow repaint does not make
    the animation drift slower than the clock."""
    assert "requestAnimationFrame(tickPlayback)" in PAGE
    assert "cancelAnimationFrame" in PAGE
    tick = PAGE[PAGE.index("function tickPlayback(now)"):]
    tick = tick[:tick.index("function drawFrame()")]
    assert "while (playback.spare >= FRAME_INTERVAL_MS" in tick


def test_playback_stops_at_its_stop_point_and_clears_the_checkbox():
    tick = PAGE[PAGE.index("function tickPlayback(now)"):]
    tick = tick[:tick.index("function drawFrame()")]
    assert "if (index >= playback.stopAt) { stopPlayback(); return; }" in tick
    stop = PAGE[PAGE.index("function stopPlayback()"):]
    stop = stop[:stop.index("function startPlayback(")]
    assert 'anim-threshold").checked = false' in stop
    assert 'anim-end").checked = false' in stop


def test_taking_the_slider_by_hand_cancels_playback():
    handler = PAGE[PAGE.index('getElementById("frame").addEventListener'):]
    handler = handler[:handler.index("});")]
    assert "stopPlayback()" in handler


def test_playback_replays_from_the_start_once_finished():
    start = PAGE[PAGE.index("function startPlayback(mode)"):]
    start = start[:start.index("function tickPlayback")]
    assert "if (currentFrame() >= stopAt) setFrame(0)" in start


def test_the_threshold_checkbox_is_disabled_when_there_is_nothing_to_stop_at():
    assert "box.disabled = !reachable" in PAGE
    assert "nothing to stop at" in PAGE


def test_playback_survives_a_re_run_that_changes_the_frame_count():
    """An input change while animating re-runs the model; the stop point has to
    be recomputed rather than pointing at a frame that no longer exists."""
    block = PAGE[PAGE.index("if (playback) {"):]
    block = block[:block.index("drawFrame();")]
    assert "stopAtFor(playback.mode)" in block
    assert "Math.min(stopAt, slider.max)" in block
