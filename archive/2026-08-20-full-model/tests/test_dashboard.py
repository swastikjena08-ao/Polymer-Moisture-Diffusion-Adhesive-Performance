"""Tests for the dashboard server: real HTTP requests against a bound port."""

from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from hygroadh import dashboard
from hygroadh.config import load_configuration
from hygroadh.diffusion.fd import Schedule
from hygroadh.simulate import ExposureCondition
from hygroadh.units import HygroadhError


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
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


# --- payload construction (no server needed) -------------------------------

def test_default_params_build_a_valid_case():
    case = dashboard.case_from_params({})
    assert case.thickness == pytest.approx(200e-6)
    assert case.condition.exposure == "one_sided"
    assert case.ari_threshold == pytest.approx(dashboard.DEFAULT_PARAMS["ari_threshold"])


def test_payload_carries_every_field_the_page_reads():
    payload = dashboard.simulate_payload({})
    for key in (
        "time_days", "uptake", "interface", "ari", "plasticization", "thermal",
        "hydrolysis", "depth_um", "profile", "profile_time_s", "summary", "map",
        "sensitivity", "elapsed_ms", "tg_dry_c", "tg_wet_final_c",
        "first_positive_day", "interface_pct_final",
    ):
        assert key in payload, f"payload missing {key}"
    assert len(payload["time_days"]) == len(payload["ari"])
    assert len(payload["depth_um"]) == len(payload["profile"])


def test_payload_is_strictly_json_serializable():
    """No Infinity or NaN tokens, which JSON.parse rejects outright."""
    payload = dashboard.simulate_payload({"duration_days": 0.2, "temperature_c": 10.0})
    text = json.dumps(payload, allow_nan=False)  # raises if any nan/inf survived
    assert "Infinity" not in text and "NaN" not in text


def test_unreached_threshold_becomes_null_not_infinity():
    payload = dashboard.simulate_payload({
        "duration_days": 0.05, "temperature_c": 8.0, "thickness_um": 1500.0,
        "plasticization_gain": 0.01, "hydrolysis_rate_ref": 1e-10,
        "relative_humidity": 0.3,
    })
    assert payload["summary"]["time_to_ari_threshold_s"] is None


def test_json_safe_maps_non_finite_values_to_null():
    cleaned = dashboard.json_safe({
        "a": math.inf, "b": -math.inf, "c": math.nan, "d": 1.5,
        "e": np.array([1.0, np.inf]), "f": np.int64(3), "g": np.True_,
    })
    assert cleaned["a"] is None and cleaned["b"] is None and cleaned["c"] is None
    assert cleaned["d"] == 1.5
    assert cleaned["e"] == [1.0, None]
    assert cleaned["f"] == 3 and cleaned["g"] is True


def test_design_map_stops_short_of_the_dry_glass_transition():
    """Above the dry Tg the adhesion model has no baseline and refuses to run."""
    case = dashboard.case_from_params({"tg_dry_c": 70.0, "temperature_c": 30.0})
    grid = dashboard._design_map(case)
    assert grid["temperature_c"], "map should still have a usable range"
    assert max(grid["temperature_c"]) <= 70.0 - dashboard.TG_HEADROOM + 1e-9


def test_design_map_degrades_gracefully_with_no_usable_temperature_range():
    # Dry Tg of 32 degC leaves the headroom ceiling below the map's 5 degC floor.
    case = dashboard.case_from_params({"tg_dry_c": 32.0, "temperature_c": 8.0})
    grid = dashboard._design_map(case)
    assert grid["t_ari_days"] == []
    assert "note" in grid


def test_sensitivity_reports_an_explanation_instead_of_failing():
    payload = dashboard.simulate_payload({
        "duration_days": 0.05, "temperature_c": 8.0, "thickness_um": 1500.0,
        "plasticization_gain": 0.01, "hydrolysis_rate_ref": 1e-10,
        "relative_humidity": 0.3,
    })
    assert "error" in payload["sensitivity"]
    assert "never reached" in payload["sensitivity"]["error"]


def test_profile_fraction_selects_a_snapshot_time():
    early = dashboard.simulate_payload({"profile_fraction": 0.02})
    late = dashboard.simulate_payload({"profile_fraction": 1.0})
    assert early["profile_time_s"] < late["profile_time_s"]
    # Later snapshots are wetter everywhere through the thickness.
    assert np.all(np.array(late["profile"]) >= np.array(early["profile"]) - 1e-12)


# --- the page --------------------------------------------------------------

def test_page_injects_defaults_and_keeps_no_placeholder():
    page = dashboard.render_page().decode()
    assert "window.__HYGROADH_DEFAULTS__" not in page
    assert '"thickness_um": 200' in page
    assert "<title>" in page


def test_page_is_self_contained():
    """No external scripts, styles, fonts, or images to fetch.

    Checked as resource-loading constructs rather than by searching for "http",
    because the SVG namespace URI is a required identifier that is never
    fetched over the network.
    """
    page = dashboard.render_page().decode().lower()
    for forbidden in (
        "<link", "@import", "<script src", "src=\"http", "src='http",
        "href=\"http", "href='http", "url(http", "//cdn", "fetch(\"http",
        "@font-face",
    ):
        assert forbidden not in page, f"page references {forbidden!r}"
    # The one network call it does make must be same-origin and relative.
    assert 'fetch("/api/simulate"' in page


def test_page_carries_the_standing_caveat():
    """The ARI's status must travel with the numbers, not just live in the docs."""
    # Whitespace-normalized: the caveat is wrapped across several source lines.
    page = " ".join(dashboard.render_page().decode().split())
    assert "phenomenological model with parameters you supply" in page
    assert "not validated against joint-strength measurements" in page
    assert "rank designs and exposure conditions" in page


def test_render_page_accepts_overridden_defaults():
    page = dashboard.render_page({"thickness_um": 777.0}).decode()
    assert "777" in page


# --- config round-trip -----------------------------------------------------

def test_params_from_case_round_trips_through_case_from_params():
    configuration = load_configuration("configs/epoxy_on_steel.yaml")
    params = dashboard.params_from_case(configuration.case)
    rebuilt = dashboard.case_from_params(params)
    original = configuration.case
    assert rebuilt.thickness == pytest.approx(original.thickness)
    assert rebuilt.condition.temperature_k == pytest.approx(original.condition.temperature_k)
    assert rebuilt.condition.relative_humidity == pytest.approx(
        original.condition.relative_humidity
    )
    assert rebuilt.polymer.diffusivity_ref == pytest.approx(
        original.polymer.diffusivity_ref, rel=1e-9
    )
    assert rebuilt.polymer.tg_dry_k == pytest.approx(original.polymer.tg_dry_k)
    assert rebuilt.ari_threshold == pytest.approx(original.ari_threshold)


def test_unsupported_features_are_named_rather_than_silently_dropped():
    configuration = load_configuration("configs/humidity_cycling.yaml")
    missing = dashboard.unsupported_dashboard_features(configuration.case)
    assert "humidity schedule" in missing


def test_no_unsupported_features_for_a_plain_case():
    configuration = load_configuration("configs/epoxy_on_steel.yaml")
    assert dashboard.unsupported_dashboard_features(configuration.case) == []


# --- HTTP behaviour --------------------------------------------------------

def test_root_serves_the_page(server):
    status, body, headers = _get(server, "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"Hygro-Adhesion Explorer" in body


def test_health_and_defaults_endpoints(server):
    status, body, _ = _get(server, "/api/health")
    assert status == 200 and json.loads(body)["status"] == "ok"
    status, body, _ = _get(server, "/api/defaults")
    assert status == 200
    assert json.loads(body)["thickness_um"] == pytest.approx(200.0)


def test_simulate_endpoint_returns_a_full_payload(server):
    status, payload = _post(server, "/api/simulate", {"thickness_um": 120.0})
    assert status == 200
    assert payload["summary"]["thickness_um"] == pytest.approx(120.0)
    assert payload["elapsed_ms"] > 0.0


def test_simulate_endpoint_accepts_an_empty_body(server):
    status, payload = _post(server, "/api/simulate", {})
    assert status == 200
    assert payload["summary"]["thickness_um"] == pytest.approx(200.0)


def test_invalid_parameters_return_a_readable_error_not_a_crash(server):
    status, payload = _post(server, "/api/simulate", {"thickness_um": -5.0})
    assert status == 400
    assert "thickness" in payload["error"]
    status, payload = _post(server, "/api/simulate", {"relative_humidity": 3.0})
    assert status == 400
    assert "relative_humidity" in payload["error"]
    status, payload = _post(server, "/api/simulate", {"exposure": "sideways"})
    assert status == 400
    assert "one_sided" in payload["error"]


def test_non_numeric_parameter_is_reported_by_name(server):
    status, payload = _post(server, "/api/simulate", {"thickness_um": "thick"})
    assert status == 400
    assert "thickness_um" in payload["error"]


def test_unknown_paths_and_methods_are_rejected_cleanly(server):
    status, payload = _post(server, "/api/nope", {})
    assert status == 404
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(server, "/api/nope")
    assert excinfo.value.code == 404


def test_server_survives_a_bad_request_and_keeps_serving(server):
    _post(server, "/api/simulate", {"thickness_um": -1.0})
    status, payload = _post(server, "/api/simulate", {})
    assert status == 200, "server must stay healthy after a rejected request"


def test_concurrent_requests_are_served(server):
    """Threading matters: a slider drag can overlap with an in-flight request."""
    results: list[int] = []
    lock = threading.Lock()

    def hit(thickness):
        status, _ = _post(server, "/api/simulate", {"thickness_um": thickness})
        with lock:
            results.append(status)

    threads = [threading.Thread(target=hit, args=(80.0 + 20 * i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=90)
    assert results == [200] * 6


# --- binding -------------------------------------------------------------

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
            assert second.port != first.port
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


# --- page/server contract --------------------------------------------------
#
# The page is hand-written HTML+JS with no build step and no way to run it in
# this environment, so the realistic failure mode is the page and the payload
# drifting apart: a renamed field leaves a blank chart with no error anywhere.
# These tests check the contract statically.

import re

PAGE = dashboard.PAGE_PATH.read_text()


def _referenced(pattern: str) -> set[str]:
    return set(re.findall(pattern, PAGE))


def test_every_slider_maps_to_a_real_parameter():
    keys = _referenced(r'data-key="([a-z_]+)"')
    assert keys, "no sliders found; the page parser is wrong"
    unknown = keys - set(dashboard.DEFAULT_PARAMS)
    assert not unknown, f"page has controls for unknown parameters: {sorted(unknown)}"


def test_every_parameter_has_a_control_on_the_page():
    """Otherwise a parameter is silently unreachable from the dashboard."""
    keys = _referenced(r'data-key="([a-z_]+)"')
    # `exposure` is a <select>, not a range input.
    covered = keys | {"exposure"}
    missing = set(dashboard.DEFAULT_PARAMS) - covered
    assert not missing, f"parameters with no control: {sorted(missing)}"
    assert 'id="exposure"' in PAGE


def test_every_payload_field_the_page_reads_is_actually_sent():
    payload = dashboard.simulate_payload({})
    referenced = _referenced(r"\bdata\.([a-z0-9_]+)")
    assert referenced, "no payload references found; the page parser is wrong"
    missing = referenced - set(payload) - {"error"}
    assert not missing, f"page reads fields the server never sends: {sorted(missing)}"


def test_every_summary_field_the_page_reads_is_actually_sent():
    summary = dashboard.simulate_payload({})["summary"]
    referenced = _referenced(r"\bs\.([a-z0-9_]+)")
    assert referenced, "no summary references found; the page parser is wrong"
    missing = referenced - set(summary)
    assert not missing, f"page reads summary fields that do not exist: {sorted(missing)}"


def test_every_map_field_the_page_reads_is_actually_sent():
    grid = dashboard.simulate_payload({})["map"]
    referenced = _referenced(r"\bgrid\.([a-z0-9_]+)")
    assert referenced, "no map references found; the page parser is wrong"
    missing = referenced - set(grid)
    assert not missing, f"page reads map fields that do not exist: {sorted(missing)}"


def test_every_svg_element_the_script_targets_exists_in_the_markup():
    targets = _referenced(r'getElementById\("([a-z-]+)"\)') | _referenced(
        r'lineChart\("([a-z-]+)"'
    ) | _referenced(r'heatmap\("([a-z-]+)"')
    for target in targets:
        assert f'id="{target}"' in PAGE, f"script targets missing element #{target}"


def test_the_page_brackets_balance():
    """A crude but effective guard against a truncated or mis-edited script."""
    script = PAGE[PAGE.index('<script>\n"use strict";'):]
    for opener, closer in (("{", "}"), ("(", ")"), ("[", "]")):
        assert script.count(opener) == script.count(closer), (
            f"unbalanced {opener}{closer} in the page script"
        )
