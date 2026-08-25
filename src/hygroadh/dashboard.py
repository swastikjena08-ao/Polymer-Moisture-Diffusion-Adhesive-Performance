"""Local web dashboard for the moisture-diffusion study.

Runs a small standard-library HTTP server and serves a self-contained page. The
browser sends parameter changes to Python, which performs all calculations.

Endpoints
---------
``GET  /``                    the dashboard page, with defaults injected
``GET  /api/defaults``        the default parameter set
``GET  /api/health``          liveness probe
``POST /api/simulate``        one run: histories, profiles, and all four studies
``POST /api/export/history``  the time history as CSV
``POST /api/export/studies``  the parameter studies as CSV
``POST /api/export/summary``  inputs and derived results as CSV

Only ``http.server``, ``csv``, and ``json`` from the standard library are used,
so the dashboard needs nothing beyond numpy.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from . import sensitivity as sensitivity_module
from .diffusion.base import equivalent_sheet_thickness
from .materials import Polymer
from .simulate import Case, ExposureCondition, run
from .sorption import HenryIsotherm
from .threshold import BASES, MoistureCriterion
from .units import KELVIN_OFFSET, HygroadhError, days, from_days, from_hours, hours

PAGE_PATH = Path(__file__).with_name("dashboard.html")

#: Starting parameter set, also the "reset" target in the page. Every value is a
#: model parameter the user can change; none is treated as a material constant.
DEFAULT_PARAMS: dict[str, Any] = {
    # Section 1 -- model inputs
    "thickness_um": 200.0,
    "temperature_c": 60.0,
    "relative_humidity_pct": 85.0,
    "d_ref": 1.0e-13,
    "t_ref_c": 25.0,
    "ea_kj": 50.0,
    "m_ref_pct": 2.4,
    "threshold_basis": "normalized",
    "threshold_value": 0.5,
    "duration_mode": "auto",
    "duration_hours": 24.0,
    "exposure": "one_sided",
    # Study ranges
    "study_temp_min_c": 20.0,
    "study_temp_max_c": 80.0,
}

#: Fourier number the automatic duration runs to. At Fo = 1.5 the film is within
#: 0.02% of equilibrium, so the whole approach is visible without guessing.
AUTO_DURATION_FOURIER = 1.5

#: Fourier numbers for the four labelled profile snapshots in Section 3.
SNAPSHOT_FOURIER = (0.01, 0.05, 0.2, 1.0)
SNAPSHOT_LABELS = ("early", "intermediate", "late", "near equilibrium")

#: Sample points in the thickness and diffusivity studies. Both are spaced
#: geometrically and centred on the value currently set, rather than fixed lists,
#: so a sweep always straddles the operating point instead of drifting off it.
#: At the default 200 um a factor-4 span reproduces 50/100/200/400/800 um exactly.
STUDY_POINTS = 5
THICKNESS_STUDY_SPAN = 4.0
DIFFUSIVITY_STUDY_SPAN = 100.0
#: Temperatures tabulated in Section 7, degrees Celsius.
TEMPERATURE_STUDY_STEP_C = 10.0
#: Points on the continuous temperature curves in Section 2.
TEMPERATURE_CURVE_POINTS = 61
#: Points on the temperature-vs-penetration-time curve (one full run each).
TEMPERATURE_PENETRATION_POINTS = 25
#: Times retained for the profile slider.
PROFILE_SLIDER_FRAMES = 80
MAX_REQUEST_BYTES = 64 * 1024


def json_safe(value: Any) -> Any:
    """Convert numpy types and non-finite floats into JSON-representable values.

    JSON has no literal for infinity or NaN, and ``json.dumps`` emits the
    non-standard ``Infinity`` token that ``JSON.parse`` rejects. Mapping them to
    ``null`` lets the page render "not reached", which is the meaning anyway.
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _reject_unknown(params: dict[str, Any]) -> None:
    """Refuse parameters the model does not have.

    The page only ever sends keys from :data:`DEFAULT_PARAMS`, so anything else
    is a caller error or a leftover from an older parameter set. Ignoring it
    would silently drop the setting and run on a default --- the same failure the
    config loader's strict key checking exists to prevent.
    """
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown:
        raise HygroadhError(
            "unknown parameter(s): " + ", ".join(unknown)
            + f". Known parameters: {', '.join(sorted(DEFAULT_PARAMS))}"
        )


def _number(params: dict[str, Any], key: str) -> float:
    raw = params.get(key, DEFAULT_PARAMS[key])
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise HygroadhError(f"{key} must be a number, got {raw!r}") from exc
    if not math.isfinite(value):
        raise HygroadhError(f"{key} must be a finite number, got {raw!r}")
    return value


def _choice(params: dict[str, Any], key: str, allowed: tuple[str, ...]) -> str:
    value = params.get(key, DEFAULT_PARAMS[key])
    if value not in allowed:
        raise HygroadhError(f"{key} must be one of {list(allowed)}, got {value!r}")
    return str(value)


def auto_duration(thickness: float, diffusivity: float, exposure: str) -> float:
    """Return a simulation window based on the case's diffusion time."""
    sheet = equivalent_sheet_thickness(thickness, exposure)
    return AUTO_DURATION_FOURIER * sheet**2 / diffusivity


def case_from_params(params: dict[str, Any]) -> Case:
    """Build a :class:`~hygroadh.simulate.Case` from a flat parameter mapping.

    Validation lives in the model's own constructors, so a nonsensical input
    combination surfaces as a :class:`~hygroadh.units.HygroadhError` with a
    message the page can display verbatim.
    """
    _reject_unknown(params)
    exposure = _choice(params, "exposure", ("one_sided", "two_sided"))
    basis = _choice(params, "threshold_basis", BASES)

    humidity_pct = _number(params, "relative_humidity_pct")
    if not 0.0 < humidity_pct <= 100.0:
        raise HygroadhError(
            f"relative humidity must be above 0% and at most 100%, got "
            f"{humidity_pct:g}%"
        )

    polymer = Polymer(
        diffusivity_ref=_number(params, "d_ref"),
        activation_energy=_number(params, "ea_kj") * 1e3,
        temperature_ref_k=_number(params, "t_ref_c") + KELVIN_OFFSET,
        isotherm=HenryIsotherm(
            m_ref_pct=_number(params, "m_ref_pct"),
            temperature_ref_k=_number(params, "t_ref_c") + KELVIN_OFFSET,
        ),
    )
    temperature_k = _number(params, "temperature_c") + KELVIN_OFFSET
    criterion = MoistureCriterion(
        value=_number(params, "threshold_value"), basis=basis
    )

    mode = _choice(params, "duration_mode", ("auto", "manual"))
    if mode == "auto":
        duration = auto_duration(
            _number(params, "thickness_um") * 1e-6,
            polymer.diffusivity(temperature_k),
            exposure,
        )
    else:
        duration = from_hours(_number(params, "duration_hours"))

    condition = ExposureCondition(
        temperature_k=temperature_k,
        duration=duration,
        relative_humidity=humidity_pct / 100.0,
        exposure=exposure,
        n_times=400,
    )
    return Case(
        thickness=_number(params, "thickness_um") * 1e-6,
        condition=condition,
        polymer=polymer,
        criterion=criterion,
        n_cells=60,
        name="dashboard",
    )


def params_from_case(case: Case) -> dict[str, Any]:
    """Project a :class:`~hygroadh.simulate.Case` back onto dashboard parameters.

    The inverse of :func:`case_from_params`, so a config file can seed the
    controls. It is deliberately lossy: the dashboard exposes a fixed subset of
    the model, and anything outside that subset (humidity schedules, surface
    resistance, a non-default equilibrium uptake) has no control to land on.
    Those settings are dropped rather than silently misrepresented.
    """
    polymer = case.polymer
    condition = case.condition
    params = dict(DEFAULT_PARAMS)
    params.update({
        "thickness_um": case.thickness * 1e6,
        "exposure": condition.exposure,
        "temperature_c": condition.temperature_k - KELVIN_OFFSET,
        "relative_humidity_pct": condition.relative_humidity * 100.0,
        "d_ref": polymer.diffusivity_ref,
        "t_ref_c": polymer.temperature_ref_k - KELVIN_OFFSET,
        "ea_kj": polymer.activation_energy / 1e3,
        "m_ref_pct": polymer.isotherm.m_ref_pct,
        "threshold_basis": case.criterion.basis,
        "threshold_value": case.criterion.value,
        "duration_mode": "manual",
        "duration_hours": hours(condition.duration),
    })
    return params


def unsupported_dashboard_features(case: Case) -> list[str]:
    """Name the parts of a case the dashboard's controls cannot represent."""
    missing = []
    if case.condition.humidity_schedule is not None:
        missing.append("humidity schedule")
    if case.condition.surface_transfer is not None:
        missing.append("surface mass-transfer resistance")
    if case.polymer.isotherm.enthalpy_sorption != 0.0:
        missing.append("sorption enthalpy")
    return missing


def _scalar_case(case: Case, n_times: int = 400) -> Case:
    """A copy tuned for extracting a threshold time cheaply."""
    return case.without_profile(n_times)


def with_auto_duration(case: Case) -> Case:
    """Resize the simulation window to this case's diffusion time."""
    duration = auto_duration(
        case.thickness,
        case.polymer.diffusivity(case.condition.temperature_k),
        case.condition.exposure,
    )
    return replace(case, condition=replace(case.condition, duration=duration))


def _threshold_time(case: Case, auto: bool) -> float:
    point = with_auto_duration(case) if auto else case
    return run(_scalar_case(point)).time_to_threshold


# --- Section 2: temperature -------------------------------------------------

def _temperature_curves(case: Case, params: dict[str, Any],
                        auto: bool) -> dict[str, Any]:
    """Diffusivity and penetration time against temperature.

    Makes the study's central chain explicit: temperature raises ``D(T)``, which
    speeds transport, which shortens the time for moisture to reach the far face.
    """
    low = _number(params, "study_temp_min_c")
    high = _number(params, "study_temp_max_c")
    if high <= low:
        raise HygroadhError(
            f"study temperature range must increase: got {low:g} to {high:g} degC"
        )
    polymer = case.polymer

    fine_c = np.linspace(low, high, TEMPERATURE_CURVE_POINTS)
    diffusivity = np.array(
        [polymer.diffusivity(t + KELVIN_OFFSET) for t in fine_c]
    )

    coarse_c = np.linspace(low, high, TEMPERATURE_PENETRATION_POINTS)
    penetration = np.array([
        _threshold_time(case.with_temperature(t + KELVIN_OFFSET), auto)
        for t in coarse_c
    ])
    return {
        "curve_temperature_c": fine_c,
        "curve_diffusivity": diffusivity,
        "penetration_temperature_c": coarse_c,
        "penetration_time_s": penetration,
    }


def _temperature_study(case: Case, params: dict[str, Any],
                       auto: bool) -> list[dict[str, Any]]:
    """Section 7: the tabulated temperature sweep, one of the main outputs."""
    low = _number(params, "study_temp_min_c")
    high = _number(params, "study_temp_max_c")
    count = int(round((high - low) / TEMPERATURE_STUDY_STEP_C)) + 1
    temperatures = np.linspace(low, high, max(count, 2))
    rows = []
    for celsius in temperatures:
        kelvin = celsius + KELVIN_OFFSET
        point = case.with_temperature(kelvin)
        rows.append({
            "temperature_c": float(celsius),
            "temperature_k": float(kelvin),
            "diffusivity": case.polymer.diffusivity(kelvin),
            "time_to_threshold_s": _threshold_time(point, auto),
        })
    return rows


# --- Sections 5 and 6: thickness and diffusivity ---------------------------

def study_points(centre: float, span: float, points: int = STUDY_POINTS) -> np.ndarray:
    """Geometric sample points spanning a factor ``span`` either side of ``centre``.

    Symmetric in the logarithm, so the middle point is exactly the value the user
    has set and each sweep reports the operating point alongside its neighbours.
    """
    return centre * np.geomspace(1.0 / span, span, points)


def _thickness_study(case: Case, auto: bool) -> dict[str, Any]:
    """Run the thickness study and return its L-squared reference curve."""
    thickness_um = study_points(case.thickness * 1e6, THICKNESS_STUDY_SPAN)
    times = np.array(
        [_threshold_time(case.with_thickness(t * 1e-6), auto) for t in thickness_um]
    )
    reference = np.full_like(times, np.nan)
    finite = np.isfinite(times)
    if np.any(finite):
        first = int(np.argmax(finite))
        reference = times[first] * (thickness_um / thickness_um[first]) ** 2
    return {
        "thickness_um": thickness_um,
        "time_to_threshold_s": times,
        "l_squared_reference_s": reference,
    }


def _diffusivity_study(case: Case, auto: bool) -> dict[str, Any]:
    """Section 6: penetration time against reference diffusivity.

    The swept quantity is ``D_ref``, the value quoted at ``T_ref`` --- that is what
    a material datasheet gives and what the user sets. The diffusivity actually
    governing transport is ``D(T)``, larger by the Arrhenius factor, so both are
    reported: labelling the axis "D" while sweeping ``D_ref`` would be wrong by a
    factor of 8 at the default 60 degC.
    """
    values = study_points(case.polymer.diffusivity_ref, DIFFUSIVITY_STUDY_SPAN)
    temperature = case.condition.temperature_k
    effective = np.array([
        case.polymer.with_diffusivity(d).diffusivity(temperature) for d in values
    ])
    times = np.array(
        [_threshold_time(case.with_diffusivity(d), auto) for d in values]
    )
    return {
        "d_ref": values,
        "diffusivity_at_temperature": effective,
        "time_to_threshold_s": times,
    }


# --- Section 8: comparison -------------------------------------------------

def _sensitivity(case: Case) -> dict[str, Any]:
    """Return local log-log sensitivities, or the model error if unavailable."""
    try:
        result = sensitivity_module.local_elasticities(_scalar_case(case))
    except HygroadhError as exc:
        return {"error": str(exc)}
    return {
        "response": result.response,
        "base_value": result.base_value,
        "elasticities": result.elasticities,
        "most_influential": result.most_influential,
        "relative_step": result.relative_step,
    }


# --- the whole payload ----------------------------------------------------

def simulate_payload(params: dict[str, Any]) -> dict[str, Any]:
    """Run everything the dashboard shows and return it as a JSON-safe payload."""
    started = time.perf_counter()
    case = case_from_params(params)
    auto = _choice(params, "duration_mode", ("auto", "manual")) == "auto"
    result = run(case)
    condition = case.condition

    time_s = result.time
    positive = time_s[time_s > 0.0]
    first_positive = float(positive[0]) if positive.size else 1.0

    depth = result.transport.depth
    profile = result.transport.profile_normalized
    if depth is None or profile is None:  # pragma: no cover - solvers always fill these
        raise HygroadhError("the solver returned no through-thickness profile")

    # Labelled snapshots at fixed Fourier numbers, so the four curves mean the
    # same thing whatever the thickness or temperature.
    fourier = result.transport.fourier_number
    snapshots = []
    for target, label in zip(SNAPSHOT_FOURIER, SNAPSHOT_LABELS):
        index = int(np.argmin(np.abs(fourier - target)))
        snapshots.append({
            "label": label,
            "fourier": float(fourier[index]),
            "time_s": float(time_s[index]),
            "profile": profile[index],
        })

    # A thinned set of frames for the time slider and the animation.
    stride = max(1, time_s.size // PROFILE_SLIDER_FRAMES)
    frames = np.arange(0, time_s.size, stride)
    if frames[-1] != time_s.size - 1:
        frames = np.append(frames, time_s.size - 1)

    saturation = result.saturation_pct

    # The frame at which the far face first reaches the threshold, so the
    # animation can stop there. Computed here rather than in the browser: the
    # page should not be re-deriving where the criterion is met from the curve it
    # was handed, or the two could disagree.
    level = case.criterion.normalized_level(saturation)
    threshold_frame: int | None = None
    if np.isfinite(level) and level < 1.0:
        far_face_frames = profile[frames][:, -1]
        reached = np.nonzero(far_face_frames >= level)[0]
        if reached.size:
            threshold_frame = int(reached[0])
    payload = {
        # echo of the resolved inputs, so a run is reproducible from the page
        "inputs": {
            "thickness_um": case.thickness * 1e6,
            "temperature_c": condition.temperature_k - KELVIN_OFFSET,
            "temperature_k": condition.temperature_k,
            "relative_humidity_pct": condition.relative_humidity * 100.0,
            "d_ref": case.polymer.diffusivity_ref,
            "t_ref_c": case.polymer.temperature_ref_k - KELVIN_OFFSET,
            "t_ref_k": case.polymer.temperature_ref_k,
            "ea_kj": case.polymer.activation_energy / 1e3,
            "threshold_basis": case.criterion.basis,
            "threshold_value": case.criterion.value,
            "exposure": condition.exposure,
            "duration_hours": hours(condition.duration),
            "duration_days": days(condition.duration),
            "m_ref_pct": case.polymer.isotherm.m_ref_pct,
        },
        "derived": {
            "diffusivity": result.diffusivity,
            "saturation_pct": saturation,
            "threshold_pct": result.threshold_pct,
            "threshold_normalized": case.criterion.normalized_level(saturation),
            "threshold_reachable": result.threshold_reachable,
            "threshold_note": case.criterion.unreachable_reason(saturation),
            "threshold_description": case.criterion.describe(saturation),
            "time_to_threshold_s": result.time_to_threshold,
            "time_to_half_uptake_s": result.time_to_half_uptake,
            "characteristic_time_s": (
                equivalent_sheet_thickness(case.thickness, condition.exposure) ** 2
                / result.diffusivity
            ),
            "solver": result.transport.solver,
        },
        "history": {
            "time_s": time_s,
            "time_hours": time_s / 3600.0,
            "first_positive_hour": hours(first_positive),
            "uptake_normalized": result.uptake_normalized,
            "uptake_pct": result.transport.uptake_pct,
            "far_face_normalized": result.interface_normalized,
            "far_face_pct": result.interface_pct,
        },
        "profile": {
            "depth_um": depth * 1e6,
            "snapshots": snapshots,
            "frame_time_s": time_s[frames],
            "frame_fourier": fourier[frames],
            "frames": profile[frames],
            "threshold_frame": threshold_frame,
        },
        "temperature": _temperature_curves(case, params, auto),
        "temperature_study": _temperature_study(case, params, auto),
        "thickness_study": _thickness_study(case, auto),
        "diffusivity_study": _diffusivity_study(case, auto),
        "sensitivity": _sensitivity(case),
        "study_window": "auto" if auto else "manual",
    }
    payload["elapsed_ms"] = (time.perf_counter() - started) * 1e3
    return json_safe(payload)


# --- CSV export -----------------------------------------------------------

def _csv(rows: list[list[Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _fmt(value: Any) -> Any:
    if value is None:
        return "not reached"
    if isinstance(value, float):
        return "not reached" if not math.isfinite(value) else f"{value:.10g}"
    return value


def export_history_csv(params: dict[str, Any]) -> str:
    """The time history: uptake and far-face moisture at every sample."""
    payload = simulate_payload(params)
    history = payload["history"]
    rows: list[list[Any]] = [[
        "time_s", "time_hours", "uptake_normalized", "uptake_pct",
        "far_face_normalized", "far_face_pct",
    ]]
    for index in range(len(history["time_s"])):
        rows.append([
            _fmt(history["time_s"][index]),
            _fmt(history["time_hours"][index]),
            _fmt(history["uptake_normalized"][index]),
            _fmt(history["uptake_pct"][index]),
            _fmt(history["far_face_normalized"][index]),
            _fmt(history["far_face_pct"][index]),
        ])
    return _csv(rows)


def export_studies_csv(params: dict[str, Any]) -> str:
    """All three parameter studies in one long-format table."""
    payload = simulate_payload(params)
    rows: list[list[Any]] = [[
        "study", "variable", "value", "unit", "temperature_k",
        "diffusivity_m2_s", "time_to_threshold_s", "time_to_threshold_hours",
    ]]

    for row in payload["temperature_study"]:
        seconds = row["time_to_threshold_s"]
        rows.append([
            "temperature", "temperature", _fmt(row["temperature_c"]), "degC",
            _fmt(row["temperature_k"]), _fmt(row["diffusivity"]), _fmt(seconds),
            _fmt(None if seconds is None else seconds / 3600.0),
        ])

    thickness = payload["thickness_study"]
    for index, value in enumerate(thickness["thickness_um"]):
        seconds = thickness["time_to_threshold_s"][index]
        rows.append([
            "thickness", "thickness", _fmt(value), "um",
            _fmt(payload["inputs"]["temperature_k"]),
            _fmt(payload["derived"]["diffusivity"]), _fmt(seconds),
            _fmt(None if seconds is None else seconds / 3600.0),
        ])

    diffusivity = payload["diffusivity_study"]
    for index, value in enumerate(diffusivity["d_ref"]):
        seconds = diffusivity["time_to_threshold_s"][index]
        rows.append([
            "diffusivity", "d_ref", _fmt(value), "m2/s",
            _fmt(payload["inputs"]["temperature_k"]),
            _fmt(diffusivity["diffusivity_at_temperature"][index]), _fmt(seconds),
            _fmt(None if seconds is None else seconds / 3600.0),
        ])
    return _csv(rows)


def export_summary_csv(params: dict[str, Any]) -> str:
    """Inputs and derived results as a two-column table, plus provenance notes."""
    payload = simulate_payload(params)
    inputs, derived = payload["inputs"], payload["derived"]
    rows: list[list[Any]] = [["quantity", "value", "unit", "kind"]]

    for key, unit in (
        ("thickness_um", "um"), ("temperature_c", "degC"), ("temperature_k", "K"),
        ("relative_humidity_pct", "%"), ("d_ref", "m2/s"), ("t_ref_c", "degC"),
        ("t_ref_k", "K"), ("ea_kj", "kJ/mol"), ("m_ref_pct", "wt%"),
        ("threshold_basis", "-"),
        ("threshold_value", "-"), ("exposure", "-"), ("duration_hours", "h"),
    ):
        rows.append([key, _fmt(inputs[key]), unit, "input"])

    for key, unit in (
        ("diffusivity", "m2/s"), ("saturation_pct", "wt%"),
        ("threshold_pct", "wt%"), ("threshold_normalized", "-"),
        ("characteristic_time_s", "s"), ("time_to_half_uptake_s", "s"),
        ("time_to_threshold_s", "s"), ("solver", "-"),
    ):
        rows.append([key, _fmt(derived[key]), unit, "calculated"])

    seconds = derived["time_to_threshold_s"]
    rows.append([
        "time_to_threshold_hours",
        _fmt(None if seconds is None else seconds / 3600.0), "h", "calculated",
    ])

    sensitivity = payload["sensitivity"]
    if "elasticities" in sensitivity:
        for factor, value in sensitivity["elasticities"].items():
            rows.append([
                f"elasticity_{factor}", _fmt(value), "dlny/dlnx", "calculated",
            ])

    rows.append([])
    rows.append(["note", "This model predicts moisture transport, not measured "
                 "adhesive strength.", "", ""])
    rows.append(["note", "d_ref and ea_kj are model parameters, not universal "
                 "material constants.", "", ""])
    rows.append(["note", derived["threshold_description"], "", ""])
    return _csv(rows)


EXPORTS = {
    "history": (export_history_csv, "hygroadh-history.csv"),
    "studies": (export_studies_csv, "hygroadh-studies.csv"),
    "summary": (export_summary_csv, "hygroadh-summary.csv"),
}


# --- page and server -----------------------------------------------------

# --- where the dashboard can be reached from ----------------------------

def is_wsl() -> bool:
    """Whether this is running under Windows Subsystem for Linux.

    Checked because WSL changes two things that matter: there is usually no
    Linux browser to open, and reaching the server from a Windows browser
    depends on WSL's localhost forwarding, which can be absent or misconfigured.
    Both the environment variables and the kernel string are checked --- the
    variables are missing under some init systems and inside some containers.
    """
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        release = Path("/proc/version").read_text()
    except OSError:
        return False
    lowered = release.lower()
    return "microsoft" in lowered or "wsl" in lowered


def local_ip() -> str | None:
    """Best guess at this host's routable address, or ``None``.

    Opens a UDP socket toward an unroutable address and reads back the local
    endpoint the kernel selected. No packet is sent, and nothing needs to be
    listening, so this works offline; it just asks the routing table which
    interface would be used.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        return str(probe.getsockname()[0])
    except OSError:  # pragma: no cover - only on a host with no routes at all
        return None
    finally:
        probe.close()


def access_urls(host: str, port: int) -> list[tuple[str, str]]:
    """URLs the dashboard can be opened at, each with who it works for."""
    urls: list[tuple[str, str]] = []
    if host in ("0.0.0.0", "::", ""):
        urls.append((f"http://localhost:{port}/", "from this machine"))
        address = local_ip()
        if address:
            urls.append((
                f"http://{address}:{port}/",
                "from Windows, or from another machine on this network",
            ))
    elif host in ("127.0.0.1", "localhost", "::1"):
        urls.append((f"http://localhost:{port}/", "from this machine"))
    else:
        urls.append((f"http://{host}:{port}/", "as bound"))
    return urls


def open_in_browser(url: str) -> bool:
    """Try to open ``url``, returning whether a launcher was actually invoked.

    Under WSL the Linux ``webbrowser`` module usually has nothing to open --- and
    can block --- so the Windows-side helpers are tried first. Everything is fire
    and forget: ``explorer.exe`` in particular returns a non-zero exit status
    even when it succeeds, so its result cannot be trusted.
    """
    if is_wsl():
        for launcher in (["wslview", url], ["explorer.exe", url]):
            if shutil.which(launcher[0]) is None:
                continue
            try:
                subprocess.Popen(
                    launcher, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return True
            except OSError:  # pragma: no cover - depends on the host
                continue
        return False
    try:
        import webbrowser

        return bool(webbrowser.open(url))
    except Exception:  # pragma: no cover - headless environments
        return False


def render_page(defaults: dict[str, Any] | None = None) -> bytes:
    """Read the page and inject the default parameters into it."""
    page = PAGE_PATH.read_text(encoding="utf-8")
    marker = "const DEFAULTS = window.__HYGROADH_DEFAULTS__ || {};"
    if marker not in page:  # pragma: no cover - guards against page edits
        raise HygroadhError(
            f"{PAGE_PATH.name} is missing the defaults marker; the page and "
            "dashboard.py have drifted apart"
        )
    injected = "const DEFAULTS = " + json.dumps(
        json_safe(DEFAULT_PARAMS if defaults is None else defaults)
    ) + ";"
    return page.replace(marker, injected).encode("utf-8")


class DashboardHandler(BaseHTTPRequestHandler):
    """Request handler for the dashboard. Stateless: no shared mutable state."""

    server_version = "hygroadh"
    protocol_version = "HTTP/1.1"
    quiet = True
    defaults: dict[str, Any] | None = None

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        if not self.quiet:  # pragma: no cover - only for debugging
            super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, content_type: str,
              extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                self._send(200, render_page(self.defaults), "text/html; charset=utf-8")
            except (HygroadhError, OSError) as exc:
                self._send_json(500, {"error": str(exc)})
            return
        if path == "/api/defaults":
            self._send_json(200, json_safe(self.defaults or DEFAULT_PARAMS))
            return
        if path == "/api/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": f"no such path: {path}"})

    def _read_params(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return None
        if length < 0:
            self._send_json(400, {"error": "invalid Content-Length"})
            return None
        if length > MAX_REQUEST_BYTES:
            self._send_json(413, {
                "error": f"request body exceeds {MAX_REQUEST_BYTES} bytes"
            })
            return None
        raw = self.rfile.read(length) if length else b"{}"
        try:
            params = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"invalid JSON body: {exc}"})
            return None
        if not isinstance(params, dict):
            self._send_json(400, {"error": "request body must be a JSON object"})
            return None
        return params

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        export = None
        if path.startswith("/api/export/"):
            export = path[len("/api/export/"):]
            if export not in EXPORTS:
                self._send_json(404, {
                    "error": f"unknown export {export!r}; "
                             f"available: {sorted(EXPORTS)}"
                })
                return
        elif path != "/api/simulate":
            self._send_json(404, {"error": f"no such path: {path}"})
            return

        params = self._read_params()
        if params is None:
            return
        try:
            if export is None:
                self._send_json(200, simulate_payload(params))
            else:
                builder, filename = EXPORTS[export]
                body = builder(params).encode("utf-8")
                self._send(
                    200, body, "text/csv; charset=utf-8",
                    {"Content-Disposition": f'attachment; filename="{filename}"'},
                )
        except HygroadhError as exc:
            # An input combination the model rejects is user error, not a crash.
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - unexpected, still no crash
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})


@dataclass
class Dashboard:
    """A bound dashboard server, ready to serve."""

    server: ThreadingHTTPServer
    host: str
    port: int

    @property
    def url(self) -> str:
        display = "localhost" if self.host in ("127.0.0.1", "0.0.0.0", "") else self.host
        return f"http://{display}:{self.port}/"

    def serve_forever(self) -> None:
        self.server.serve_forever()

    def shutdown(self) -> None:
        self.server.shutdown()

    def close(self) -> None:
        self.server.server_close()

    def __enter__(self) -> "Dashboard":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def build_dashboard(host: str = "127.0.0.1", port: int = 8765,
                    max_attempts: int = 20, quiet: bool = True,
                    defaults: dict[str, Any] | None = None) -> Dashboard:
    """Bind a dashboard server, walking forward if the port is taken.

    Passing ``port=0`` asks the operating system for any free port, which is what
    the tests use so they never collide with a real dashboard.
    """
    handler = type(
        "BoundHandler", (DashboardHandler,), {"quiet": quiet, "defaults": defaults}
    )
    last_error: OSError | None = None
    attempts = 1 if port == 0 else max(1, max_attempts)
    for offset in range(attempts):
        candidate = port if port == 0 else port + offset
        try:
            server = ThreadingHTTPServer((host, candidate), handler)
        except OSError as exc:
            if exc.errno not in (98, 48):
                raise
            last_error = exc
            continue
        server.daemon_threads = True
        return Dashboard(server=server, host=host, port=server.server_address[1])
    raise HygroadhError(
        f"could not bind a port in {port}..{port + attempts - 1} on {host}: "
        f"{last_error}"
    )


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False,
          quiet: bool = True, defaults: dict[str, Any] | None = None) -> None:
    """Start the dashboard and block until interrupted.

    No browser is opened unless asked for: under WSL there is usually no Linux
    browser to open, and printing the URL for the user to paste is both more
    reliable and less presumptuous than launching something.
    """
    dashboard = build_dashboard(host=host, port=port, quiet=quiet, defaults=defaults)
    urls = access_urls(host, dashboard.port)

    lines = [f"hygroadh dashboard is serving on port {dashboard.port}.", ""]
    label = "  open in your browser:"
    for url, note in urls:
        lines.append(f"{label} {url:<32} ({note})")
        label = " " * len(label)
    notes: list[str] = []
    if host in ("0.0.0.0", "::", ""):
        notes.append(
            "  Bound to all interfaces. The dashboard has no authentication, so\n"
            "  anyone who can reach this host can drive it."
        )
    elif is_wsl():
        notes.append(
            "  Running under WSL. Windows forwards localhost into WSL, so the URL\n"
            "  above should work in a Windows browser. If it does not, stop this\n"
            "  and rerun with:\n"
            "      ./hygroadh serve --host 0.0.0.0"
        )
        address = local_ip()
        if address:
            notes.append(
                f"  then open  http://{address}:{dashboard.port}/  from Windows."
            )
    if notes:
        lines.append("")
        lines.extend(notes)
    lines.append("")
    lines.append("  Physics runs in Python; the page only draws it. Ctrl-C to stop.")
    print("\n".join(lines), flush=True)

    if open_browser and not open_in_browser(urls[0][0]):
        print("  (could not launch a browser; open the URL above by hand)",
              flush=True)
    try:
        dashboard.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        dashboard.close()
