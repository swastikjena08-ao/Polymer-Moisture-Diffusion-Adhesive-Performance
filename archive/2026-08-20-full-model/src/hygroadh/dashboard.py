"""Local web dashboard for exploring the model interactively.

Runs a small stdlib HTTP server and serves a self-contained page. The browser
holds no physics: every slider move posts the parameters back and the Python
model computes the answer. That is deliberate --- a JavaScript reimplementation
would be a second model free to drift from the tested one, and the numbers on
screen would stop being the numbers the test suite verifies.

Endpoints
---------
``GET  /``                 the dashboard page, with defaults injected
``POST /api/simulate``     one case: time histories, design map, sensitivity
``GET  /api/defaults``     the default parameter set
``GET  /api/health``       liveness probe

Only ``http.server`` and ``json`` from the standard library are used, so the
dashboard has no dependencies beyond numpy.
"""

from __future__ import annotations

import json
import math
import socket
import time
import webbrowser
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from . import sensitivity as sensitivity_module
from . import sweep as sweep_module
from .adhesion import AdhesionModel
from .materials import Polymer
from .simulate import Case, ExposureCondition, run
from .sorption import PowerLawIsotherm
from .units import (
    KELVIN_OFFSET,
    HygroadhError,
    days,
    from_days,
)

PAGE_PATH = Path(__file__).with_name("dashboard.html")

#: Starting parameter set, also the "reset" target in the page. Chosen so the
#: opening view separates the three adhesion mechanisms in time rather than
#: having them all fire at once when the bondline wets.
DEFAULT_PARAMS: dict[str, Any] = {
    "thickness_um": 200.0,
    "exposure": "one_sided",
    "temperature_c": 60.0,
    "relative_humidity": 0.85,
    "duration_days": 90.0,
    "diffusivity_ref": 1.0e-13,
    "activation_energy": 50000.0,
    "conc_dependence": 0.0,
    "m_ref_pct": 2.4,
    "isotherm_exponent": 0.8,
    "enthalpy_sorption": 6000.0,
    "tg_dry_c": 140.0,
    "tg_depression_per_pct": 12.0,
    "plasticization_gain": 0.15,
    "plasticization_floor": 0.55,
    "hydrolysis_rate_ref": 7.0e-8,
    "hydrolysis_max_loss": 0.55,
    "ari_threshold": 0.7,
    "profile_fraction": 0.3,
}

#: Resolution of the thickness x temperature design map. A case that needs the
#: finite-volume solver costs roughly ten times as much per point as one the
#: exact series can handle, so the grid is coarsened rather than letting a
#: slider drag stall for over a second.
MAP_POINTS_EXACT = 7
MAP_POINTS_NUMERIC = 5
MAP_THICKNESS_SPAN = 4.0
#: Output samples per map point. Coarser than the displayed run because the map
#: needs only one interpolated threshold crossing per point, not a smooth curve.
#: Costs 3e-4 relative error in the threshold time, against 2e-4 at the
#: displayed resolution.
MAP_TIME_POINTS = 160
#: Cells across the film for scalar-only runs. Only the finite-volume solver
#: uses these, and dropping from 60 to 30 changes the threshold time by 5e-5
#: relative while cutting the cost by a factor of about 1.7.
SCALAR_CELLS_NUMERIC = 30
#: Kelvin of headroom kept below the dry Tg, where ARI has no baseline.
TG_HEADROOM = 30.0


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


def _number(params: dict[str, Any], key: str) -> float:
    raw = params.get(key, DEFAULT_PARAMS[key])
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise HygroadhError(f"{key} must be a number, got {raw!r}") from exc


def case_from_params(params: dict[str, Any]) -> Case:
    """Build a :class:`~hygroadh.simulate.Case` from a flat parameter mapping.

    Validation lives in the model's own constructors, so a nonsensical slider
    combination surfaces as a :class:`~hygroadh.units.HygroadhError` with a
    message the page can display verbatim.
    """
    exposure = params.get("exposure", DEFAULT_PARAMS["exposure"])
    if exposure not in ("one_sided", "two_sided"):
        raise HygroadhError(f"exposure must be one_sided or two_sided, got {exposure!r}")

    polymer = Polymer(
        diffusivity_ref=_number(params, "diffusivity_ref"),
        activation_energy=_number(params, "activation_energy"),
        temperature_ref_k=298.15,
        isotherm=PowerLawIsotherm(
            m_ref_pct=_number(params, "m_ref_pct"),
            rh_ref=1.0,
            exponent=_number(params, "isotherm_exponent"),
            enthalpy_sorption=_number(params, "enthalpy_sorption"),
            temperature_ref_k=298.15,
        ),
        tg_dry_k=_number(params, "tg_dry_c") + KELVIN_OFFSET,
        tg_model="linear",
        tg_depression_per_pct=_number(params, "tg_depression_per_pct"),
        tg_floor_k=KELVIN_OFFSET - 40.0,
        conc_dependence=_number(params, "conc_dependence"),
    )
    condition = ExposureCondition(
        temperature_k=_number(params, "temperature_c") + KELVIN_OFFSET,
        duration=from_days(_number(params, "duration_days")),
        relative_humidity=_number(params, "relative_humidity"),
        exposure=exposure,
        n_times=400,
    )
    adhesion = AdhesionModel(
        plasticization_gain=_number(params, "plasticization_gain"),
        plasticization_floor=_number(params, "plasticization_floor"),
        hydrolysis_rate_ref=_number(params, "hydrolysis_rate_ref"),
        hydrolysis_max_loss=_number(params, "hydrolysis_max_loss"),
    )
    return Case(
        thickness=_number(params, "thickness_um") * 1e-6,
        condition=condition,
        polymer=polymer,
        adhesion=adhesion,
        n_cells=60,
        ari_threshold=_number(params, "ari_threshold"),
        name="dashboard",
    )


def params_from_case(case: Case) -> dict[str, Any]:
    """Project a :class:`~hygroadh.simulate.Case` back onto dashboard parameters.

    The inverse of :func:`case_from_params`, so a config file can seed the
    sliders. It is deliberately lossy: the dashboard exposes a fixed subset of
    the model, and anything outside that subset (humidity schedules, surface
    resistance, the Fox Tg model) has no slider to land on. Those settings are
    dropped here rather than silently misrepresented, so a config using them
    should be run through the CLI instead.
    """
    polymer = case.polymer
    condition = case.condition
    adhesion = case.adhesion
    params = dict(DEFAULT_PARAMS)
    params.update({
        "thickness_um": case.thickness * 1e6,
        "exposure": condition.exposure,
        "temperature_c": condition.temperature_k - KELVIN_OFFSET,
        "relative_humidity": condition.relative_humidity,
        "duration_days": days(condition.duration),
        "diffusivity_ref": polymer.diffusivity(298.15),
        "activation_energy": polymer.activation_energy,
        "conc_dependence": polymer.conc_dependence,
        "m_ref_pct": polymer.isotherm.saturation_pct(298.15, 1.0),
        "isotherm_exponent": polymer.isotherm.exponent,
        "enthalpy_sorption": polymer.isotherm.enthalpy_sorption,
        "tg_dry_c": polymer.tg_dry_k - KELVIN_OFFSET,
        "tg_depression_per_pct": polymer.tg_depression_per_pct,
        "plasticization_gain": adhesion.plasticization_gain,
        "plasticization_floor": adhesion.plasticization_floor,
        "hydrolysis_rate_ref": adhesion.hydrolysis_rate_ref,
        "hydrolysis_max_loss": adhesion.hydrolysis_max_loss,
        "ari_threshold": case.ari_threshold,
    })
    return params


def unsupported_dashboard_features(case: Case) -> list[str]:
    """Name the parts of a case the dashboard's sliders cannot represent."""
    missing = []
    if case.condition.humidity_schedule is not None:
        missing.append("humidity schedule")
    if case.condition.surface_transfer is not None:
        missing.append("surface mass-transfer resistance")
    if case.polymer.tg_model != "linear":
        missing.append(f"{case.polymer.tg_model!r} Tg model")
    if case.polymer.temperature_ref_k != 298.15:
        missing.append("non-25C reference temperature")
    return missing


def _scalar_case(case: Case, n_times: int | None = None) -> Case:
    """A copy of ``case`` tuned for extracting scalars cheaply.

    Drops the through-thickness profile, optionally coarsens the output grid,
    and thins the spatial grid when --- and only when --- the finite-volume
    solver is in play. For the exact series the cell count merely sets the
    profile sampling, which is already discarded here.
    """
    lean = case.without_profile(n_times)
    if case.requires_finite_volume:
        lean = replace(lean, n_cells=SCALAR_CELLS_NUMERIC)
    return lean


def _design_map(case: Case) -> dict[str, Any]:
    """Sweep thickness against temperature for the design map.

    The temperature axis stops short of the dry glass transition: above it the
    adhesion model has no dry baseline and correctly refuses to evaluate, so
    including those points would fail the whole request rather than shade one
    corner of a plot.
    """
    points = MAP_POINTS_NUMERIC if case.requires_finite_volume else MAP_POINTS_EXACT
    centre = case.thickness
    thickness = np.geomspace(
        centre / MAP_THICKNESS_SPAN, centre * MAP_THICKNESS_SPAN, points
    )
    ceiling_c = case.polymer.tg_dry_k - TG_HEADROOM - KELVIN_OFFSET
    low_c = 5.0
    high_c = min(95.0, ceiling_c)
    if high_c <= low_c:
        return {
            "thickness_um": [],
            "temperature_c": [],
            "t_ari_days": [],
            "note": "no temperature range below the dry glass transition",
        }
    temperature_c = np.linspace(low_c, high_c, points)
    result = sweep_module.run_sweep(
        _scalar_case(case, MAP_TIME_POINTS),
        {"thickness": thickness, "temperature_k": temperature_c + KELVIN_OFFSET},
    )
    surface = result.surface("time_to_ari_threshold_s") / 86400.0
    return {
        "thickness_um": (thickness * 1e6).tolist(),
        "temperature_c": temperature_c.tolist(),
        "t_ari_days": surface.tolist(),
        "finite_fraction": result.finite_fraction("time_to_ari_threshold_s"),
    }


def _sensitivity(case: Case) -> dict[str, Any]:
    """Local elasticities, degrading to an explanation rather than an error."""
    try:
        result = sensitivity_module.local_elasticities(_scalar_case(case))
    except HygroadhError as exc:
        return {"error": str(exc)}
    return {
        "response": result.response,
        "base_value": result.base_value,
        "elasticities": result.elasticities,
        "most_influential": result.most_influential,
    }


def simulate_payload(params: dict[str, Any]) -> dict[str, Any]:
    """Run everything the dashboard shows and return it as a JSON-safe payload."""
    started = time.perf_counter()
    case = case_from_params(params)
    result = run(case)

    time_s = result.time
    positive = time_s[time_s > 0.0]
    first_positive = float(positive[0]) if positive.size else 1.0

    fraction = float(np.clip(_number(params, "profile_fraction"), 0.0, 1.0))
    target = fraction * case.condition.duration
    index = int(np.argmin(np.abs(time_s - target)))
    profile = result.transport.profile_normalized
    depth = result.transport.depth
    if profile is None or depth is None:  # pragma: no cover - solvers always fill these
        profile_row, depth_um = [], []
    else:
        profile_row = profile[index].tolist()
        depth_um = (depth * 1e6).tolist()

    interface_pct_final = float(result.adhesion.interface_pct[-1])
    payload = {
        "time_s": time_s,
        "time_days": days_array(time_s),
        "first_positive_day": days(first_positive),
        "uptake": result.uptake_normalized,
        "interface": result.interface_normalized,
        "uptake_pct": result.transport.uptake_pct,
        "ari": result.adhesion.index,
        "plasticization": result.adhesion.plasticization,
        "thermal": result.adhesion.thermal,
        "hydrolysis": result.adhesion.hydrolysis,
        "damage": result.adhesion.damage,
        "depth_um": depth_um,
        "profile": profile_row,
        "profile_time_s": float(time_s[index]),
        "tg_dry_c": float(case.polymer.tg_dry_k - KELVIN_OFFSET),
        "tg_wet_final_c": float(result.adhesion.glass_transition_k[-1] - KELVIN_OFFSET),
        "interface_pct_final": interface_pct_final,
        "summary": result.summary(),
        "map": _design_map(case),
        "sensitivity": _sensitivity(case),
    }
    payload["elapsed_ms"] = (time.perf_counter() - started) * 1e3
    return json_safe(payload)


def days_array(seconds: np.ndarray) -> np.ndarray:
    """Convert an array of seconds to days."""
    return np.asarray(seconds, dtype=float) / 86400.0


def render_page(defaults: dict[str, Any] | None = None) -> bytes:
    """Read the page and inject the default parameters into it."""
    page = PAGE_PATH.read_text()
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

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
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

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/api/simulate":
            self._send_json(404, {"error": f"no such path: {path}"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        raw = self.rfile.read(length) if length else b"{}"
        try:
            params = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"invalid JSON body: {exc}"})
            return
        if not isinstance(params, dict):
            self._send_json(400, {"error": "request body must be a JSON object"})
            return
        try:
            self._send_json(200, simulate_payload(params))
        except HygroadhError as exc:
            # A slider combination the model rejects is user error, not a crash.
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
            if exc.errno not in (socket.EADDRINUSE if hasattr(socket, "EADDRINUSE") else (98,), 98, 48):
                raise
            last_error = exc
            continue
        server.daemon_threads = True
        return Dashboard(server=server, host=host, port=server.server_address[1])
    raise HygroadhError(
        f"could not bind a port in {port}..{port + attempts - 1} on {host}: {last_error}"
    )


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True,
          quiet: bool = True, defaults: dict[str, Any] | None = None) -> None:
    """Start the dashboard and block until interrupted."""
    dashboard = build_dashboard(host=host, port=port, quiet=quiet, defaults=defaults)
    # flush=True so the URL appears immediately even when stdout is redirected
    # to a file or pipe, where Python would otherwise buffer it indefinitely.
    print(f"hygroadh dashboard: {dashboard.url}", flush=True)
    print("physics runs in Python; the page only draws it. Ctrl-C to stop.", flush=True)
    if open_browser:
        try:
            webbrowser.open(dashboard.url)
        except Exception:  # pragma: no cover - headless environments
            pass
    try:
        dashboard.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        dashboard.close()
