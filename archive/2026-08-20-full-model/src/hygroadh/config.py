"""Load and validate case definitions from YAML or JSON.

Two choices shape this module. First, **unknown keys are errors**: a silently
ignored ``tg_depresion_per_pct`` would leave the run using a default and quietly
report the wrong physics, which is far worse than a failed load. Second, keys
accept the units people actually write --- ``thickness_um``, ``temperature_c``,
``duration_days`` --- because forcing metres, Kelvin, and seconds into a hand-
edited file is how sign-and-magnitude mistakes get made.

JSON is always available; YAML is imported lazily so the package works without
PyYAML installed.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .adhesion import AdhesionModel
from .diffusion.fd import Schedule
from .materials import Polymer
from .simulate import Case, ExposureCondition
from .sorption import PowerLawIsotherm
from .units import (
    KELVIN_OFFSET,
    ConfigError,
    HygroadhError,
    MissingDependencyError,
    from_days,
    from_hours,
)

_MISSING = object()


@contextmanager
def _at(path: str) -> Iterator[None]:
    """Re-raise a model validation failure as a ConfigError naming its location.

    The dataclasses validate their own arguments and raise
    :class:`~hygroadh.units.PhysicsError`, which is right for library callers but
    unhelpful when the value came from line 30 of a YAML file. Wrapping the
    construction turns "relative_humidity must lie in [0, 1]" into
    "exposure: relative_humidity must lie in [0, 1]", so the reader knows which
    section to edit.
    """
    try:
        yield
    except ConfigError:
        raise
    except HygroadhError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{path or 'config'} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _pop_number(data: dict, key: str, path: str, default: Any = _MISSING) -> Any:
    if key not in data:
        if default is _MISSING:
            raise ConfigError(f"{path}.{key} is required")
        return default
    value = data.pop(key)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}.{key} must be a number, got {value!r}") from exc


def _pop_int(data: dict, key: str, path: str, default: Any) -> int:
    value = _pop_number(data, key, path, default)
    if float(value) != int(value):
        raise ConfigError(f"{path}.{key} must be a whole number, got {value!r}")
    return int(value)


def _pop_str(data: dict, key: str, path: str, default: Any) -> Any:
    value = data.pop(key, default)
    if value is not None and not isinstance(value, str):
        raise ConfigError(f"{path}.{key} must be a string, got {value!r}")
    return value


def _pop_temperature(data: dict, base: str, path: str, default: Any = _MISSING) -> Any:
    """Accept ``<base>_k`` in Kelvin or ``<base>_c`` in Celsius, never both."""
    kelvin_key, celsius_key = f"{base}_k", f"{base}_c"
    has_k, has_c = kelvin_key in data, celsius_key in data
    if has_k and has_c:
        raise ConfigError(
            f"{path} sets both {kelvin_key} and {celsius_key}; give exactly one"
        )
    if has_k:
        return _pop_number(data, kelvin_key, path)
    if has_c:
        return _pop_number(data, celsius_key, path) + KELVIN_OFFSET
    if default is _MISSING:
        raise ConfigError(f"{path}.{kelvin_key} or {path}.{celsius_key} is required")
    return default


def _pop_length(data: dict, base: str, path: str, default: Any = _MISSING) -> Any:
    """Accept a length in metres, millimetres, or micrometres."""
    keys = {base: 1.0, f"{base}_mm": 1e-3, f"{base}_um": 1e-6}
    present = [key for key in keys if key in data]
    if len(present) > 1:
        raise ConfigError(f"{path} sets more than one of {sorted(keys)}; give exactly one")
    if present:
        key = present[0]
        return _pop_number(data, key, path) * keys[key]
    if default is _MISSING:
        raise ConfigError(f"{path}.{base} is required (or {base}_mm / {base}_um)")
    return default


def _pop_duration(data: dict, path: str) -> float:
    """Accept a duration in seconds, hours, or days."""
    keys = {"duration": 1.0, "duration_hours": from_hours(1.0), "duration_days": from_days(1.0)}
    present = [key for key in keys if key in data]
    if len(present) > 1:
        raise ConfigError(f"{path} sets more than one of {sorted(keys)}; give exactly one")
    if not present:
        raise ConfigError(f"{path}.duration is required (or duration_hours / duration_days)")
    key = present[0]
    return _pop_number(data, key, path) * keys[key]


def _no_leftovers(data: dict, path: str) -> None:
    if data:
        raise ConfigError(
            f"unknown key(s) in {path}: {', '.join(sorted(data))}. "
            "Unknown keys are rejected because a silently ignored one would "
            "leave the run using a default value."
        )


def _pop_list(data: dict, key: str, path: str) -> np.ndarray | None:
    if key not in data:
        return None
    value = data.pop(key)
    if isinstance(value, (int, float)):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(f"{path}.{key} must be a non-empty list of numbers")
    try:
        return np.array([float(item) for item in value], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}.{key} must contain only numbers") from exc


@dataclass(frozen=True)
class SweepAxes:
    """Design-space axes to sweep, each ``None`` when that axis is held fixed."""

    thickness: np.ndarray | None = None
    temperature_k: np.ndarray | None = None
    diffusivity: np.ndarray | None = None

    @property
    def is_empty(self) -> bool:
        return all(
            axis is None
            for axis in (self.thickness, self.temperature_k, self.diffusivity)
        )


@dataclass(frozen=True)
class Configuration:
    """A validated case plus the optional analysis settings that accompany it."""

    case: Case
    sweep: SweepAxes = SweepAxes()


def _build_isotherm(data: dict, path: str) -> PowerLawIsotherm:
    with _at(path):
        isotherm = PowerLawIsotherm(
            m_ref_pct=_pop_number(data, "m_ref_pct", path),
            rh_ref=_pop_number(data, "rh_ref", path, 1.0),
            exponent=_pop_number(data, "exponent", path, 1.0),
            enthalpy_sorption=_pop_number(data, "enthalpy_sorption", path, 0.0),
            temperature_ref_k=_pop_temperature(data, "temperature_ref", path, 298.15),
        )
    _no_leftovers(data, path)
    return isotherm


def _build_polymer(data: dict, path: str) -> Polymer:
    isotherm_data = _mapping(data.pop("isotherm", None), f"{path}.isotherm")
    if not isotherm_data:
        raise ConfigError(f"{path}.isotherm is required")
    with _at(path):
        polymer = Polymer(
            name=_pop_str(data, "name", path, "polymer"),
            diffusivity_ref=_pop_number(data, "diffusivity_ref", path),
            activation_energy=_pop_number(data, "activation_energy", path, 50.0e3),
            temperature_ref_k=_pop_temperature(data, "temperature_ref", path, 298.15),
            isotherm=_build_isotherm(isotherm_data, f"{path}.isotherm"),
            tg_dry_k=_pop_temperature(data, "tg_dry", path),
            tg_model=_pop_str(data, "tg_model", path, "linear"),
            tg_depression_per_pct=_pop_number(data, "tg_depression_per_pct", path, 15.0),
            tg_floor_k=_pop_temperature(data, "tg_floor", path, KELVIN_OFFSET),
            conc_dependence=_pop_number(data, "conc_dependence", path, 0.0),
        )
    _no_leftovers(data, path)
    return polymer


def _build_schedule(data: dict, path: str) -> Schedule | None:
    """Build a humidity schedule from either explicit knots or a cycle spec."""
    if not data:
        return None
    kind = _pop_str(data, "type", path, "knots")
    if kind == "knots":
        times = _pop_list(data, "times", path)
        values = _pop_list(data, "relative_humidity", path)
        unit = _pop_str(data, "time_unit", path, "s")
        if times is None or values is None:
            raise ConfigError(
                f"{path} of type 'knots' needs both times and relative_humidity"
            )
        scale = {"s": 1.0, "h": from_hours(1.0), "d": from_days(1.0)}.get(unit)
        if scale is None:
            raise ConfigError(f"{path}.time_unit must be 's', 'h' or 'd', got {unit!r}")
        _no_leftovers(data, path)
        try:
            return Schedule(times * scale, values)
        except HygroadhError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
    if kind == "cycle":
        unit = _pop_str(data, "time_unit", path, "s")
        scale = {"s": 1.0, "h": from_hours(1.0), "d": from_days(1.0)}.get(unit)
        if scale is None:
            raise ConfigError(f"{path}.time_unit must be 's', 'h' or 'd', got {unit!r}")
        high = _pop_number(data, "high", path)
        low = _pop_number(data, "low", path)
        period = _pop_number(data, "period", path) * scale
        n_cycles = _pop_int(data, "n_cycles", path, 1)
        duty = _pop_number(data, "duty", path, 0.5)
        ramp = _pop_number(data, "ramp", path, 0.0) * scale
        _no_leftovers(data, path)
        try:
            return Schedule.cycle(high, low, period, n_cycles, duty=duty, ramp=ramp)
        except HygroadhError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
    raise ConfigError(f"{path}.type must be 'knots' or 'cycle', got {kind!r}")


def _build_condition(data: dict, path: str) -> ExposureCondition:
    schedule = _build_schedule(
        _mapping(data.pop("humidity_schedule", None), f"{path}.humidity_schedule"),
        f"{path}.humidity_schedule",
    )
    with _at(path):
        condition = ExposureCondition(
            temperature_k=_pop_temperature(data, "temperature", path),
            duration=_pop_duration(data, path),
            relative_humidity=_pop_number(data, "relative_humidity", path, 1.0),
            humidity_schedule=schedule,
            exposure=_pop_str(data, "exposure", path, "one_sided"),
            surface_transfer=data.pop("surface_transfer", None),
            n_times=_pop_int(data, "n_times", path, 240),
            time_spacing=_pop_str(data, "time_spacing", path, "auto"),
        )
    _no_leftovers(data, path)
    return condition


def _build_adhesion(data: dict, path: str) -> AdhesionModel:
    with _at(path):
        model = AdhesionModel(
            moisture_reference_pct=_pop_number(data, "moisture_reference_pct", path, 1.0),
            plasticization_gain=_pop_number(data, "plasticization_gain", path, 0.15),
            plasticization_exponent=_pop_number(data, "plasticization_exponent", path, 1.0),
            plasticization_floor=_pop_number(data, "plasticization_floor", path, 0.55),
            tg_offset=_pop_number(data, "tg_offset", path, 20.0),
            tg_width=_pop_number(data, "tg_width", path, 8.0),
            hydrolysis_rate_ref=_pop_number(data, "hydrolysis_rate_ref", path, 7.0e-8),
            hydrolysis_activation=_pop_number(data, "hydrolysis_activation", path, 60.0e3),
            hydrolysis_exponent=_pop_number(data, "hydrolysis_exponent", path, 1.0),
            hydrolysis_max_loss=_pop_number(data, "hydrolysis_max_loss", path, 0.55),
            temperature_ref_k=_pop_temperature(data, "temperature_ref", path, 298.15),
        )
    _no_leftovers(data, path)
    return model


def _build_sweep(data: dict, path: str, polymer: Polymer) -> SweepAxes:
    if not data:
        return SweepAxes()
    if "thickness" in data:
        raise ConfigError(
            f"{path}.thickness is not a sweep axis; use thickness_um or "
            "thickness_mm with a list of values"
        )
    thickness_values = _pop_list(data, "thickness_um", path)
    if thickness_values is not None:
        thickness_values = thickness_values * 1e-6
    else:
        millimetres = _pop_list(data, "thickness_mm", path)
        if millimetres is not None:
            thickness_values = millimetres * 1e-3

    celsius = _pop_list(data, "temperature_c", path)
    kelvin = _pop_list(data, "temperature_k", path)
    if celsius is not None and kelvin is not None:
        raise ConfigError(f"{path} sets both temperature_c and temperature_k")
    temperature = celsius + KELVIN_OFFSET if celsius is not None else kelvin

    diffusivity = _pop_list(data, "diffusivity", path)
    scale = _pop_list(data, "diffusivity_scale", path)
    if diffusivity is not None and scale is not None:
        raise ConfigError(f"{path} sets both diffusivity and diffusivity_scale")
    if scale is not None:
        diffusivity = scale * polymer.diffusivity_ref

    _no_leftovers(data, path)
    return SweepAxes(
        thickness=thickness_values, temperature_k=temperature, diffusivity=diffusivity
    )


def build_configuration(raw: dict[str, Any]) -> Configuration:
    """Build a validated :class:`Configuration` from a plain dictionary."""
    data = _mapping(raw, "config")
    name = _pop_str(data, "name", "config", "case")

    film = _mapping(data.pop("film", None), "film")
    if not film:
        raise ConfigError("film section is required")
    thickness = _pop_length(film, "thickness", "film")
    film_exposure = _pop_str(film, "exposure", "film", None)
    _no_leftovers(film, "film")

    polymer_data = _mapping(data.pop("polymer", None), "polymer")
    if not polymer_data:
        raise ConfigError("polymer section is required")
    polymer = _build_polymer(polymer_data, "polymer")

    exposure_data = _mapping(data.pop("exposure", None), "exposure")
    if not exposure_data:
        raise ConfigError("exposure section is required")
    if film_exposure is not None:
        exposure_data.setdefault("exposure", film_exposure)
    condition = _build_condition(exposure_data, "exposure")

    adhesion = _build_adhesion(
        _mapping(data.pop("adhesion", None), "adhesion"), "adhesion"
    )

    solver_data = _mapping(data.pop("solver", None), "solver")
    solver = _pop_str(solver_data, "method", "solver", "auto")
    n_cells = _pop_int(solver_data, "n_cells", "solver", 80)
    _no_leftovers(solver_data, "solver")

    threshold = _pop_number(data, "ari_threshold", "config", 0.8)
    sweep = _build_sweep(_mapping(data.pop("sweep", None), "sweep"), "sweep", polymer)
    _no_leftovers(data, "config")

    with _at("config"):
        case = Case(
            thickness=thickness,
            condition=condition,
            polymer=polymer,
            adhesion=adhesion,
            solver=solver,
            n_cells=n_cells,
            ari_threshold=threshold,
            name=name,
        )
    return Configuration(case=case, sweep=sweep)


def load_raw(path: str | Path) -> dict[str, Any]:
    """Read a YAML or JSON file into a dictionary, chosen by extension."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"config file not found: {file_path}")
    text = file_path.read_text()
    suffix = file_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise MissingDependencyError(
                "reading YAML configs needs PyYAML (pip install pyyaml); "
                "JSON configs work without it"
            ) from exc
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{file_path} is not valid YAML: {exc}") from exc
    elif suffix == ".json":
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{file_path} is not valid JSON: {exc}") from exc
    else:
        raise ConfigError(
            f"unsupported config extension {suffix!r}; use .yaml, .yml, or .json"
        )
    return _mapping(loaded, str(file_path))


def load_configuration(path: str | Path) -> Configuration:
    """Read and validate a configuration file."""
    return build_configuration(load_raw(path))
