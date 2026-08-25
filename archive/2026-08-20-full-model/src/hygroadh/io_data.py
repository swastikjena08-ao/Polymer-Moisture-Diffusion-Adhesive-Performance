"""CSV and JSON output for simulation and sweep results.

Uses the standard library ``csv`` module rather than hand-formatted strings, so
quoting and line endings are correct on every platform. Histories are written in
long format (one row per sample) and profiles in tidy format (one row per
time/depth pair) because both load into any analysis tool without reshaping.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .simulate import SimulationResult
from .sweep import SweepResult

#: Columns written by :func:`write_history_csv`, in order.
HISTORY_COLUMNS = (
    "time_s",
    "time_days",
    "uptake_normalized",
    "uptake_pct",
    "interface_normalized",
    "interface_pct",
    "ari",
    "plasticization",
    "thermal",
    "hydrolysis",
    "damage",
    "glass_transition_c",
)


def _ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _rows(columns: Sequence[np.ndarray]) -> Iterable[tuple]:
    return zip(*(np.asarray(column, dtype=float).tolist() for column in columns))


def write_history_csv(path: str | Path, result: SimulationResult) -> Path:
    """Write the full time history of uptake and adhesion retention."""
    target = _ensure_parent(path)
    adhesion = result.adhesion
    columns = [
        result.time,
        result.time / 86400.0,
        result.uptake_normalized,
        result.transport.uptake_pct,
        result.interface_normalized,
        adhesion.interface_pct,
        adhesion.index,
        adhesion.plasticization,
        adhesion.thermal,
        adhesion.hydrolysis,
        adhesion.damage,
        adhesion.glass_transition_k - 273.15,
    ]
    with target.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HISTORY_COLUMNS)
        for row in _rows(columns):
            writer.writerow([f"{value:.10g}" for value in row])
    return target


def write_profile_csv(path: str | Path, result: SimulationResult) -> Path:
    """Write the through-thickness concentration profile in tidy format.

    Raises
    ------
    ValueError
        If the result carries no profile, which happens when it was produced
        with ``store_profile=False``.
    """
    profile = result.transport.profile_normalized
    depth = result.transport.depth
    if profile is None or depth is None:
        raise ValueError(
            "this result has no through-thickness profile; it was run with "
            "store_profile=False (as sweeps and sensitivity runs are)"
        )
    target = _ensure_parent(path)
    with target.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", "time_days", "depth_um", "c_over_csat", "moisture_pct"))
        saturation = result.saturation_pct
        for index, time_value in enumerate(result.time.tolist()):
            for position, depth_value in enumerate(depth.tolist()):
                value = float(profile[index, position])
                writer.writerow([
                    f"{time_value:.10g}",
                    f"{time_value / 86400.0:.10g}",
                    f"{depth_value * 1e6:.10g}",
                    f"{value:.10g}",
                    f"{value * saturation:.10g}",
                ])
    return target


def write_sweep_csv(path: str | Path, result: SweepResult) -> Path:
    """Write one row per sweep grid point, swept axes first."""
    target = _ensure_parent(path)
    if not result.records:  # pragma: no cover - run_sweep never yields none
        raise ValueError("sweep produced no records")
    fieldnames = list(result.records[0].keys())
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in result.records:
            writer.writerow({
                key: (f"{value:.10g}" if isinstance(value, float) else value)
                for key, value in record.items()
            })
    return target


def write_summary_json(path: str | Path, payload: dict) -> Path:
    """Write a summary dictionary as JSON, with infinities preserved as strings.

    ``json`` emits a bare ``Infinity`` token that strict parsers reject, so
    non-finite values become the string ``"not reached"`` --- which is what an
    infinite time-to-threshold means.
    """
    target = _ensure_parent(path)

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            number = float(value)
            return number if np.isfinite(number) else "not reached"
        return value

    target.write_text(json.dumps(clean(payload), indent=2) + "\n")
    return target


def read_uptake_csv(path: str | Path, time_column: str = "time_s",
                    uptake_column: str = "uptake_normalized") -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column gravimetric uptake curve.

    Accepts any CSV with a header containing the two named columns, so a file
    written by :func:`write_history_csv` round-trips.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"no such file: {target}")
    with target.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{target} has no header row")
        missing = {time_column, uptake_column} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{target} is missing column(s) {sorted(missing)}; "
                f"found {reader.fieldnames}"
            )
        times: list[float] = []
        values: list[float] = []
        for line, row in enumerate(reader, start=2):
            try:
                times.append(float(row[time_column]))
                values.append(float(row[uptake_column]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{target} line {line}: {exc}") from exc
    if len(times) < 2:
        raise ValueError(f"{target} needs at least two data rows")
    return np.array(times), np.array(values)
