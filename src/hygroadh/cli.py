"""Command-line interface.

Subcommands
-----------
``serve``        launch the interactive dashboard on a local port
``run``          one case, writing history CSVs and a summary
``sweep``        a parametric sweep, writing a response table
``sensitivity``  elasticities and Morris screening
``info``         print the resolved parameters without running anything

Every subcommand takes a config file except ``serve``, where it is optional and
only seeds the initial slider positions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__, dashboard, io_data, sensitivity, sweep
from .config import Configuration, load_configuration
from .simulate import run
from .units import HygroadhError, days, to_celsius


def _load(path: str) -> Configuration:
    return load_configuration(path)


def _format_time(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "not reached"
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{days(seconds):.2f} d"


def _print_summary(summary: dict) -> None:
    order = (
        ("name", "case"),
        ("thickness_um", "thickness (um)"),
        ("temperature_c", "temperature (degC)"),
        ("relative_humidity", "relative humidity"),
        ("solver", "solver"),
        ("diffusivity_m2_s", "D at temperature (m2/s)"),
        ("saturation_pct", "equilibrium uptake (wt%)"),
        ("final_uptake_normalized", "final M/Minf"),
        ("final_interface_normalized", "final far-face C/Csat"),
        ("final_interface_pct", "final far-face moisture (wt%)"),
    )
    width = max(len(label) for _, label in order)
    for key, label in order:
        value = summary[key]
        rendered = f"{value:.6g}" if isinstance(value, float) else str(value)
        print(f"  {label:<{width}} : {rendered}")
    basis = summary["threshold_basis"]
    level = summary["threshold_value"]
    stated = f"{level:g} wt%" if basis == "wt_pct" else f"{level:g} of equilibrium"
    print(f"  {'moisture threshold':<{width}} : {stated} "
          f"({summary['threshold_pct']:.4g} wt% at the far face)")
    for label, seconds in (
        ("time to half uptake", summary["time_to_half_uptake_s"]),
        ("time to threshold", summary["time_to_threshold_s"]),
    ):
        print(f"  {label:<{width}} : {_format_time(seconds)}")
    if not summary["threshold_reachable"]:
        print(f"  {'':<{width}}   (the threshold exceeds what the film can hold "
              "at this humidity)")


STANDING_NOTE = (
    "Note: this model predicts moisture transport, not adhesive strength.\n"
    "The reported time is when moisture reaches the chosen far-face threshold.\n"
    "It is a proxy for possible performance change; diffusivity and activation\n"
    "energy are model parameters, not universal material constants."
)


def cmd_run(args: argparse.Namespace) -> int:
    configuration = _load(args.config)
    result = run(configuration.case)
    print(f"case: {configuration.case.name}")
    _print_summary(result.summary())
    if args.outdir:
        outdir = Path(args.outdir)
        history = io_data.write_history_csv(outdir / "history.csv", result)
        summary = io_data.write_summary_json(outdir / "summary.json", result.summary())
        written = [history, summary]
        try:
            written.append(io_data.write_profile_csv(outdir / "profile.csv", result))
        except ValueError:  # pragma: no cover - profiles are on by default
            pass
        print("\nwrote:")
        for path in written:
            print(f"  {path}")
    print()
    print(STANDING_NOTE)
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    configuration = _load(args.config)
    axes = configuration.sweep
    if axes.is_empty:
        print(
            "no sweep section in the config. Add one, for example:\n"
            "\n"
            "sweep:\n"
            "  thickness_um: [50, 100, 200, 400]\n"
            "  temperature_c: [25, 40, 60, 80]\n",
            file=sys.stderr,
        )
        return 2
    total_seen = {"n": 0}

    def progress(done: int, total: int) -> None:
        total_seen["n"] = total
        if args.quiet:
            return
        print(f"\r  {done}/{total} points", end="", file=sys.stderr, flush=True)

    result = sweep.run_sweep(configuration.case, axes, progress=progress)
    if not args.quiet:
        print(file=sys.stderr)

    print(f"swept axes: {', '.join(result.axis_names)}  shape {result.shape}")
    finite = result.finite_fraction("time_to_threshold_s")
    print(f"moisture threshold reached at {finite:.0%} of grid points")
    if finite < 1.0:
        print("  (the rest do not reach it in the simulated window; "
              "reported as 'not reached')")

    surface = result.surface("time_to_threshold_s")
    reachable = surface[np.isfinite(surface)]
    if reachable.size:
        print(f"  fastest: {_format_time(float(reachable.min()))}")
        print(f"  slowest: {_format_time(float(reachable.max()))}")

    if args.outdir:
        path = io_data.write_sweep_csv(Path(args.outdir) / "sweep.csv", result)
        print(f"\nwrote:\n  {path}")
    print()
    print(STANDING_NOTE)
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    configuration = _load(args.config)
    case = configuration.case
    elasticity = sensitivity.local_elasticities(case, response=args.response)
    print(f"response: {elasticity.response}  (base value {elasticity.base_value:.6g})")
    print("local elasticities, d ln y / d ln x:")
    width = max(len(name) for name in elasticity.elasticities)
    for name, value in elasticity.ranking:
        print(f"  {name:<{width}} : {value:+.4f}")
    print(f"most influential: {elasticity.most_influential}")

    if args.morris:
        screening = sensitivity.morris_screening(
            case, response=args.response, n_trajectories=args.trajectories,
            seed=args.seed,
        )
        print(
            f"\nMorris screening, {screening.n_trajectories} trajectories, "
            f"{screening.n_evaluations} evaluations:"
        )
        print(f"  {'factor':<{width}}   {'mu*':>12}  {'sigma':>12}  interacts")
        for name, mu_star in screening.ranking:
            print(
                f"  {name:<{width}} : {mu_star:12.4g}  "
                f"{screening.sigma[name]:12.4g}  "
                f"{'yes' if screening.interacts(name) else 'no'}"
            )
        print(
            "  a large sigma relative to mu* means the factor's effect depends on\n"
            "  where you are in the design space, so one elasticity will not\n"
            "  describe it everywhere"
        )
    print()
    print(STANDING_NOTE)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    configuration = _load(args.config)
    case = configuration.case
    condition = case.condition
    polymer = case.polymer
    temperature = condition.temperature_k
    print(f"case: {case.name}")
    print(f"  film thickness        : {case.thickness * 1e6:.3g} um ({condition.exposure})")
    print(f"  temperature           : {to_celsius(temperature):.3g} degC")
    print(f"  relative humidity     : {condition.relative_humidity:.3g}")
    print(f"  duration              : {_format_time(condition.duration)}")
    print(f"  solver                : {case.resolved_solver}")
    print(f"  D at temperature      : {polymer.diffusivity(temperature):.4g} m2/s")
    print(f"  D at 25 degC          : {polymer.diffusivity(298.15):.4g} m2/s")
    print(f"  saturation uptake     : "
          f"{polymer.saturation_pct(temperature, condition.relative_humidity):.4g} wt%")
    saturation = polymer.saturation_pct(temperature, condition.relative_humidity)
    print(f"  activation energy     : {polymer.activation_energy / 1e3:.3g} kJ/mol")
    print(f"  moisture threshold    : {case.criterion.describe(saturation)}")
    reason = case.criterion.unreachable_reason(saturation)
    if reason:
        print(f"  WARNING               : {reason}")
    if not configuration.sweep.is_empty:
        axes = configuration.sweep
        names = [
            name for name, values in (
                ("thickness", axes.thickness),
                ("temperature_k", axes.temperature_k),
                ("diffusivity", axes.diffusivity),
            ) if values is not None
        ]
        print(f"  sweep axes            : {', '.join(names)}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    defaults = None
    if args.config:
        configuration = _load(args.config)
        defaults = dashboard.params_from_case(configuration.case)
        dropped = dashboard.unsupported_dashboard_features(configuration.case)
        if dropped:
            print(
                "note: the dashboard has no control for "
                + ", ".join(dropped)
                + ";\n      those settings are not applied. Use `hygroadh run` for them.",
                file=sys.stderr,
            )
    dashboard.serve(
        host=args.host, port=args.port,
        open_browser=args.open_browser and not args.no_browser,
        defaults=defaults,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hygroadh",
        description=(
            "A computational study of moisture diffusion in polymer adhesive "
            "films: how thickness, temperature, and diffusivity set the time "
            "for moisture to reach the far face."
        ),
    )
    parser.add_argument("--version", action="version", version=f"hygroadh {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="launch the interactive dashboard")
    serve.add_argument("config", nargs="?", help="optional config to seed the sliders")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765,
                       help="port to bind; walks forward if taken (default 8765)")
    serve.add_argument("--open", dest="open_browser", action="store_true",
                       help="also try to open a browser (off by default; under "
                            "WSL this uses wslview or explorer.exe)")
    # Accepted for compatibility: not opening a browser is now the default, so
    # this is a no-op rather than an error for anyone with it in a script.
    serve.add_argument("--no-browser", action="store_true",
                       help=argparse.SUPPRESS)
    serve.set_defaults(func=cmd_serve)

    run_parser = subparsers.add_parser("run", help="run one case")
    run_parser.add_argument("config")
    run_parser.add_argument("--outdir", help="directory for CSV and JSON output")
    run_parser.set_defaults(func=cmd_run)

    sweep_parser = subparsers.add_parser("sweep", help="run a parametric sweep")
    sweep_parser.add_argument("config")
    sweep_parser.add_argument("--outdir", help="directory for the sweep table")
    sweep_parser.add_argument("--quiet", action="store_true", help="suppress progress")
    sweep_parser.set_defaults(func=cmd_sweep)

    sens_parser = subparsers.add_parser("sensitivity", help="rank the design variables")
    sens_parser.add_argument("config")
    sens_parser.add_argument("--response", default="time_to_threshold",
                             choices=sorted(sensitivity.RESPONSES))
    sens_parser.add_argument("--morris", action="store_true",
                             help="also run global Morris screening")
    sens_parser.add_argument("--trajectories", type=int, default=10)
    sens_parser.add_argument("--seed", type=int, default=0)
    sens_parser.set_defaults(func=cmd_sensitivity)

    info_parser = subparsers.add_parser("info", help="show resolved parameters")
    info_parser.add_argument("config")
    info_parser.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except HygroadhError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
