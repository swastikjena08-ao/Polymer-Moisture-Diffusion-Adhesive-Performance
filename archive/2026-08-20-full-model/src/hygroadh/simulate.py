"""Orchestration: turn a case definition into transport plus adhesion results.

This module is the single seam where the transport half and the adhesion half
meet, and the only place that knows about both. Two responsibilities live here
deliberately rather than further down:

* **Resolving temperature.** ``D(T)`` and ``M_sat(T, RH)`` are evaluated here,
  so the solvers receive plain numbers and stay unaware of Arrhenius laws.
* **Mapping humidity to a surface condition.** A relative-humidity history is
  converted into the normalized surface concentration the solver wants by
  passing it through the isotherm. Doing this here keeps
  :mod:`hygroadh.diffusion.fd` isotherm-agnostic --- it solves a boundary-value
  problem in normalized concentration and has no opinion about sorption.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from .adhesion import AdhesionModel, AdhesionResult
from .diffusion import analytical, fd
from .diffusion.base import DiffusionResult, Exposure
from .materials import Polymer
from .units import (
    PhysicsError,
    require_fraction,
    require_positive,
    require_temperature,
)

SolverName = Literal["auto", "analytical", "fd"]
TimeSpacing = Literal["auto", "log", "linear"]


@dataclass(frozen=True)
class ExposureCondition:
    """The environment a film is held in.

    Attributes
    ----------
    temperature_k:
        Service temperature, K. Drives ``D``, ``M_sat``, the hydrolysis rate,
        and the gap to the wet Tg.
    relative_humidity:
        Nominal humidity as a fraction. This sets the ``M_inf`` that uptake is
        normalized against, so ``M/M_inf -> 1`` means equilibrium *with this
        humidity*, not with liquid water.
    duration:
        Length of the exposure in seconds.
    humidity_schedule:
        Optional relative-humidity history for cycling or drying legs. When
        given, the solver switches to finite volumes automatically.
    exposure:
        ``"one_sided"`` for a film on an impermeable substrate,
        ``"two_sided"`` for a free film.
    surface_transfer:
        Optional surface mass-transfer coefficient in m/s. ``None`` means the
        face equilibrates instantly with the environment.
    n_times:
        Number of output samples.
    time_spacing:
        ``"log"`` resolves the early square-root region where the curve moves
        fastest; ``"linear"`` suits cyclic exposure. ``"auto"`` picks linear
        when a humidity schedule is present and log otherwise.
    """

    temperature_k: float
    duration: float
    relative_humidity: float = 1.0
    humidity_schedule: fd.Schedule | None = None
    exposure: Exposure = "one_sided"
    surface_transfer: float | None = None
    n_times: int = 240
    time_spacing: TimeSpacing = "auto"

    def __post_init__(self) -> None:
        require_temperature(self.temperature_k)
        require_positive(self.duration, "duration")
        require_fraction(self.relative_humidity, "relative_humidity")
        if self.relative_humidity == 0.0:
            raise PhysicsError(
                "relative_humidity must be greater than zero: it sets the "
                "saturation that uptake is normalized against"
            )
        if self.n_times < 2:
            raise PhysicsError("n_times must be at least 2")
        if self.exposure not in ("one_sided", "two_sided"):
            raise PhysicsError(
                f"exposure must be 'one_sided' or 'two_sided', got {self.exposure!r}"
            )
        if self.time_spacing not in ("auto", "log", "linear"):
            raise PhysicsError(
                f"time_spacing must be 'auto', 'log' or 'linear', got "
                f"{self.time_spacing!r}"
            )
        if self.surface_transfer is not None:
            require_positive(self.surface_transfer, "surface_transfer")

    @property
    def resolved_spacing(self) -> str:
        """The spacing actually used once ``"auto"`` is resolved."""
        if self.time_spacing != "auto":
            return self.time_spacing
        return "linear" if self.humidity_schedule is not None else "log"

    def time_grid(self) -> np.ndarray:
        """Build the output time grid, always starting at exactly zero."""
        if self.resolved_spacing == "linear":
            return np.linspace(0.0, self.duration, self.n_times)
        early = self.duration * 1e-5
        return np.concatenate(
            [[0.0], np.geomspace(early, self.duration, self.n_times - 1)]
        )


@dataclass(frozen=True)
class Case:
    """A complete, self-contained specification of one simulation."""

    thickness: float
    condition: ExposureCondition
    polymer: Polymer = field(default_factory=Polymer)
    adhesion: AdhesionModel = field(default_factory=AdhesionModel)
    solver: SolverName = "auto"
    n_cells: int = 80
    ari_threshold: float = 0.8
    name: str = "case"
    store_profile: bool = True
    """Whether to retain the through-thickness profile.

    Sweeps and sensitivity runs only ever read scalars off the result, and
    evaluating the profile series at every output time is the single largest
    cost in a run. Disabling it makes a 49-point design map several times
    cheaper at identical accuracy in the scalars.
    """

    def __post_init__(self) -> None:
        require_positive(self.thickness, "thickness")
        if self.solver not in ("auto", "analytical", "fd"):
            raise PhysicsError(
                f"solver must be 'auto', 'analytical' or 'fd', got {self.solver!r}"
            )
        if self.n_cells < 2:
            raise PhysicsError("n_cells must be at least 2")
        require_fraction(self.ari_threshold, "ari_threshold")

    @property
    def requires_finite_volume(self) -> bool:
        """Whether this case has a feature the analytical solution cannot express."""
        return (
            self.condition.humidity_schedule is not None
            or self.condition.surface_transfer is not None
            or self.polymer.conc_dependence != 0.0
        )

    @property
    def resolved_solver(self) -> str:
        """The solver actually used once ``"auto"`` is resolved.

        Raises
        ------
        PhysicsError
            If ``"analytical"`` was requested for a case that needs finite
            volumes. Silently substituting a different physics would be worse
            than refusing.
        """
        if self.solver == "auto":
            return "fd" if self.requires_finite_volume else "analytical"
        if self.solver == "analytical" and self.requires_finite_volume:
            raise PhysicsError(
                "the analytical solution assumes constant diffusivity, a constant "
                "surface condition, and no surface resistance; this case uses at "
                "least one of a humidity schedule, a surface transfer "
                "coefficient, or a concentration-dependent diffusivity. Use "
                "solver='fd' or 'auto'."
            )
        return self.solver

    def without_profile(self, n_times: int | None = None) -> "Case":
        """Return a copy tuned for extracting scalars only.

        Drops the through-thickness profile and optionally coarsens the output
        grid. Used by sweeps and sensitivity analysis, which read only summary
        numbers.
        """
        condition = self.condition
        if n_times is not None:
            condition = replace(condition, n_times=int(n_times))
        return replace(self, condition=condition, store_profile=False)

    def with_thickness(self, thickness: float) -> "Case":
        """Return a copy at a different film thickness."""
        return replace(self, thickness=thickness)

    def with_temperature(self, temperature_k: float) -> "Case":
        """Return a copy at a different service temperature."""
        return replace(
            self, condition=replace(self.condition, temperature_k=temperature_k)
        )

    def with_diffusivity(self, diffusivity_ref: float) -> "Case":
        """Return a copy with a different reference diffusivity."""
        return replace(self, polymer=self.polymer.with_diffusivity(diffusivity_ref))


@dataclass
class SimulationResult:
    """Transport and adhesion histories for one case, plus derived scalars."""

    case: Case
    transport: DiffusionResult
    adhesion: AdhesionResult
    diffusivity: float
    saturation_pct: float

    @property
    def time(self) -> np.ndarray:
        return self.transport.time

    @property
    def uptake_normalized(self) -> np.ndarray:
        """``M(t)/M_inf`` --- the gravimetric observable."""
        return self.transport.uptake_normalized

    @property
    def interface_normalized(self) -> np.ndarray:
        """``C_interface/C_sat`` --- what actually drives adhesion."""
        return self.transport.interface_normalized

    @property
    def index(self) -> np.ndarray:
        """The Adhesion Retention Index history."""
        return self.adhesion.index

    @property
    def time_to_half_uptake(self) -> float:
        """Time for the film to reach half of its equilibrium uptake."""
        return self.transport.time_to_uptake(0.5)

    @property
    def time_to_ari_threshold(self) -> float:
        """Time for ARI to fall to ``case.ari_threshold``, or ``inf``.

        The headline engineering response: it collapses a whole simulation into
        one number that can be mapped over a design space.
        """
        return self.adhesion.time_to_index(self.case.ari_threshold)

    @property
    def final_index(self) -> float:
        return self.adhesion.final_index

    @property
    def dominant_mechanism(self) -> str:
        return self.adhesion.dominant_mechanism

    def summary(self) -> dict[str, object]:
        """Flat dictionary of the scalars worth tabulating or reporting."""
        return {
            "name": self.case.name,
            "thickness_m": self.case.thickness,
            "thickness_um": self.case.thickness * 1e6,
            "temperature_k": self.case.condition.temperature_k,
            "temperature_c": self.case.condition.temperature_k - 273.15,
            "relative_humidity": self.case.condition.relative_humidity,
            "diffusivity_m2_s": self.diffusivity,
            "saturation_pct": self.saturation_pct,
            "solver": self.transport.solver,
            "duration_s": self.case.condition.duration,
            "final_uptake_normalized": float(self.uptake_normalized[-1]),
            "final_interface_normalized": float(self.interface_normalized[-1]),
            "time_to_half_uptake_s": self.time_to_half_uptake,
            "final_ari": self.final_index,
            "ari_threshold": self.case.ari_threshold,
            "time_to_ari_threshold_s": self.time_to_ari_threshold,
            "dominant_mechanism": self.dominant_mechanism,
        }


def surface_condition(polymer: Polymer, condition: ExposureCondition,
                      saturation_pct: float):
    """Build the normalized surface condition the solver consumes.

    Returns ``1.0`` for steady exposure at the nominal humidity. With a
    humidity schedule, returns a callable mapping time to
    ``M_sat(T, RH(t)) / M_sat(T, RH_nominal)`` --- the isotherm is applied here
    so that a non-linear isotherm is honoured, rather than assuming surface
    concentration is proportional to humidity.
    """
    if condition.humidity_schedule is None:
        return 1.0
    schedule = condition.humidity_schedule
    temperature = condition.temperature_k

    def normalized(time: float) -> float:
        humidity = float(np.clip(schedule(time), 0.0, 1.0))
        return polymer.saturation_pct(temperature, humidity) / saturation_pct

    # Expose the peak so the solver can size its time step.
    peak_humidity = float(np.clip(schedule.maximum, 0.0, 1.0))
    normalized.maximum = (
        polymer.saturation_pct(temperature, peak_humidity) / saturation_pct
    )
    return normalized


def run(case: Case) -> SimulationResult:
    """Run one case: resolve temperature, solve transport, then evaluate ARI."""
    condition = case.condition
    polymer = case.polymer
    temperature = condition.temperature_k

    diffusivity = polymer.diffusivity(temperature)
    saturation_pct = polymer.saturation_pct(temperature, condition.relative_humidity)
    if saturation_pct <= 0.0:
        raise PhysicsError(
            "the isotherm gives zero saturation uptake for this temperature and "
            "humidity, so normalized uptake is undefined"
        )

    time = condition.time_grid()
    solver = case.resolved_solver

    if solver == "analytical":
        transport = analytical.solve(
            time,
            case.thickness,
            diffusivity,
            saturation_pct,
            temperature,
            exposure=condition.exposure,
            n_depth=case.n_cells if case.store_profile else None,
        )
    else:
        transport = fd.solve(
            time,
            case.thickness,
            diffusivity,
            saturation_pct,
            temperature,
            exposure=condition.exposure,
            n_cells=case.n_cells,
            surface=surface_condition(polymer, condition, saturation_pct),
            conc_dependence=polymer.conc_dependence,
            surface_transfer=condition.surface_transfer,
            store_profile=case.store_profile,
        )

    adhesion = case.adhesion.evaluate(
        transport.time,
        transport.interface_normalized,
        temperature,
        saturation_pct,
        polymer.glass_transition_k,
    )

    return SimulationResult(
        case=case,
        transport=transport,
        adhesion=adhesion,
        diffusivity=diffusivity,
        saturation_pct=saturation_pct,
    )
