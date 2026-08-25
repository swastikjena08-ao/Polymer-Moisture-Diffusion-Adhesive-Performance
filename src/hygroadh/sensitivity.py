"""Sensitivity analysis over the three design variables.

Two complementary views, both numpy-only:

**Local elasticities** answer "if I change this by one percent, how much does
the answer move, in percent?" They are dimensionless, so thickness in metres,
temperature in Kelvin, and diffusivity in m^2/s can be ranked against each
other directly --- which a raw partial derivative cannot do.

**Morris elementary effects** screen globally. Elasticities are evaluated at a
single point and say nothing about whether the ranking holds elsewhere; Morris
walks the design space one factor at a time and reports both the average
magnitude of each factor's effect (``mu_star``) and how much that effect varies
with where you are (``sigma``). A large ``sigma`` is the signal that a factor
interacts with the others, so its elasticity should not be trusted far from the
point where it was measured.

Sobol variance decomposition is deliberately not implemented: it needs a
quasi-random sequence and many more model evaluations, and buys little over
Morris for three factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .simulate import Case, run
from .units import PhysicsError, require_positive

#: Response name -> extractor. Every one must be positive for elasticities to
#: be well defined in log space.
RESPONSES: dict[str, Callable[[object], float]] = {
    "time_to_threshold": lambda result: float(result.time_to_threshold),
    "time_to_half_uptake": lambda result: float(result.time_to_half_uptake),
    "final_uptake_normalized": lambda result: float(result.uptake_normalized[-1]),
    "final_interface_normalized": lambda result: float(result.interface_normalized[-1]),
    "final_interface_pct": lambda result: float(result.interface_pct[-1]),
}

#: The three design variables, in the order they are reported.
FACTORS = ("thickness", "temperature_k", "diffusivity")

#: Factors whose plausible ranges span orders of magnitude, and which are
#: therefore perturbed and interpolated geometrically.
LOG_FACTORS = ("thickness", "diffusivity")


def _factor_value(case: Case, factor: str) -> float:
    if factor == "thickness":
        return float(case.thickness)
    if factor == "temperature_k":
        return float(case.condition.temperature_k)
    if factor == "diffusivity":
        return float(case.polymer.diffusivity_ref)
    raise PhysicsError(f"unknown factor {factor!r}; available: {sorted(FACTORS)}")


def _with_factor(case: Case, factor: str, value: float) -> Case:
    if factor == "thickness":
        return case.with_thickness(value)
    if factor == "temperature_k":
        return case.with_temperature(value)
    if factor == "diffusivity":
        return case.with_diffusivity(value)
    raise PhysicsError(f"unknown factor {factor!r}; available: {sorted(FACTORS)}")


def _evaluate(case: Case, response: str) -> float:
    if response not in RESPONSES:
        raise PhysicsError(
            f"unknown response {response!r}; available: {sorted(RESPONSES)}"
        )
    return RESPONSES[response](run(case.without_profile()))


def _require_usable(value: float, response: str, context: str) -> float:
    if not np.isfinite(value):
        raise PhysicsError(
            f"response {response!r} is {value} {context}. For "
            "'time_to_threshold' this means the threshold is never reached in "
            "the simulated window: lengthen the simulation, lower the threshold, "
            "or raise the humidity if the threshold is set in wt%."
        )
    if value <= 0.0:
        raise PhysicsError(
            f"response {response!r} is {value} {context}; elasticities are "
            "computed in log space and need a strictly positive response"
        )
    return value


@dataclass(frozen=True)
class ElasticityResult:
    """Local log-log sensitivities, dimensionless and directly comparable."""

    response: str
    base_value: float
    elasticities: dict[str, float]
    relative_step: float

    @property
    def ranking(self) -> list[tuple[str, float]]:
        """Factors ordered by the magnitude of their influence, strongest first."""
        return sorted(
            self.elasticities.items(), key=lambda item: abs(item[1]), reverse=True
        )

    @property
    def most_influential(self) -> str:
        return self.ranking[0][0]


def local_elasticities(case: Case, response: str = "time_to_threshold",
                       factors: tuple[str, ...] = FACTORS,
                       relative_step: float = 0.02) -> ElasticityResult:
    """Central-difference elasticities ``d ln y / d ln x`` at the base case.

    A value of ``-2`` means a one percent increase in that factor reduces the
    response by about two percent. Temperature is perturbed multiplicatively in
    Kelvin like the others, so the three numbers share one scale; note that this
    makes the temperature elasticity large in magnitude, because one percent of
    an absolute temperature is a few Kelvin.

    ``time_to_threshold`` is the default because of a degeneracy worth knowing
    about: once a simulation is long enough to saturate the film, every case ends
    at ``M/M_inf = 1`` regardless of thickness, temperature, or diffusivity ---
    those three set how *fast* moisture arrives, not how much arrives eventually.
    Any end-of-run response is therefore blind to all of them. The time to cross
    a threshold is the response that carries the study's question.
    """
    step = require_positive(relative_step, "relative_step")
    if step >= 0.5:
        raise PhysicsError("relative_step must be well below 0.5")

    base = _require_usable(_evaluate(case, response), response, "at the base case")
    values: dict[str, float] = {}
    for factor in factors:
        centre = _factor_value(case, factor)
        low_case = _with_factor(case, factor, centre * (1.0 - step))
        high_case = _with_factor(case, factor, centre * (1.0 + step))
        low = _require_usable(
            _evaluate(low_case, response), response, f"at {factor} decreased by {step:.1%}"
        )
        high = _require_usable(
            _evaluate(high_case, response), response,
            f"at {factor} increased by {step:.1%}",
        )
        values[factor] = float(
            (np.log(high) - np.log(low)) / (np.log(1.0 + step) - np.log(1.0 - step))
        )
    return ElasticityResult(
        response=response, base_value=base, elasticities=values, relative_step=step
    )


def default_ranges(case: Case, factor_span: float = 3.0,
                   temperature_span: float = 25.0) -> dict[str, tuple[float, float]]:
    """Plausible screening ranges around a base case.

    Thickness and diffusivity get a geometric span, since their plausible values
    range over orders of magnitude; temperature gets an additive one in Kelvin.
    """
    span = require_positive(factor_span, "factor_span")
    require_positive(temperature_span, "temperature_span")
    thickness = _factor_value(case, "thickness")
    diffusivity = _factor_value(case, "diffusivity")
    temperature = _factor_value(case, "temperature_k")
    return {
        "thickness": (thickness / span, thickness * span),
        "temperature_k": (
            max(temperature - temperature_span, 200.0),
            temperature + temperature_span,
        ),
        "diffusivity": (diffusivity / span, diffusivity * span),
    }


@dataclass(frozen=True)
class MorrisResult:
    """Global screening statistics from Morris elementary effects."""

    response: str
    factors: tuple[str, ...]
    mu_star: dict[str, float]
    sigma: dict[str, float]
    mean: dict[str, float]
    n_trajectories: int
    n_levels: int
    n_evaluations: int

    @property
    def ranking(self) -> list[tuple[str, float]]:
        """Factors ordered by mean absolute effect, strongest first."""
        return sorted(self.mu_star.items(), key=lambda item: item[1], reverse=True)

    @property
    def most_influential(self) -> str:
        return self.ranking[0][0]

    def interacts(self, factor: str, tolerance: float = 0.5) -> bool:
        """Whether a factor's effect varies strongly with location in the space.

        True when ``sigma`` is a large fraction of ``mu_star``, which indicates
        the factor interacts with the others or acts non-linearly, so a single
        local elasticity will not describe it everywhere.
        """
        magnitude = self.mu_star[factor]
        if magnitude == 0.0:
            return False
        return self.sigma[factor] / magnitude > tolerance


def _to_value(factor: str, unit: float, low: float, high: float) -> float:
    """Map a normalized coordinate in [0, 1] onto a factor's range."""
    if factor in LOG_FACTORS:
        return float(low * (high / low) ** unit)
    return float(low + unit * (high - low))


def morris_screening(case: Case, ranges: dict[str, tuple[float, float]] | None = None,
                     response: str = "time_to_threshold", n_trajectories: int = 10,
                     n_levels: int = 4, seed: int = 0) -> MorrisResult:
    """Screen the design space with Morris elementary effects.

    Each trajectory starts at a random point on a ``n_levels`` grid in
    normalized factor space and changes one factor at a time by ``delta``,
    recording the resulting change in the response. Costs
    ``n_trajectories * (k + 1)`` model evaluations for ``k`` factors.

    Effects are reported in normalized units --- per unit of the factor's full
    range --- so they are comparable across factors regardless of physical
    units, exactly as the elasticities are.
    """
    if ranges is None:
        ranges = default_ranges(case)
    factors = tuple(name for name in FACTORS if name in ranges)
    if not factors:
        raise PhysicsError(f"ranges must cover at least one of {sorted(FACTORS)}")
    for name in factors:
        low, high = ranges[name]
        if not (np.isfinite(low) and np.isfinite(high)) or low <= 0.0 or high <= low:
            raise PhysicsError(
                f"range for {name!r} must be a finite increasing positive pair, "
                f"got {(low, high)!r}"
            )
    if n_trajectories < 1:
        raise PhysicsError("n_trajectories must be at least 1")
    if n_levels < 2:
        raise PhysicsError("n_levels must be at least 2")

    k = len(factors)
    delta = n_levels / (2.0 * (n_levels - 1))
    grid = np.linspace(0.0, 1.0, n_levels)
    startable = grid[grid <= 1.0 - delta + 1e-12]
    if startable.size == 0:  # pragma: no cover - excluded by n_levels >= 2
        raise PhysicsError("n_levels is too small to take a step of delta")

    rng = np.random.default_rng(seed)
    effects: dict[str, list[float]] = {name: [] for name in factors}
    evaluations = 0

    for _ in range(int(n_trajectories)):
        unit = rng.choice(startable, size=k)
        order = rng.permutation(k)
        current_case = case
        for position, name in enumerate(factors):
            current_case = _with_factor(
                current_case, name, _to_value(name, float(unit[position]), *ranges[name])
            )
        current = _require_usable(
            _evaluate(current_case, response), response, "at a Morris trajectory start"
        )
        evaluations += 1

        for factor_index in order:
            name = factors[factor_index]
            moved = unit.copy()
            # Step away from the boundary so the point stays inside the range.
            step = delta if moved[factor_index] + delta <= 1.0 + 1e-12 else -delta
            moved[factor_index] = float(np.clip(moved[factor_index] + step, 0.0, 1.0))
            trial_case = case
            for position, other in enumerate(factors):
                trial_case = _with_factor(
                    trial_case, other,
                    _to_value(other, float(moved[position]), *ranges[other]),
                )
            trial = _require_usable(
                _evaluate(trial_case, response), response,
                f"at a Morris step in {name!r}",
            )
            evaluations += 1
            effects[name].append((trial - current) / step)
            unit = moved
            current = trial

    return MorrisResult(
        response=response,
        factors=factors,
        mu_star={name: float(np.mean(np.abs(values))) for name, values in effects.items()},
        sigma={name: float(np.std(values)) for name, values in effects.items()},
        mean={name: float(np.mean(values)) for name, values in effects.items()},
        n_trajectories=int(n_trajectories),
        n_levels=int(n_levels),
        n_evaluations=evaluations,
    )
