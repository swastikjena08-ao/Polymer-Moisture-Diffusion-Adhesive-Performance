"""Polymer material model: temperature-dependent moisture transport.

This module owns the constitutive relation that turns two of the framework's
three design variables --- temperature and diffusivity --- into a single number
the solvers can use.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .sorption import HenryIsotherm
from .units import arrhenius, require_finite, require_positive, require_temperature


@dataclass(frozen=True)
class Polymer:
    """A polymer film's moisture transport properties.

    Attributes
    ----------
    name:
        Label used in reports.
    diffusivity_ref:
        Moisture diffusivity in m^2/s at ``temperature_ref_k``, and the third of
        the framework's three primary design variables. Constant --- independent
        of moisture content --- so the exact Fickian series applies.
    activation_energy:
        Arrhenius activation energy for diffusion, J/mol. Typically 40-70 kJ/mol
        for water in epoxies, which makes diffusion roughly ten to thirty times
        faster over a 50 K rise.
    temperature_ref_k:
        Reference temperature for ``diffusivity_ref``, K.
    isotherm:
        Equilibrium sorption isotherm.
    """

    name: str = "polymer"
    diffusivity_ref: float = 1.0e-13
    activation_energy: float = 50.0e3
    temperature_ref_k: float = 298.15
    isotherm: HenryIsotherm = field(default_factory=lambda: HenryIsotherm(2.0))

    def __post_init__(self) -> None:
        require_positive(self.diffusivity_ref, "diffusivity_ref")
        require_finite(self.activation_energy, "activation_energy")
        require_temperature(self.temperature_ref_k, "temperature_ref_k")

    def diffusivity(self, temperature_k: float) -> float:
        """Return moisture diffusivity in m^2/s at the given temperature."""
        return arrhenius(
            self.diffusivity_ref,
            self.activation_energy,
            temperature_k,
            self.temperature_ref_k,
        )

    def saturation_pct(self, temperature_k: float, relative_humidity: float) -> float:
        """Return equilibrium moisture content in wt% (delegates to the isotherm)."""
        return self.isotherm.saturation_pct(temperature_k, relative_humidity)

    def with_diffusivity(self, diffusivity_ref: float) -> "Polymer":
        """Return a copy with a different reference diffusivity.

        Used by parameter sweeps and sensitivity analysis, which need to vary
        this one number while holding everything else fixed.
        """
        return replace(self, diffusivity_ref=diffusivity_ref)
