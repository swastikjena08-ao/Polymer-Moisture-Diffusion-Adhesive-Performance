"""Polymer material model: temperature-dependent transport and wet Tg.

This module owns the two constitutive relations that turn the user's
three design variables into solver inputs: how diffusivity responds to
temperature, and how absorbed water depresses the glass transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from .sorption import PowerLawIsotherm
from .units import (
    TG_WATER_K,
    PhysicsError,
    arrhenius,
    require_finite,
    require_non_negative,
    require_positive,
    require_temperature,
)

TgModel = Literal["linear", "fox"]


@dataclass(frozen=True)
class Polymer:
    """A polymer film's moisture transport and thermal properties.

    Attributes
    ----------
    name:
        Label used in reports.
    diffusivity_ref:
        Moisture diffusivity in m^2/s at ``temperature_ref_k``. This is
        the third of the framework's three primary design variables, and
        is the quantity :func:`hygroadh.calibrate` fits to gravimetric data.
    activation_energy:
        Arrhenius activation energy for diffusion, J/mol. Typically
        40-70 kJ/mol for water in epoxies.
    temperature_ref_k:
        Reference temperature for ``diffusivity_ref``, K.
    isotherm:
        Equilibrium sorption isotherm.
    tg_dry_k:
        Dry glass transition temperature, K.
    tg_model:
        ``"linear"`` depresses Tg by ``tg_depression_per_pct`` K per weight
        percent of water; ``"fox"`` uses the Fox mixing rule against the Tg
        of amorphous water and needs no fitted slope.
    tg_depression_per_pct:
        Slope for the linear Tg model, K per wt%. Reported values for
        epoxies cluster around 10-20.
    tg_floor_k:
        Lower bound on the wet Tg for the linear model, preventing the
        unphysical extrapolation to arbitrarily low Tg at high uptake.
    conc_dependence:
        Dimensionless ``beta_c`` in ``D(C) = D_ref * exp(beta_c * C/C_sat)``.
        Zero gives constant diffusivity and the Fickian analytical
        solution; positive values give the accelerating uptake seen when
        water plasticizes the matrix it is diffusing through.
    """

    name: str = "polymer"
    diffusivity_ref: float = 5.0e-13
    activation_energy: float = 50.0e3
    temperature_ref_k: float = 298.15
    isotherm: PowerLawIsotherm = field(default_factory=lambda: PowerLawIsotherm(2.0))
    tg_dry_k: float = 393.15
    tg_model: TgModel = "linear"
    tg_depression_per_pct: float = 15.0
    tg_floor_k: float = 273.15
    conc_dependence: float = 0.0

    def __post_init__(self) -> None:
        require_positive(self.diffusivity_ref, "diffusivity_ref")
        require_finite(self.activation_energy, "activation_energy")
        require_temperature(self.temperature_ref_k, "temperature_ref_k")
        require_temperature(self.tg_dry_k, "tg_dry_k")
        require_temperature(self.tg_floor_k, "tg_floor_k")
        require_non_negative(self.tg_depression_per_pct, "tg_depression_per_pct")
        require_finite(self.conc_dependence, "conc_dependence")
        if self.tg_model not in ("linear", "fox"):
            raise PhysicsError(
                f"tg_model must be 'linear' or 'fox', got {self.tg_model!r}"
            )
        if self.tg_floor_k > self.tg_dry_k:
            raise PhysicsError(
                f"tg_floor_k ({self.tg_floor_k} K) cannot exceed tg_dry_k "
                f"({self.tg_dry_k} K)"
            )

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

    def glass_transition_k(self, moisture_pct):
        """Return the wet glass transition temperature in K.

        Accepts a scalar or an array of moisture contents in weight
        percent and returns the same shape.
        """
        w = np.asarray(moisture_pct, dtype=float)
        if np.any(w < 0.0):
            raise PhysicsError("moisture_pct must be non-negative")
        if self.tg_model == "linear":
            tg = self.tg_dry_k - self.tg_depression_per_pct * w
            tg = np.maximum(tg, self.tg_floor_k)
        else:
            mass_fraction = np.clip(w / 100.0, 0.0, 1.0 - 1e-12)
            inv = (1.0 - mass_fraction) / self.tg_dry_k + mass_fraction / TG_WATER_K
            tg = 1.0 / inv
        return float(tg) if np.ndim(tg) == 0 else tg

    def with_diffusivity(self, diffusivity_ref: float) -> "Polymer":
        """Return a copy with a different reference diffusivity.

        Used by parameter sweeps and by the calibration optimizer, both of
        which need to vary this one number while holding everything else
        fixed.
        """
        return replace(self, diffusivity_ref=diffusivity_ref)
