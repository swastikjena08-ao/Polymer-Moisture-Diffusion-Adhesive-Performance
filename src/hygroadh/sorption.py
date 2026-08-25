"""Equilibrium moisture sorption using Henry's law.

The isotherm supplies the film's saturation moisture content and converts the
dimensionless transport result to weight percent. It implements the linear
humidity case; curved polymer isotherms are outside this model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .units import (
    GAS_CONSTANT,
    require_finite,
    require_fraction,
    require_positive,
    require_temperature,
)


@dataclass(frozen=True)
class HenryIsotherm:
    """Saturation moisture content proportional to relative humidity.

    ``M_sat(T, RH) = m_ref_pct * RH * exp(-(dH_s/R)(1/T - 1/T_ref))``

    Attributes
    ----------
    m_ref_pct:
        Saturation moisture content in weight percent at ``RH = 1`` and the
        reference temperature. This is the single number that sets how wet the
        film can get.
    enthalpy_sorption:
        ``dH_s`` in J/mol, a van 't Hoff term for how saturation uptake shifts
        with temperature. **Defaults to zero**, which makes saturation
        temperature-independent --- so by default temperature changes only how
        *fast* the film wets, not how wet it ends up. Set it positive to make a
        hotter film also hold more water, as epoxies weakly do.
    temperature_ref_k:
        Reference temperature in K for both quantities above.
    """

    m_ref_pct: float
    enthalpy_sorption: float = 0.0
    temperature_ref_k: float = 298.15

    def __post_init__(self) -> None:
        require_positive(self.m_ref_pct, "m_ref_pct")
        require_finite(self.enthalpy_sorption, "enthalpy_sorption")
        require_temperature(self.temperature_ref_k, "temperature_ref_k")

    def _temperature_factor(self, temperature_k: float) -> float:
        if self.enthalpy_sorption == 0.0:
            require_temperature(temperature_k)
            return 1.0
        t = require_temperature(temperature_k)
        exponent = -(self.enthalpy_sorption / GAS_CONSTANT) * (
            1.0 / t - 1.0 / self.temperature_ref_k
        )
        return float(np.exp(exponent))

    def saturation_pct(self, temperature_k: float, relative_humidity: float) -> float:
        """Return equilibrium moisture content in weight percent."""
        rh = require_fraction(relative_humidity, "relative_humidity")
        if rh == 0.0:
            require_temperature(temperature_k)
            return 0.0
        return self.m_ref_pct * rh * self._temperature_factor(temperature_k)

    def humidity_for(self, saturation_pct: float, temperature_k: float) -> float:
        """Invert the isotherm: the humidity giving a target saturation uptake.

        Raises
        ------
        ValueError
            If the requested uptake is unreachable at any humidity up to
            ``RH = 1`` at this temperature.
        """
        target = float(saturation_pct)
        if target < 0.0:
            raise ValueError(f"saturation_pct must be non-negative, got {target!r}")
        if target == 0.0:
            return 0.0
        ceiling = self.saturation_pct(temperature_k, 1.0)
        if target > ceiling:
            raise ValueError(
                f"saturation of {target:g} wt% is unreachable at "
                f"{temperature_k:g} K; the RH=1 ceiling is {ceiling:g} wt%"
            )
        return float(target / ceiling)
