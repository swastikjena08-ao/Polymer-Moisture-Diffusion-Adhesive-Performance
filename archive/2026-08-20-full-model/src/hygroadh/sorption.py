"""Equilibrium moisture sorption isotherms.

The isotherm sets the *saturation* moisture content the film equilibrates
to at a given temperature and relative humidity. It is what makes
``M/M_inf`` normalizable, and it converts the dimensionless transport
solution into a physical weight-percent uptake that the adhesion model
can act on.
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
class PowerLawIsotherm:
    """Power-law (Freundlich-like) isotherm with a van 't Hoff temperature term.

    ``M_sat(T, RH) = m_ref * (RH/RH_ref)**b * exp(-(dH_s/R)(1/T - 1/T_ref))``

    ``b = 1`` recovers Henry's law, which is a good description of water in
    many lightly polar polymers. ``b < 1`` describes the concave isotherm
    typical of epoxies at high humidity; ``b > 1`` the convex,
    clustering-dominated shape seen in some hydrophobic films.

    ``enthalpy_sorption`` may be either sign. Positive makes saturation
    uptake increase with temperature, which is the usual observation for
    epoxies over a modest range.

    Attributes
    ----------
    m_ref_pct:
        Saturation moisture content in weight percent at the reference
        temperature and humidity.
    rh_ref:
        Reference relative humidity as a fraction in (0, 1].
    exponent:
        Humidity exponent ``b``.
    enthalpy_sorption:
        ``dH_s`` in J/mol.
    temperature_ref_k:
        Reference temperature in K.
    """

    m_ref_pct: float
    rh_ref: float = 1.0
    exponent: float = 1.0
    enthalpy_sorption: float = 0.0
    temperature_ref_k: float = 298.15

    def __post_init__(self) -> None:
        require_positive(self.m_ref_pct, "m_ref_pct")
        rh_ref = require_fraction(self.rh_ref, "rh_ref")
        if rh_ref == 0.0:
            raise ValueError("rh_ref must be greater than zero")
        require_positive(self.exponent, "exponent")
        require_finite(self.enthalpy_sorption, "enthalpy_sorption")
        require_temperature(self.temperature_ref_k, "temperature_ref_k")

    def _temperature_factor(self, temperature_k: float) -> float:
        t = require_temperature(temperature_k)
        t_ref = self.temperature_ref_k
        exponent = -(self.enthalpy_sorption / GAS_CONSTANT) * (1.0 / t - 1.0 / t_ref)
        return float(np.exp(exponent))

    def saturation_pct(self, temperature_k: float, relative_humidity: float) -> float:
        """Return equilibrium moisture content in weight percent."""
        rh = require_fraction(relative_humidity, "relative_humidity")
        if rh == 0.0:
            return 0.0
        return (
            self.m_ref_pct
            * (rh / self.rh_ref) ** self.exponent
            * self._temperature_factor(temperature_k)
        )

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
        scale = self.m_ref_pct * self._temperature_factor(temperature_k)
        return float(self.rh_ref * (target / scale) ** (1.0 / self.exponent))
