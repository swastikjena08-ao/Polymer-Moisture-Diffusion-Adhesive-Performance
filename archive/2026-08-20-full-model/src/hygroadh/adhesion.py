"""Modelled Adhesion Retention Index (ARI) for the film/substrate interface.

ARI is the modelled fraction of dry interfacial fracture energy still
retained. It is the product of three mechanism factors, each returned
separately so the dominant mechanism is visible rather than buried in a
single number::

    ARI(t) = R_plast(phi) * R_Tg(T, w) * R_hyd(t)

Two choices about the driving variable, both load-bearing.

**It is the interface, not the film average.** On a film bonded to an
impermeable substrate the bondline is the last point to wet, so a thick film
keeps its interface dry long after the gravimetric curve has risen. Driving ARI
from average uptake would erase the thickness effect this framework exists to
predict.

**It is absolute water content in wt%, not normalized moisture.** Normalized
interfacial moisture ``C/C_sat`` always tends to 1 given enough time, because it
is normalized against the saturation of *this* exposure. Driving degradation
from it would make a film at 30% RH degrade exactly as fast as one at 90% RH,
and would have a bone-dry film destroyed as thoroughly as an immersed one.
The mechanism methods therefore take local water content in weight percent,
non-dimensionalized by :attr:`AdhesionModel.moisture_reference_pct`.
:meth:`AdhesionModel.evaluate` does that conversion from the solver's
normalized field, so callers still pass the solver output directly.

Standing of this model
----------------------
Unlike the diffusion solvers, which are verified against exact analytical
solutions, this is a **phenomenological model with user-supplied
parameters**. The functional forms encode accepted degradation mechanisms
and are constructed to be bounded and monotone, but absolute ARI values are
not validated against joint-strength measurements. Read ARI as a
comparative index for ranking designs and exposure conditions, not as a
prediction of residual bond strength.

Mechanisms
----------
1. **Reversible plasticization.** Water in the interphase lowers cohesive
   and interfacial toughness. Decays from 1 toward a residual floor and
   recovers fully on drying.
2. **Thermal softening near the wet Tg.** Absorbed water depresses the glass
   transition toward the service temperature; as the gap closes the
   interphase softens and adhesion collapses. This is how temperature
   reaches adhesion by a route entirely independent of its effect on
   diffusivity.
3. **Irreversible hydrolysis.** Thermally activated chemical attack on
   interfacial bonds, accumulated along the wetting history. Because the
   damage variable is monotone, ARI does not fully recover on drying --- which
   is what makes it history-dependent and distinguishes cyclic from steady
   exposure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .diffusion.base import _first_crossing
from .units import (
    GAS_CONSTANT,
    PhysicsError,
    require_finite,
    require_fraction,
    require_non_negative,
    require_positive,
    require_temperature,
)

#: Names of the three mechanism factors, in the order they are reported.
MECHANISMS = ("plasticization", "thermal", "hydrolysis")


def _logistic_complement(z: np.ndarray) -> np.ndarray:
    """Return ``1/(1+exp(z))`` without overflowing for large positive ``z``.

    The naive form raises a RuntimeWarning and loses precision once ``z``
    exceeds about 700. Branching on the sign keeps every exponential
    argument non-positive.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z > 0.0
    exp_neg = np.exp(-z[positive])
    out[positive] = exp_neg / (1.0 + exp_neg)
    out[~positive] = 1.0 / (1.0 + np.exp(z[~positive]))
    return out


def _cumulative_trapezoid(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Running trapezoidal integral of ``values`` over ``time``, starting at zero."""
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(time)
    return np.concatenate([[0.0], np.cumsum(increments)])


@dataclass
class AdhesionResult:
    """ARI history together with the three factors that produced it."""

    time: np.ndarray
    index: np.ndarray
    plasticization: np.ndarray
    thermal: np.ndarray
    hydrolysis: np.ndarray
    damage: np.ndarray
    interface_normalized: np.ndarray
    interface_pct: np.ndarray
    glass_transition_k: np.ndarray
    temperature_k: float

    @property
    def final_index(self) -> float:
        """ARI at the end of the exposure."""
        return float(self.index[-1])

    @property
    def factors(self) -> dict[str, np.ndarray]:
        """The three mechanism factors keyed by name."""
        return {
            "plasticization": self.plasticization,
            "thermal": self.thermal,
            "hydrolysis": self.hydrolysis,
        }

    @property
    def dominant_mechanism(self) -> str:
        """Whichever factor has fallen furthest by the end of the exposure.

        Reported because two designs can share an ARI while degrading for
        completely different reasons, and the remedy differs: a plasticization
        limit calls for a less hydrophilic interphase, a thermal limit for a
        higher-Tg resin, a hydrolysis limit for a coupling agent.
        """
        return min(self.factors, key=lambda name: float(self.factors[name][-1]))

    def time_to_index(self, threshold: float) -> float:
        """Return the time at which ARI first falls to ``threshold``.

        Returns ``inf`` when the threshold is never reached in the simulated
        window --- a meaningful engineering answer, not an error.
        """
        return _first_crossing(self.time, self.index, float(threshold), rising=False)


@dataclass(frozen=True)
class AdhesionModel:
    """Parameters of the three-mechanism adhesion retention model.

    Every parameter is validated on construction so that ARI is guaranteed to
    lie in ``(0, 1]`` by the algebra of the factors, rather than being forced
    there by clipping afterwards.

    Attributes
    ----------
    moisture_reference_pct:
        The local water content in wt% that non-dimensionalizes the moisture
        driver, so that ``psi = w_interface / moisture_reference_pct``. The
        default of 1 wt% makes ``plasticization_gain`` read directly as "loss
        coefficient per weight percent of water".
    plasticization_gain:
        ``k_p``, per unit of ``psi``. Larger values mean adhesion is lost faster
        per unit of interfacial water.
    plasticization_exponent:
        ``m``. Below 1 the loss is steepest at low moisture; above 1 there is
        a tolerant regime before adhesion falls away.
    plasticization_floor:
        ``r_min``, the residual retention as plasticization saturates. Keeps
        the reversible mechanism from predicting total bond loss on its own.
    tg_offset:
        ``Delta_off`` in K. Softening reaches its half-point this far *below*
        the wet Tg, since modulus falls before the transition proper.
    tg_width:
        ``s_Tg`` in K, the width of the softening transition.
    hydrolysis_rate_ref:
        ``k_h,ref`` in 1/s at ``temperature_ref_k``, for fully wetted
        interface.
    hydrolysis_activation:
        ``Ea_h`` in J/mol.
    hydrolysis_exponent:
        ``q``, the order of the hydrolysis rate in interfacial moisture.
    hydrolysis_max_loss:
        ``xi_max``, the largest fraction of adhesion that irreversible attack
        can remove.
    temperature_ref_k:
        Reference temperature for ``hydrolysis_rate_ref``.
    """

    moisture_reference_pct: float = 1.0
    plasticization_gain: float = 0.15
    plasticization_exponent: float = 1.0
    plasticization_floor: float = 0.55
    tg_offset: float = 20.0
    tg_width: float = 8.0
    hydrolysis_rate_ref: float = 7.0e-8
    hydrolysis_activation: float = 60.0e3
    hydrolysis_exponent: float = 1.0
    hydrolysis_max_loss: float = 0.55
    temperature_ref_k: float = 298.15

    def __post_init__(self) -> None:
        require_positive(self.moisture_reference_pct, "moisture_reference_pct")
        require_non_negative(self.plasticization_gain, "plasticization_gain")
        require_positive(self.plasticization_exponent, "plasticization_exponent")
        require_fraction(self.plasticization_floor, "plasticization_floor")
        require_finite(self.tg_offset, "tg_offset")
        require_positive(self.tg_width, "tg_width")
        require_non_negative(self.hydrolysis_rate_ref, "hydrolysis_rate_ref")
        require_finite(self.hydrolysis_activation, "hydrolysis_activation")
        require_positive(self.hydrolysis_exponent, "hydrolysis_exponent")
        require_fraction(self.hydrolysis_max_loss, "hydrolysis_max_loss")
        require_temperature(self.temperature_ref_k, "temperature_ref_k")

    # --- individual mechanisms ---------------------------------------------

    def moisture_driver(self, moisture_pct) -> np.ndarray:
        """Dimensionless driver ``psi = w / moisture_reference_pct``.

        Negative inputs are clipped to zero: solver round-off can leave a
        concentration a fraction of an epsilon below zero, and a fractional
        exponent would turn that into NaN rather than a harmless zero.
        """
        water = np.clip(np.asarray(moisture_pct, dtype=float), 0.0, None)
        return water / self.moisture_reference_pct

    def plasticization_factor(self, moisture_pct) -> np.ndarray:
        """Reversible retention factor, given local water content in wt%."""
        psi = self.moisture_driver(moisture_pct)
        decay = np.exp(-self.plasticization_gain * psi**self.plasticization_exponent)
        return self.plasticization_floor + (1.0 - self.plasticization_floor) * decay

    def thermal_factor(self, temperature_k: float, glass_transition_k) -> np.ndarray:
        """Softening factor from the gap between service temperature and wet Tg.

        This is the raw logistic, which is always strictly less than 1.
        :meth:`evaluate` divides it by its dry-state value so that the reported
        factor is 1 for a dry interface.
        """
        temperature = require_temperature(temperature_k)
        tg = np.asarray(glass_transition_k, dtype=float)
        return _logistic_complement((temperature - tg + self.tg_offset) / self.tg_width)

    def hydrolysis_rate(self, temperature_k: float) -> float:
        """Arrhenius hydrolysis rate constant in 1/s at the reference water content."""
        temperature = require_temperature(temperature_k)
        exponent = -(self.hydrolysis_activation / GAS_CONSTANT) * (
            1.0 / temperature - 1.0 / self.temperature_ref_k
        )
        return self.hydrolysis_rate_ref * float(np.exp(exponent))

    def hydrolysis_damage(self, time, moisture_pct,
                          temperature_k: float) -> np.ndarray:
        """Integrate ``dxi/dt = k_h psi^q (1 - xi)`` along the wetting history.

        ``moisture_pct`` is local water content at the interface in weight
        percent, so a drier exposure genuinely hydrolyses more slowly.

        Solved in closed form rather than stepped numerically. Writing
        ``a(t) = k_h psi(t)^q``, the equation is linear in ``1 - xi``, so

            1 - xi(t) = exp(-integral of a dt)

        Evaluating that integral by the running trapezoid rule makes the
        result exact for piecewise-linear ``phi`` and keeps ``xi`` in ``[0, 1]``
        on any time grid, however coarse --- no step-size condition, and no
        possibility of overshooting into unphysical territory. In exact
        arithmetic ``xi < 1`` strictly; at extreme exposure the exponential
        underflows and ``xi`` saturates at exactly 1.0, which is the correct
        limit rather than an overshoot.
        """
        t = np.asarray(time, dtype=float)
        psi = self.moisture_driver(moisture_pct)
        if t.shape != psi.shape:
            raise PhysicsError("time and moisture_pct must have the same shape")
        if t.size < 2:
            return np.zeros_like(t)
        rate = self.hydrolysis_rate(temperature_k) * psi**self.hydrolysis_exponent
        return 1.0 - np.exp(-_cumulative_trapezoid(rate, t))

    def hydrolysis_factor(self, damage) -> np.ndarray:
        """Convert accumulated damage into a retention factor."""
        return 1.0 - self.hydrolysis_max_loss * np.asarray(damage, dtype=float)

    # --- combined index ----------------------------------------------------

    def evaluate(
        self,
        time,
        interface_normalized,
        temperature_k: float,
        saturation_pct: float,
        glass_transition_k: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> AdhesionResult:
        """Evaluate ARI along an interfacial moisture history.

        Parameters
        ----------
        time:
            Ascending times in seconds.
        interface_normalized:
            ``C_interface/C_sat`` at those times, from a
            :class:`~hygroadh.diffusion.base.DiffusionResult`.
        temperature_k:
            Service temperature, K.
        saturation_pct:
            Equilibrium uptake in wt%. Used to convert the solver's normalized
            field into the absolute local water content that all three
            mechanisms are driven by.
        glass_transition_k:
            Callable mapping water content in wt% to wet Tg in K, normally
            :meth:`hygroadh.materials.Polymer.glass_transition_k`. Passing
            ``None`` disables the thermal factor (holds it at 1), which is how
            a single mechanism is isolated for study.
        """
        t = np.atleast_1d(np.asarray(time, dtype=float))
        phi = np.atleast_1d(np.asarray(interface_normalized, dtype=float))
        if t.shape != phi.shape:
            raise PhysicsError("time and interface_normalized must have the same shape")
        if np.any(np.diff(t) < 0.0):
            raise PhysicsError("time must be ascending")
        require_temperature(temperature_k)
        saturation = require_non_negative(saturation_pct, "saturation_pct")

        phi = np.clip(phi, 0.0, None)
        water_pct = phi * saturation

        plasticization = self.plasticization_factor(water_pct)
        if glass_transition_k is None:
            tg = np.full_like(t, np.inf)
            thermal = np.ones_like(t)
        else:
            tg = np.atleast_1d(
                np.asarray(glass_transition_k(water_pct), dtype=float)
            )
            # Normalize by the dry-state softening. ARI is the fraction of *dry*
            # adhesion retained, and the dry measurement already contains
            # whatever softening the service temperature causes at Tg(0). Using
            # the raw logistic would double-count it and report ARI < 1 for a
            # bone-dry interface.
            tg_dry = float(np.asarray(glass_transition_k(0.0), dtype=float).reshape(-1)[0])
            baseline = float(self.thermal_factor(temperature_k, tg_dry))
            if baseline < 1e-6:
                raise PhysicsError(
                    f"service temperature {temperature_k:.2f} K is at or above the "
                    f"dry glass transition {tg_dry:.2f} K, so there is no dry "
                    "adhesion baseline for ARI to be a fraction of"
                )
            thermal = self.thermal_factor(temperature_k, tg) / baseline
        damage = self.hydrolysis_damage(t, water_pct, temperature_k)
        hydrolysis = self.hydrolysis_factor(damage)

        # The factors are individually bounded in (0, 1] by construction, so
        # this clip is a backstop against round-off, not the guarantee.
        index = np.clip(plasticization * thermal * hydrolysis, 0.0, 1.0)

        return AdhesionResult(
            time=t,
            index=index,
            plasticization=plasticization,
            thermal=thermal,
            hydrolysis=hydrolysis,
            damage=damage,
            interface_normalized=phi,
            interface_pct=water_pct,
            glass_transition_k=tg,
            temperature_k=float(temperature_k),
        )


def wet_work_of_adhesion(substrate_water: float, polymer_water: float,
                         substrate_polymer: float) -> float:
    """Thermodynamic work of adhesion in the presence of water, in J/m^2.

    ``W_A,wet = gamma_sw + gamma_pw - gamma_sp`` from the interfacial energies
    of the substrate/water, polymer/water, and substrate/polymer interfaces.
    Reported as a standalone screening diagnostic and deliberately *not* folded
    into ARI: it is an equilibrium statement about whether debonding is
    favourable, carrying no rate information, so multiplying it into a kinetic
    index would confuse two different kinds of claim.
    """
    return (
        require_finite(substrate_water, "substrate_water")
        + require_finite(polymer_water, "polymer_water")
        - require_finite(substrate_polymer, "substrate_polymer")
    )


def interface_is_displaceable(substrate_water: float, polymer_water: float,
                              substrate_polymer: float) -> bool:
    """Whether water can spontaneously displace the polymer from the substrate.

    True when the wet work of adhesion is negative, meaning debonding lowers
    the system's free energy and no amount of kinetic protection makes the
    interface stable in the long run.
    """
    return wet_work_of_adhesion(substrate_water, polymer_water, substrate_polymer) < 0.0
