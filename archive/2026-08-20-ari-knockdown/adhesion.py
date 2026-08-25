"""Modelled Adhesion Retention Index (ARI) --- a single moisture knockdown.

ARI is the modelled fraction of dry interfacial fracture energy still retained.
Here it is one equation with one parameter: retention falls linearly with the
water that has actually reached the bondline, and recovers completely when the
bondline dries.

    ARI(t) = clip(1 - k * w_interface(t), 0, 1)

Two properties of the driving variable carry the physics.

**It is the interface, not the film average.** On a film bonded to an
impermeable substrate the bondline is the last point in the film to wet, so a
thick film keeps its interface dry long after the gravimetric curve has risen.
Driving ARI from film-average uptake would erase the thickness effect this
framework exists to predict --- normalized uptake is a function of
``Fo = D t / l**2`` alone, so it cannot distinguish a thick film from a
permeable one.

**It is absolute water content in wt%, not normalized moisture.** Normalized
interfacial moisture ``C/C_sat`` tends to 1 given enough time whatever the
humidity, because it is normalized against *this* exposure's saturation.
Driving degradation from it would make a film at 30% RH lose exactly as much
adhesion as one at 90% RH, and would have a bone-dry film destroyed as
thoroughly as an immersed one.

What this model deliberately does not include
---------------------------------------------
There is no time integral, so ARI has **no memory**: it depends only on how wet
the bondline is at that instant, never on how long it has been wet. Dry the film
and adhesion returns to 1 exactly. Real hygrothermal ageing is partly
irreversible, and a real interface also softens as absorbed water depresses the
glass transition toward the service temperature. Neither is modelled here.

A consequence worth knowing: because the retention law itself has no temperature
term, temperature reaches ARI *only* through the transport that produced the
moisture history --- and if the isotherm is temperature-independent (the default),
temperature changes only how fast the film gets there, not the steady-state ARI.

Standing of these numbers
-------------------------
Unlike the diffusion solvers, which are verified against exact analytical
solutions, this is a **phenomenological knockdown with a parameter you supply**.
It is constructed to be bounded and monotone, but absolute ARI values are not
validated against joint-strength measurements. Read ARI as a comparative index
for ranking designs and exposure conditions, not as a prediction of residual
bond strength.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diffusion.base import _first_crossing
from .units import PhysicsError, require_non_negative, require_temperature


@dataclass
class AdhesionResult:
    """ARI history and the interfacial moisture that produced it."""

    time: np.ndarray
    index: np.ndarray
    interface_normalized: np.ndarray
    interface_pct: np.ndarray
    temperature_k: float

    @property
    def final_index(self) -> float:
        """ARI at the end of the exposure."""
        return float(self.index[-1])

    def time_to_index(self, threshold: float) -> float:
        """Return the time at which ARI first falls to ``threshold``.

        Returns ``inf`` when the threshold is never reached in the simulated
        window --- a meaningful engineering answer, not an error. A thin, cool,
        dry exposure legitimately never gets there.
        """
        return _first_crossing(self.time, self.index, float(threshold), rising=False)


@dataclass(frozen=True)
class AdhesionModel:
    """The moisture knockdown law.

    Attributes
    ----------
    knockdown_per_pct:
        ``k``, the fraction of dry adhesion lost per weight percent of water at
        the bondline. Zero means moisture does not affect adhesion at all;
        ``1/k`` is the water content at which the index would reach zero.
    """

    knockdown_per_pct: float = 0.25

    def __post_init__(self) -> None:
        require_non_negative(self.knockdown_per_pct, "knockdown_per_pct")

    def retention(self, moisture_pct) -> np.ndarray:
        """Return retained adhesion fraction for a local water content in wt%.

        Negative inputs are clipped to zero first: solver round-off can leave a
        concentration a fraction of an epsilon below zero, and that would
        otherwise report an index above 1.
        """
        water = np.clip(np.asarray(moisture_pct, dtype=float), 0.0, None)
        return np.clip(1.0 - self.knockdown_per_pct * water, 0.0, 1.0)

    def moisture_at_index(self, index: float) -> float:
        """The bondline water content in wt% at which ARI equals ``index``.

        The exact inverse of :meth:`retention`, which makes the interpretation of
        a threshold concrete: an ARI limit of 0.8 with ``k = 0.25`` is simply a
        budget of 0.8 wt% of water at the bondline.

        Raises
        ------
        PhysicsError
            If ``knockdown_per_pct`` is zero, where the law has no inverse
            because every water content gives an index of 1.
        """
        target = float(index)
        if not 0.0 <= target <= 1.0:
            raise PhysicsError(f"index must lie in [0, 1], got {target!r}")
        if self.knockdown_per_pct == 0.0:
            raise PhysicsError(
                "knockdown_per_pct is zero, so adhesion never varies and the "
                "law has no inverse"
            )
        return (1.0 - target) / self.knockdown_per_pct

    def evaluate(self, time, interface_normalized, temperature_k: float,
                 saturation_pct: float) -> AdhesionResult:
        """Evaluate ARI along an interfacial moisture history.

        Parameters
        ----------
        time:
            Ascending times in seconds.
        interface_normalized:
            ``C_interface/C_sat`` at those times, from a
            :class:`~hygroadh.diffusion.base.DiffusionResult`.
        temperature_k:
            Service temperature, K. Recorded on the result for reporting; the
            knockdown law itself carries no temperature term, so temperature
            reaches ARI only through the transport that produced the history.
        saturation_pct:
            Equilibrium uptake in wt%, used to convert the solver's normalized
            field into the absolute water content the law is driven by.
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
        return AdhesionResult(
            time=t,
            index=self.retention(water_pct),
            interface_normalized=phi,
            interface_pct=water_pct,
            temperature_k=float(temperature_k),
        )
