"""Report the time for moisture to reach the film's far-face threshold.

The far face is the bondline for a film on an impermeable substrate. This is a
transport metric and a proxy for performance change, not a bond-strength
prediction. Thresholds can be normalized to equilibrium or given as weight
percent; the latter depends on humidity and may be unreachable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .diffusion.base import _first_crossing
from .units import PhysicsError, require_positive

ThresholdBasis = Literal["normalized", "wt_pct"]

#: The two admissible bases, for validation and for building UI choices.
BASES: tuple[str, ...] = ("normalized", "wt_pct")


@dataclass(frozen=True)
class MoistureCriterion:
    """A moisture level at the far face, and the basis it is expressed in.

    Attributes
    ----------
    value:
        The threshold. On the ``normalized`` basis a fraction in ``(0, 1)``; on
        the ``wt_pct`` basis an absolute water content in weight percent.
    basis:
        ``"normalized"`` or ``"wt_pct"``.
    """

    value: float = 0.5
    basis: ThresholdBasis = "normalized"

    def __post_init__(self) -> None:
        require_positive(self.value, "threshold value")
        if self.basis not in BASES:
            raise PhysicsError(
                f"basis must be one of {BASES}, got {self.basis!r}"
            )
        if self.basis == "normalized" and self.value >= 1.0:
            raise PhysicsError(
                "a normalized threshold must be below 1: the far face approaches "
                "its equilibrium concentration asymptotically and reaches it only "
                "as time goes to infinity, so a threshold of 1 is never crossed"
            )

    # --- reachability ------------------------------------------------------

    def normalized_level(self, saturation_pct: float) -> float:
        """The threshold expressed as a fraction of equilibrium concentration.

        This is the form the transport solution is compared against. On the
        ``wt_pct`` basis it depends on how much water the film can hold, so a
        drier exposure raises the required fraction and may push it past 1.
        """
        if self.basis == "normalized":
            return float(self.value)
        saturation = float(saturation_pct)
        if saturation <= 0.0:
            return float("inf")
        return float(self.value) / saturation

    def is_reachable(self, saturation_pct: float) -> bool:
        """Whether the threshold can ever be crossed at this saturation."""
        return self.normalized_level(saturation_pct) < 1.0

    def unreachable_reason(self, saturation_pct: float) -> str | None:
        """A one-line explanation, or ``None`` when the threshold is reachable."""
        if self.is_reachable(saturation_pct):
            return None
        return (
            f"the film holds only {float(saturation_pct):.3g} wt% at equilibrium "
            f"here, so a threshold of {self.value:g} wt% at the far face can never "
            "be reached; lower the threshold or raise the humidity"
        )

    # --- evaluation --------------------------------------------------------

    def time_to_threshold(self, time, interface_normalized,
                          saturation_pct: float) -> float:
        """Time at which the far face first reaches the threshold.

        Linearly interpolates between the two bracketing samples. Returns ``inf``
        when the threshold is not reached within the simulated window --- which is
        a meaningful engineering answer, not an error: a thick, cool film
        legitimately may not get there, and on the ``wt_pct`` basis a dry
        exposure may make it impossible at any time.
        """
        level = self.normalized_level(saturation_pct)
        if not np.isfinite(level) or level >= 1.0:
            return float("inf")
        return _first_crossing(
            np.asarray(time, dtype=float),
            np.asarray(interface_normalized, dtype=float),
            level,
            rising=True,
        )

    def moisture_pct(self, saturation_pct: float) -> float:
        """The threshold as an absolute water content in wt%."""
        if self.basis == "wt_pct":
            return float(self.value)
        return float(self.value) * float(saturation_pct)

    def describe(self, saturation_pct: float | None = None) -> str:
        """A short human-readable statement of the criterion."""
        if self.basis == "normalized":
            text = (
                f"far-face moisture reaches {self.value:.0%} of its equilibrium "
                "value"
            )
            if saturation_pct is not None:
                text += f" ({self.moisture_pct(saturation_pct):.3g} wt%)"
            return text
        text = f"far-face moisture reaches {self.value:g} wt%"
        if saturation_pct is not None:
            level = self.normalized_level(saturation_pct)
            if np.isfinite(level):
                text += f" ({level:.0%} of equilibrium)"
        return text
