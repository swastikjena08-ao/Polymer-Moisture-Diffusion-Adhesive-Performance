"""Numpy compatibility shims for the test suite.

``pyproject.toml`` declares ``numpy>=1.23`` for the library, and the library
really does work that far back --- nothing in ``src/`` uses a numpy 2.0 API. Only
the tests did, through ``np.trapezoid``, which was added in numpy 2.0 (the older
spelling is ``np.trapz``). Rather than raise the declared floor to 2.0 and lose
compatibility the library actually has, the tests use this shim so the whole
suite runs on both, and the declared floor is something that has been verified
rather than assumed.
"""

from __future__ import annotations

import numpy as np

#: ``np.trapezoid`` on numpy >= 2.0, ``np.trapz`` before it.
trapezoid = getattr(np, "trapezoid", None)
if trapezoid is None:  # pragma: no cover - depends on the installed numpy
    trapezoid = np.trapz
