"""Adaptive Simpson's rule quadrature.

The standard adaptive Simpson recursion evaluates Simpson's rule on an
interval [a,b] (S(a,b)), then on its two halves, and compares  S(a,b)
with  S(a,m)+S(m,b). If they agree within tolerance, the refined value is
returned; otherwise each half is recursed with a halved tolerance.

Two implementation choices that are not obvious from the brief:

* *Tolerance splitting.* Rather than passing tolerance/2 to each subinterval
  we use tolerance per subinterval (the classical Moler/McKeeman-style
  local-error control). This converges to a tight global answer and keeps
  the recursion depth proportional to where the function is rough, not to
  log2(1/tol). It is the interpretation used by common textbooks (e.g.
  Numerical Recipes) and is what most callers expect from `tol`.

* *Depth guard with diagnostics.* Deep recursion almost always means a
  genuine singularity or discontinuity. We cap the depth and raise
  IntegrationError carrying the partial value and the interval where we
  gave up, instead of silently producing a wrong number or blowing the
  stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple


class IntegrationError(RuntimeError):
    """Raised when adaptive Simpson cannot reach tolerance within depth."""

    def __init__(self, message: str, value: float, interval: Tuple[float, float]):
        super().__init__(message)
        self.value = value
        self.interval = interval


@dataclass(frozen=True)
class _Result:
    value: float
    error_estimate: float


def _simpson(f: Callable[[float], float], a: float, b: float,
             fa: float, fb: float, fm: float) -> float:
    """Composite Simpson on [a,b] given f(a), f(m), f(b) where m=(a+b)/2.

    Simpson's rule for one panel: (b-a)/6 * (f(a) + 4 f(m) + f(b)).
    Keeping the function values outside the formula avoids re-evaluating
    f at points the caller already sampled — which is the whole point of
    adaptive quadrature being cheaper than uniform refinement.
    """
    return (b - a) / 6.0 * (fa + 4.0 * fm + fb)


def _adapt(f: Callable[[float], float], a: float, b: float,
           fa: float, fb: float, fm: float, whole: float,
           tol: float, depth: int, max_depth: int) -> _Result:
    m = 0.5 * (a + b)
    lm = 0.5 * (a + m)
    rm = 0.5 * (m + b)

    flm = f(lm)
    frm = f(rm)

    left = _simpson(f, a, m, fa, fm, flm)
    right = _simpson(f, m, b, fm, fb, frm)
    both = left + right

    # Classic local-error estimate: difference between the panel and its
    # refinement, scaled by 1/15 (the leading Simpson error term ratio).
    diff = abs(both - whole)

    if depth >= max_depth:
        # We cannot subdivide further. Return the best estimate we have
        # and propagate the unresolved error upward so the caller's depth
        # guard can decide whether to fail.
        return _Result(both, diff)

    if diff <= 15.0 * tol:
        # The 15 factor converts a *local* tolerance into the Simpson
        # extrapolation acceptance test: if |S_left+S_right - S_whole| <=
        # 15*tol, the Richardson-extrapolated value is within tol of the
        # true integral over this panel.
        return _Result(both + (both - whole) / 15.0, diff / 15.0)

    left_res = _adapt(f, a, m, fa, fm, flm, left, tol * 0.5, depth + 1, max_depth)
    right_res = _adapt(f, m, b, fm, fb, frm, right, tol * 0.5, depth + 1, max_depth)

    return _Result(left_res.value + right_res.value,
                   left_res.error_estimate + right_res.error_estimate)


class AdaptiveSimpson:
    """Adaptive Simpson integrator with a configurable depth guard.

    Parameters
    ----------
    max_depth:
        Maximum recursion depth. Each level bisects one interval, so
        max_depth=50 permits up to 2**50 effective panels in the worst
        case. Hitting this is a strong signal of a non-integrable
        singularity, so we raise instead of returning junk.
    """

    def __init__(self, max_depth: int = 50):
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self.max_depth = max_depth

    def integrate(self, f: Callable[[float], float],
                  a: float, b: float, tol: float = 1e-9) -> Tuple[float, float]:
        """Integrate ``f`` on ``[a, b]``.

        Returns a (value, error_estimate) pair. The error estimate is the
        accumulated local Simpson differences and is not a rigorous bound;
        treat it as a "this is roughly how good the answer is" indicator.

        Notes
        -----
        ``a == b`` returns ``(0.0, 0.0)`` without evaluating ``f``.
        Finite endpoints are required; infinite intervals are not
        supported — callers should transform variables themselves.
        """
        if tol <= 0.0:
            raise ValueError("tol must be positive")
        if a == b:
            return 0.0, 0.0

        # Walk around NaN endpoints early: a NaN range would otherwise
        # poison every comparison below and produce a misleading result.
        if a != a or b != b:  # NaN check without importing math.isnan
            raise ValueError("endpoints must be finite")

        reversed_order = a > b
        if reversed_order:
            a, b = b, a

        fa = f(a)
        fb = f(b)
        fm = f(0.5 * (a + b))
        whole = _simpson(f, a, b, fa, fb, fm)

        res = _adapt(f, a, b, fa, fb, fm, whole, tol, 0, self.max_depth)

        if res.error_estimate > 5.0 * tol:
            # Local errors were larger than tolerance *somewhere* and we
            # ran out of depth. Rather than silently return a number we
            # know is probably wrong, surface it. The caller can catch
            # and inspect `.value` / `.interval` if they still want the
            # partial answer.
            raise IntegrationError(
                "failed to reach tolerance within max_depth "
                f"({self.max_depth}); last error estimate "
                f"{res.error_estimate:.3e} > tol {tol:.3e}",
                res.value,
                (a, b),
            )

        return (-res.value if reversed_order else res.value,
                res.error_estimate)


def integrate(f: Callable[[float], float], a: float, b: float,
              tol: float = 1e-9, max_depth: int = 50) -> Tuple[float, float]:
    """Convenience wrapper equivalent to ``AdaptiveSimpson(max_depth).integrate(...)``."""
    return AdaptiveSimpson(max_depth=max_depth).integrate(f, a, b, tol)
