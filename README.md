# adaptive-quadrature

A tiny, dependency-free Python library that integrates a scalar function on a finite interval using adaptive Simpson's rule to a caller-specified tolerance.

```python
import math
from adaptive_quadrature import AdaptiveSimpson, integrate

value, err = integrate(lambda x: x * x, 0.0, 3.0, tol=1e-10)
# (9.0, ~1e-13)

asi = AdaptiveSimpson(max_depth=50)
value, err = asi.integrate(math.sin, 0.0, 3.141592653589793, tol=1e-12)
# (2.0, ~1e-15)
```

`integrate(f, a, b, tol=1e-9, max_depth=50)` returns a `(value, error_estimate)` tuple. `AdaptiveSimpson(max_depth=50)` exposes the same behavior as a class so you can reuse one configured integrator.

## Why this exists

Numerical integration where the integrand is smooth almost everywhere but has a rough region — a steep slope, a cusp, mild endpoint behavior — is exactly where uniform Simpson quadrature wastes work. Adaptive Simpson refines only where the local error estimate is large, which is the trade-off this library makes: it is cheap and stack-based, but it is *local-error* adaptive, not a rigorous global-error-bound method. If you need a certified bound (e.g. for proof), use a validated-interval library instead.

The tolerance `tol` is interpreted per subinterval in the classical Numerical-Recipes style: each recursive half is asked to meet `tol/2`, and acceptance uses the Richardson ratio `|S_left + S_right - S_whole| <= 15 * tol`. This converges to a tight global answer at a depth driven by where the function is rough, not by `log2(1/tol)`.

## The awkward edge

Non-integrable singularities and discontinuities inside `[a, b]` will drive the recursion to `max_depth`. When that happens the library raises `IntegrationError` rather than silently returning a wrong number; the exception carries `.value` (the best partial estimate) and `.interval` if you want to recover something. Integrable singularities (e.g. `1/sqrt(x)` at the left endpoint) usually converge, but slowly — raise `max_depth` or loosen `tol` if you hit the guard. Infinite intervals are not supported; transform variables first.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

## Limitations

Values are coerced to floats, so very large integers lose precision. If you need
exact integer aggregates over a window, this is the wrong tool.

