import math
import unittest

from adaptive_quadrature import AdaptiveSimpson, integrate, IntegrationError


class TestPolynomialHappyPath(unittest.TestCase):
    """Simpson's rule is exact for cubics, so these should be spot on."""

    def test_constant(self):
        val, err = integrate(lambda x: 3.0, 0.0, 2.0, tol=1e-12)
        self.assertAlmostEqual(val, 6.0, places=12)
        self.assertGreaterEqual(err, 0.0)

    def test_linear(self):
        val, _ = integrate(lambda x: 2.0 * x + 1.0, -1.0, 3.0, tol=1e-12)
        # Integral of 2x+1 on [-1,3] = [x^2+x] from -1 to 3 = (9+3)-(1-1)=12
        self.assertAlmostEqual(val, 12.0, places=11)

    def test_cubic_exact(self):
        # Simpson is exact for degree <= 3.
        val, _ = integrate(lambda x: x ** 3 - 2 * x ** 2 + x - 7, 0.0, 2.0, tol=1e-12)
        exact = (4.0 - 16.0 / 3.0 + 2.0 - 14.0) - (0.0 - 0.0 + 0.0 - 0.0)
        self.assertAlmostEqual(val, exact, places=10)


class TestNonTrivialFunctions(unittest.TestCase):

    def test_sine_over_period(self):
        val, _ = integrate(math.sin, 0.0, 2.0 * math.pi, tol=1e-10)
        self.assertAlmostEqual(val, 0.0, places=9)

    def test_sine_quarter_period(self):
        val, _ = integrate(math.sin, 0.0, math.pi / 2.0, tol=1e-10)
        self.assertAlmostEqual(val, 1.0, places=9)

    def test_gaussian(self):
        # Integral of exp(-x^2) over (-inf, inf) is sqrt(pi); over (-3,3)
        # it is erf(3)*sqrt(pi) = 0.9999779... * sqrt(pi).
        f = lambda x: math.exp(-x * x)
        val, _ = integrate(f, -3.0, 3.0, tol=1e-10)
        expected = math.erf(3.0) * math.sqrt(math.pi)
        self.assertAlmostEqual(val, expected, places=8)

    def test_highly_oscillatory(self):
        # sin(50 x) over [0, 2pi] integrates to 0; adaptive refinement
        # has to work hard here. Use a modest tolerance and generous depth.
        val, _ = integrate(lambda x: math.sin(50.0 * x), 0.0, 2.0 * math.pi,
                          tol=1e-6, max_depth=40)
        self.assertAlmostEqual(val, 0.0, places=4)


class TestEdgeCases(unittest.TestCase):

    def test_zero_width_interval(self):
        # Must not evaluate f at all — important if f has a removable
        # singularity at the endpoint that the caller is avoiding.
        calls = {"n": 0}

        def f(x):
            calls["n"] += 1
            return 1.0 / x if x != 0.0 else float("inf")

        val, err = integrate(f, 1.0, 1.0, tol=1e-9)
        self.assertEqual(val, 0.0)
        self.assertEqual(err, 0.0)
        self.assertEqual(calls["n"], 0)

    def test_reversed_endpoints(self):
        # integrate(f, b, a) = -integrate(f, a, b). This is the standard
        # orientation convention and lets callers write things like
        # integrate(f, x_upper, x_lower) without flipping signs manually.
        val_fwd, _ = integrate(lambda x: x * x, 0.0, 3.0, tol=1e-10)
        val_rev, _ = integrate(lambda x: x * x, 3.0, 0.0, tol=1e-10)
        self.assertAlmostEqual(val_fwd, 9.0, places=9)
        self.assertAlmostEqual(val_rev, -9.0, places=9)

    def test_negative_interval(self):
        val, _ = integrate(lambda x: x * x, -2.0, -1.0, tol=1e-10)
        # Integral of x^2 on [-2,-1] = (1/3)(-1 - (-8)) = 7/3.
        self.assertAlmostEqual(val, 7.0 / 3.0, places=9)

    def test_asymmetric_endpoint_values(self):
        # Make sure fa/fb/fm are wired in the right order — a classic
        # off-by-one is to swap fa and fb, which still integrates cubics
        # correctly (Simpson is symmetric) but breaks asymmetric cases.
        val, _ = integrate(lambda x: math.sin(x) + math.cos(x), 0.0, 1.0, tol=1e-10)
        exact = (1.0 - math.cos(1.0)) + math.sin(1.0)
        self.assertAlmostEqual(val, exact, places=9)

    def test_negative_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            integrate(lambda x: x, 0.0, 1.0, tol=-1e-9)

    def test_zero_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            integrate(lambda x: x, 0.0, 1.0, tol=0.0)

    def test_max_depth_must_be_positive(self):
        with self.assertRaises(ValueError):
            AdaptiveSimpson(max_depth=0)

    def test_nan_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            integrate(lambda x: x, float("nan"), 1.0, tol=1e-9)


class TestDepthGuard(unittest.TestCase):

    def test_depth_failure_raises(self):
        # f(x) = 1/sqrt(x) has an integrable singularity at 0 (true value
        # 2*sqrt(b)), but the *adaptive* error test will keep refusing to
        # accept because the error near 0 shrinks slowly. With a tiny
        # max_depth we force the guard to fire.
        def f(x):
            return 1.0 / math.sqrt(x) if x > 0.0 else float("inf")

        with self.assertRaises(IntegrationError) as ctx:
            AdaptiveSimpson(max_depth=3).integrate(f, 1e-6, 1.0, tol=1e-12)

        self.assertIsInstance(ctx.exception.value, float)
        self.assertEqual(len(ctx.exception.interval), 2)

    def test_depth_failure_on_smooth_with_low_depth(self):
        # Even a smooth function will trip the guard if max_depth is
        # absurdly low relative to tol — sanity check that the guard is
        # driven by error_estimate, not just depth.
        with self.assertRaises(IntegrationError):
            AdaptiveSimpson(max_depth=1).integrate(
                lambda x: math.sin(7.0 * x), 0.0, 10.0, tol=1e-14
            )


class TestClassUsage(unittest.TestCase):

    def test_reuse_integrator(self):
        asi = AdaptiveSimpson()
        v1, _ = asi.integrate(lambda x: x * x, 0.0, 1.0, tol=1e-10)
        v2, _ = asi.integrate(math.sin, 0.0, math.pi, tol=1e-10)
        self.assertAlmostEqual(v1, 1.0 / 3.0, places=9)
        self.assertAlmostEqual(v2, 2.0, places=9)

    def test_error_estimate_nonneg(self):
        # The accumulated error estimate is never negative by construction
        # (we sum absolute values of local diffs / 15).
        _, err = integrate(lambda x: math.exp(x), 0.0, 1.0, tol=1e-10)
        self.assertGreaterEqual(err, 0.0)


if __name__ == "__main__":
    unittest.main()
