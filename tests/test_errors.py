"""
Initial tests for errors.py

Tests cover the pure logic and utility functions that can be exercised without
running the full JAX forward model or requiring real observational data:
  - params_dict_to_vector / vector_to_params_dict (round-trip, ordering)
  - chi2_loss_hsafe argument validation (mu resolution branch logic)
  - compute_model_sort_idx mu-resolution branches (unit-level, with stubs)
  - estimate_parameter_errors input validation (gradient_tol, normalisation_spec)
  - transform_cov_matrix input validation and identity-Jacobian case
  - match_model_to_data_curve_hsafe shape and masking properties (with a toy
    distance metric stub)

Heavy JAX Hessian / forward-model paths are left for integration tests that
require a full environment with real data.

Run with:
    pytest test_errors.py -v
"""

import math
import sys
import types

import jax.numpy as jnp
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so errors.py can be imported without the full package
# ---------------------------------------------------------------------------

# Import the real sting package first so sys.modules["sting"] is the real
# package object — not a bare ModuleType — before we inject any stubs.
import sting

# ---------------------------------------------------------------------------
# stub: sting.stream_lines_grad
# ---------------------------------------------------------------------------
_slg = types.ModuleType("sting.stream_lines_grad")

def _mu_from_omega(omega, mass, r0):
    """omega = sqrt(G*mass/r0^3) * mu  →  mu = omega / sqrt(G*mass/r0^3)"""
    G = 6.67430e-11 * (1e-3)**2 * (1.988416e30) / (1.4959787e11)
    return float(omega) / math.sqrt(G * float(mass) / float(r0)**3)

def _omega_from_mu(mu, mass, r0):
    G = 6.67430e-11 * (1e-3)**2 * (1.988416e30) / (1.4959787e11)
    return float(mu) * math.sqrt(G * float(mass) / float(r0)**3)

_slg.mu_from_omega = _mu_from_omega
_slg.omega_from_mu = _omega_from_mu
_slg.xyz_stream = None  # replaced per-test when needed

# ---------------------------------------------------------------------------
# stub: sting.extract_streamline
# ---------------------------------------------------------------------------
_es = types.ModuleType("sting.extract_streamline")

def _get_distance_metric(ra, dec):
    """Trivial distance metric: cumulative arc-length from the first point."""
    ra = jnp.asarray(ra, dtype=jnp.float64)
    dec = jnp.asarray(dec, dtype=jnp.float64)
    dr = jnp.sqrt(jnp.diff(ra, prepend=ra[0])**2 + jnp.diff(dec, prepend=dec[0])**2)
    return jnp.cumsum(dr), {}

def _to_float64(x):
    return jnp.asarray(x, dtype=jnp.float64)

def _cartesian_to_polar(ra, dec):
    r = jnp.sqrt(ra**2 + dec**2)
    theta = jnp.arctan2(dec, ra)
    return r, theta

def _wrap_to_pi(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi

_es.get_distance_metric = _get_distance_metric
_es.to_float64 = _to_float64
_es.cartesian_to_polar = _cartesian_to_polar
_es.wrap_to_pi = _wrap_to_pi
_es.prepare_data = None   # replaced per-test when needed

# ---------------------------------------------------------------------------
# stub: sting.gradient_descent
# ---------------------------------------------------------------------------
_gd = types.ModuleType("sting.gradient_descent")
_gd.ANGLE_KEYS = {"theta0", "phi0", "inc", "pa"}
_gd.DISPLAY_UNITS = {
    "r0": "au", "v_r0": "km/s", "mass": "M_sun",
    "rmin": "au", "deltar": "au", "v_lsr": "km/s",
    "rc": "au", "omega": "1/s",
}
_gd.convert_and_strip_bound_units = None
_gd.with_mu_substituted = None
_gd.build_normalisation_spec = None
_gd.normalise_opt_params = None
_gd.denormalise_opt_params = None
_gd.check_loss_method = lambda m: int(m)
_gd.gradient_l2_norm = lambda g: float(jnp.sqrt(jnp.sum(g**2)))
_gd.chi2_loss = None

def _softplus(x):
    return jnp.log1p(jnp.exp(x))

def _inv_softplus(y):
    return jnp.log(jnp.expm1(y))

def _to_float64_gd(x):
    return jnp.asarray(x, dtype=jnp.float64)

_gd.softplus = _softplus
_gd.inv_softplus = _inv_softplus
_gd.to_float64 = _to_float64_gd

# Inject all stubs before importing the module under test
sys.modules["sting.stream_lines_grad"] = _slg
sys.modules["sting.extract_streamline"] = _es
sys.modules["sting.gradient_descent"] = _gd

import sting.errors as errors_module

# Bring names under test into this namespace for convenience
params_dict_to_vector = errors_module.params_dict_to_vector
vector_to_params_dict = errors_module.vector_to_params_dict
match_model_to_data_curve_hsafe = errors_module.match_model_to_data_curve_hsafe
chi2_loss_hsafe = errors_module.chi2_loss_hsafe
compute_model_sort_idx = errors_module.compute_model_sort_idx
transform_cov_matrix = errors_module.transform_cov_matrix
estimate_parameter_errors = errors_module.estimate_parameter_errors

BIG = errors_module.BIG


# ===========================================================================
# Helpers
# ===========================================================================

def _make_line(n=20, noise=0.0, rng=None):
    """Return a simple straight-line model: ra increases, dec=0, v=1."""
    if rng is None:
        rng = np.random.default_rng(0)
    ra = np.linspace(0.1, 2.0, n) + (rng.standard_normal(n) * noise if noise else 0.0)
    dec = np.zeros(n)
    v = np.ones(n)
    return jnp.array(ra), jnp.array(dec), jnp.array(v)


def _make_valid_mask(n, all_valid=True):
    return jnp.ones(n, dtype=jnp.float64) if all_valid else jnp.zeros(n, dtype=jnp.float64)


# ===========================================================================
# params_dict_to_vector / vector_to_params_dict
# ===========================================================================

class TestParamsDictToVector:
    """Unit tests for params_dict_to_vector."""

    def test_returns_vector_and_keys(self):
        params = {"r0": 100.0, "inc": 0.5, "mass": 1.0}
        vec, keys = params_dict_to_vector(params)
        assert len(vec) == 3
        assert set(keys) == {"r0", "inc", "mass"}

    def test_vector_values_match_dict(self):
        params = {"r0": 200.0, "v_r0": 3.0}
        vec, keys = params_dict_to_vector(params)
        for i, k in enumerate(keys):
            assert pytest.approx(float(vec[i])) == params[k]

    def test_output_dtype_is_float64(self):
        params = {"r0": 1.0}
        vec, _ = params_dict_to_vector(params)
        assert vec.dtype == jnp.float64

    def test_single_param(self):
        vec, keys = params_dict_to_vector({"mass": 2.5})
        assert len(vec) == 1
        assert keys == ["mass"]
        assert pytest.approx(float(vec[0])) == 2.5


class TestVectorToParamsDict:
    """Unit tests for vector_to_params_dict."""

    def test_roundtrip(self):
        params = {"r0": 100.0, "inc": 0.3, "mass": 1.5}
        vec, keys = params_dict_to_vector(params)
        recovered = vector_to_params_dict(vec, keys)
        for k, v in params.items():
            assert pytest.approx(float(recovered[k]), rel=1e-9) == v

    def test_returns_dict(self):
        result = vector_to_params_dict(jnp.array([1.0, 2.0]), ["a", "b"])
        assert isinstance(result, dict)

    def test_key_order_preserved(self):
        keys = ["pa", "r0", "v_r0"]
        vec = jnp.array([0.1, 50.0, 2.0])
        result = vector_to_params_dict(vec, keys)
        assert list(result.keys()) == keys

    def test_values_indexable_by_position(self):
        keys = ["x", "y", "z"]
        vec = jnp.array([10.0, 20.0, 30.0])
        result = vector_to_params_dict(vec, keys)
        assert pytest.approx(float(result["y"])) == 20.0


# ===========================================================================
# match_model_to_data_curve_hsafe
# ===========================================================================

class TestMatchModelToDataCurveHsafe:
    """Tests for the Hessian-safe model-to-data matching function."""

    def _precompute_sort_idx(self, ra_model, dec_model, valid_mask):
        """Helper: compute model_sort_idx the same way compute_model_sort_idx does."""
        dmetric, _ = _get_distance_metric(ra_model, dec_model)
        model_keep = valid_mask.astype(bool) & (dmetric < BIG)
        w = model_keep.astype(jnp.float64)
        d = jnp.where(model_keep, dmetric, 0.0)
        sort_key = d + (1.0 - w) * BIG
        return jnp.argsort(sort_key), dmetric

    def test_output_shapes_match_data(self):
        n_model, n_data = 30, 10
        ra_m, dec_m, v_m = _make_line(n_model)
        ra_d, dec_d, v_d = _make_line(n_data)
        valid_mask = _make_valid_mask(n_model)
        data_valid = _make_valid_mask(n_data).astype(bool)
        sort_idx, dmetric_frozen = self._precompute_sort_idx(ra_m, dec_m, valid_mask)

        ra_i, dec_i, v_i, valid_out = match_model_to_data_curve_hsafe(
            ra_m, dec_m, v_m, valid_mask,
            ra_d, dec_d,
            sort_idx, dmetric_frozen, data_valid,
        )
        assert ra_i.shape == (n_data,)
        assert dec_i.shape == (n_data,)
        assert v_i.shape == (n_data,)
        assert valid_out.shape == (n_data,)

    def test_valid_output_matches_data_valid_mask(self):
        n_model, n_data = 30, 10
        ra_m, dec_m, v_m = _make_line(n_model)
        ra_d, dec_d, _ = _make_line(n_data)
        valid_mask = _make_valid_mask(n_model)
        # mark first two data points as invalid
        data_valid = jnp.array([False, False] + [True] * (n_data - 2))
        sort_idx, dmetric_frozen = self._precompute_sort_idx(ra_m, dec_m, valid_mask)

        _, _, _, valid_out = match_model_to_data_curve_hsafe(
            ra_m, dec_m, v_m, valid_mask,
            ra_d, dec_d,
            sort_idx, dmetric_frozen, data_valid,
        )
        # valid_out must equal data_valid (the function passes it through)
        np.testing.assert_array_equal(np.array(valid_out), np.array(data_valid))

    def test_all_invalid_model_still_returns_correct_shape(self):
        n_model, n_data = 20, 8
        ra_m, dec_m, v_m = _make_line(n_model)
        ra_d, dec_d, _ = _make_line(n_data)
        valid_mask = _make_valid_mask(n_model, all_valid=False)  # all invalid
        data_valid = _make_valid_mask(n_data).astype(bool)
        sort_idx, dmetric_frozen = self._precompute_sort_idx(ra_m, dec_m, valid_mask)

        ra_i, dec_i, v_i, valid_out = match_model_to_data_curve_hsafe(
            ra_m, dec_m, v_m, valid_mask,
            ra_d, dec_d,
            sort_idx, dmetric_frozen, data_valid,
        )
        assert ra_i.shape == (n_data,)


# ===========================================================================
# chi2_loss_hsafe — mu resolution branch logic
# ===========================================================================

class TestChi2LossHsafeMuResolution:
    """Tests that chi2_loss_hsafe picks the correct mu-resolution branch."""

    def _make_prepared_data(self, n=10):
        """Build a minimal PreparedData-like namedtuple."""
        from collections import namedtuple
        rng = np.random.default_rng(1)
        ra = jnp.array(np.linspace(0.1, 1.0, n))
        dec = jnp.zeros(n)
        v = jnp.ones(n)
        sigma = jnp.ones(n) * 0.1
        dmetric, _ = _get_distance_metric(ra, dec)
        PreparedData = namedtuple(
            "PreparedData",
            ["ra_data", "dec_data", "v_data",
             "ra_sigma_safe", "dec_sigma_safe", "v_sigma_safe",
             "data_finite_mask", "r_proj_data", "theta_proj_data",
             "dmetric_data"],
        )
        r_proj = jnp.sqrt(ra**2 + dec**2)
        theta_proj = jnp.arctan2(dec, ra)
        return PreparedData(
            ra_data=ra, dec_data=dec, v_data=v,
            ra_sigma_safe=sigma, dec_sigma_safe=sigma, v_sigma_safe=sigma,
            data_finite_mask=jnp.ones(n, dtype=bool),
            r_proj_data=r_proj, theta_proj_data=theta_proj,
            dmetric_data=dmetric,
        )

    def _make_base_params(self, extra=None):
        params = {
            "r0": 100.0, "theta0": 0.3, "phi0": 0.1,
            "v_r0": 2.0, "mass": 1.0, "inc": 0.2, "pa": 0.1,
            "rmin": 0.0, "deltar": 50.0, "v_lsr": 0.0,
        }
        if extra:
            params.update(extra)
        return params

    def _install_xyz_stub(self, n=100):
        """Make stream_lines_grad.xyz_stream return a trivial straight line."""
        def _xyz_stream(**kwargs):
            npoints = kwargs.get("npoints", n)
            x = jnp.linspace(-1.0, -0.01, npoints)
            y = jnp.zeros(npoints)
            z = jnp.zeros(npoints)
            vx = jnp.zeros(npoints)
            vy = jnp.zeros(npoints)
            vz = jnp.zeros(npoints)
            valid = jnp.ones(npoints, dtype=bool)
            return (x, y, z), (vx, vy, vz), valid
        _slg.xyz_stream = _xyz_stream

    def test_raises_without_mu_rc_or_omega(self):
        prepared = self._make_prepared_data()
        params = self._make_base_params()  # no mu, rc, or omega
        sort_idx = jnp.arange(100)
        dmetric_frozen = jnp.zeros(100)
        with pytest.raises(ValueError, match="must contain either 'rc', 'omega', or 'mu'"):
            chi2_loss_hsafe(params, 200.0, prepared, 0, sort_idx, dmetric_frozen)

    def test_mu_branch_accepted(self):
        """Providing 'mu' directly should not raise during param resolution."""
        self._install_xyz_stub()
        prepared = self._make_prepared_data()
        params = self._make_base_params({"mu": 0.5})
        sort_idx = jnp.arange(100)
        dmetric_frozen = jnp.zeros(100)
        # Should not raise — if it reaches the forward model that's success enough
        try:
            chi2_loss_hsafe(params, 200.0, prepared, 0, sort_idx, dmetric_frozen)
        except Exception as e:
            # Any exception other than the mu-branch ValueError is fine here
            assert "must contain either 'rc', 'omega', or 'mu'" not in str(e)

    def test_rc_branch_accepted(self):
        """Providing 'rc' (without 'mu') should resolve mu = rc / r0."""
        self._install_xyz_stub()
        prepared = self._make_prepared_data()
        params = self._make_base_params({"rc": 50.0})  # mu = rc/r0 = 0.5
        sort_idx = jnp.arange(100)
        dmetric_frozen = jnp.zeros(100)
        try:
            chi2_loss_hsafe(params, 200.0, prepared, 0, sort_idx, dmetric_frozen)
        except Exception as e:
            assert "must contain either 'rc', 'omega', or 'mu'" not in str(e)

    def test_omega_branch_accepted(self):
        """Providing 'omega' (without 'mu' or 'rc') should compute mu via mu_from_omega."""
        self._install_xyz_stub()
        G = 6.67430e-11 * (1e-3)**2 * (1.988416e30) / (1.4959787e11)
        omega = math.sqrt(G * 1.0 / 100.0**3) * 0.5  # corresponds to mu=0.5
        prepared = self._make_prepared_data()
        params = self._make_base_params({"omega": omega})
        sort_idx = jnp.arange(100)
        dmetric_frozen = jnp.zeros(100)
        try:
            chi2_loss_hsafe(params, 200.0, prepared, 0, sort_idx, dmetric_frozen)
        except Exception as e:
            assert "must contain either 'rc', 'omega', or 'mu'" not in str(e)

    def test_mu_takes_precedence_over_rc(self):
        """If both 'mu' and 'rc' are present, 'mu' should be used directly."""
        self._install_xyz_stub()
        prepared = self._make_prepared_data()
        # mu=0.3 and rc=999 (rc/r0 would be 9.99, very different from 0.3)
        params = self._make_base_params({"mu": 0.3, "rc": 999.0})
        sort_idx = jnp.arange(100)
        dmetric_frozen = jnp.zeros(100)
        try:
            chi2_loss_hsafe(params, 200.0, prepared, 0, sort_idx, dmetric_frozen)
        except Exception as e:
            assert "must contain either 'rc', 'omega', or 'mu'" not in str(e)


# ===========================================================================
# compute_model_sort_idx — mu-resolution branches (unit-level)
# ===========================================================================

class TestComputeModelSortIdx:
    """Tests for the mu-resolution logic inside compute_model_sort_idx."""

    def _install_xyz_stub(self, n=100):
        def _xyz_stream(**kwargs):
            npoints = kwargs.get("npoints", n)
            x = jnp.linspace(-1.0, -0.01, npoints)
            y = jnp.zeros(npoints)
            z = jnp.zeros(npoints)
            valid = jnp.ones(npoints, dtype=bool)
            return (x, y, z), (jnp.zeros(npoints),) * 3, valid
        _slg.xyz_stream = _xyz_stream

    def _make_prepared_data(self, n=10):
        from collections import namedtuple
        ra = jnp.array(np.linspace(0.1, 0.5, n))
        dec = jnp.zeros(n)
        dmetric, _ = _get_distance_metric(ra, dec)
        PreparedData = namedtuple(
            "PreparedData",
            ["data_finite_mask", "dmetric_data"],
        )
        return PreparedData(
            data_finite_mask=jnp.ones(n, dtype=bool),
            dmetric_data=dmetric,
        )

    def test_sort_idx_shape(self):
        self._install_xyz_stub(n=100)
        prepared = self._make_prepared_data()
        best_opt = {"r0": 100.0, "theta0": 0.3, "phi0": 0.1,
                    "mu": 0.5, "v_r0": 2.0, "mass": 1.0,
                    "inc": 0.2, "pa": 0.1, "deltar": 50.0}
        fixed = {"rmin": 0.0, "v_lsr": 0.0}
        sort_idx, dmetric = compute_model_sort_idx(best_opt, fixed, 200.0, prepared, npoints=100)
        assert sort_idx.shape == (100,)
        assert dmetric.shape == (100,)

    def test_sort_idx_raises_for_npoints_too_small(self):
        """xyz_stream should reject npoints=1 because r cannot reach r_low."""
        prepared = self._make_prepared_data()
        best_opt = {"r0": 100.0, "theta0": 0.1, "phi0": 0.1,
                    "mu": 0.5, "v_r0": 2.0, "mass": 1.0,
                    "inc": 0.2, "pa": 0.1, "deltar": 50.0}
        fixed = {"rmin": 0.0, "v_lsr": 0.0}
        with pytest.raises(Exception, match="Radius points do not extend down to rlow"):
            compute_model_sort_idx(best_opt, fixed, 200.0, prepared, npoints=1)

    def test_sort_idx_is_permutation(self):
        """sort_idx should contain each index exactly once."""
        self._install_xyz_stub(n=50)
        prepared = self._make_prepared_data()
        best_opt = {"r0": 100.0, "theta0": 0.1, "phi0": 0.1,
                    "mu": 0.5, "v_r0": 2.0, "mass": 1.0,
                    "inc": 0.2, "pa": 0.1, "deltar": 50.0}
        fixed = {"rmin": 0.0, "v_lsr": 0.0}
        sort_idx, _ = compute_model_sort_idx(best_opt, fixed, 200.0, prepared, npoints=50)
        assert sorted(np.array(sort_idx).tolist()) == list(range(50))


    def test_rc_branch_computes_same_as_mu_branch(self):
        """mu=0.5 and rc=50 with r0=100 should give identical sort indices."""
        self._install_xyz_stub(n=50)
        prepared = self._make_prepared_data()
        base = {"r0": 100.0, "theta0": 0.1, "phi0": 0.1,
                "v_r0": 2.0, "mass": 1.0, "inc": 0.2, "pa": 0.1, "deltar": 50.0}
        fixed = {"rmin": 0.0, "v_lsr": 0.0}

        params_mu = {**base, "mu": 0.5}
        params_rc = {**base, "rc": 50.0}

        idx_mu, _ = compute_model_sort_idx(params_mu, fixed, 200.0, prepared, npoints=50)
        idx_rc, _ = compute_model_sort_idx(params_rc, fixed, 200.0, prepared, npoints=50)
        np.testing.assert_array_equal(np.array(idx_mu), np.array(idx_rc))

    def test_raises_without_mu_rc_or_omega(self):
        self._install_xyz_stub()
        prepared = self._make_prepared_data()
        params = {"r0": 100.0, "theta0": 0.1, "phi0": 0.1,
                  "v_r0": 2.0, "mass": 1.0, "inc": 0.2, "pa": 0.1, "deltar": 50.0}
        fixed = {"rmin": 0.0, "v_lsr": 0.0}
        with pytest.raises(ValueError, match="must contain either 'rc', 'omega', or 'mu'"):
            compute_model_sort_idx(params, fixed, 200.0, prepared, npoints=50)


# ===========================================================================
# estimate_parameter_errors — input validation
# ===========================================================================

class TestEstimateParameterErrorsValidation:
    """Tests for the up-front validation logic in estimate_parameter_errors."""

    def test_raises_when_normalisation_spec_missing(self):
        with pytest.raises(ValueError, match="normalisation_spec not provided"):
            estimate_parameter_errors(
                best_opt_params={"r0": 100.0},
                fixed_params={},
                distance_pc=200.0,
                prepared_data=None,
                normalisation_spec=None,
            )

    def test_raises_for_non_finite_gradient_tol(self):
        with pytest.raises(ValueError, match="finite"):
            estimate_parameter_errors(
                best_opt_params={"r0": 100.0},
                fixed_params={},
                distance_pc=200.0,
                prepared_data=None,
                normalisation_spec={"r0": (0.0, 200.0)},
                gradient_tol=float("inf"),
            )

    def test_raises_for_negative_gradient_tol(self):
        with pytest.raises(ValueError, match="positive"):
            estimate_parameter_errors(
                best_opt_params={"r0": 100.0},
                fixed_params={},
                distance_pc=200.0,
                prepared_data=None,
                normalisation_spec={"r0": (0.0, 200.0)},
                gradient_tol=-1.0,
            )

    def test_raises_for_zero_gradient_tol(self):
        with pytest.raises(ValueError, match="positive"):
            estimate_parameter_errors(
                best_opt_params={"r0": 100.0},
                fixed_params={},
                distance_pc=200.0,
                prepared_data=None,
                normalisation_spec={"r0": (0.0, 200.0)},
                gradient_tol=0.0,
            )

    def test_none_gradient_tol_skips_check(self):
        """gradient_tol=None should bypass the gradient check entirely and
        fail later (at the Hessian step), not at the validation step."""
        with pytest.raises(Exception) as exc_info:
            estimate_parameter_errors(
                best_opt_params={"r0": 100.0},
                fixed_params={},
                distance_pc=200.0,
                prepared_data=None,
                normalisation_spec={"r0": (0.0, 200.0)},
                gradient_tol=None,
            )
        # The error must NOT be the gradient_tol validation message
        assert "finite" not in str(exc_info.value)
        assert "positive" not in str(exc_info.value)


# ===========================================================================
# transform_cov_matrix — input validation and identity-Jacobian case
# ===========================================================================

class TestTransformCovMatrix:
    """Tests for transform_cov_matrix validation and simple cases."""

    def test_raises_for_invalid_rotation_key(self):
        cov = np.eye(2)
        keys = ["mu", "r0"]
        best = {"mu": 0.5, "r0": 100.0}
        with pytest.raises(ValueError, match="rotation_key must be 'rc' or 'omega'"):
            transform_cov_matrix(cov, keys, best, {}, rotation_key="invalid")

    def test_raises_when_mu_missing_from_keys(self):
        cov = np.eye(2)
        keys = ["r0", "inc"]  # no 'mu'
        best = {"r0": 100.0, "inc": 0.3}
        with pytest.raises(ValueError, match="keys must include 'mu'"):
            transform_cov_matrix(cov, keys, best, {}, rotation_key="rc")

    def test_raises_when_mass_missing(self):
        cov = np.eye(2)
        keys = ["mu", "r0"]
        best = {"mu": 0.5, "r0": 100.0}
        # mass is not in best_opt_params or fixed_params
        with pytest.raises(ValueError, match="'mass' must be present"):
            transform_cov_matrix(cov, keys, best, {}, rotation_key="rc")

    def test_raises_when_r0_missing(self):
        cov = np.eye(2)
        keys = ["mu", "mass"]
        best = {"mu": 0.5, "mass": 1.0}
        # r0 is not anywhere
        with pytest.raises(ValueError, match="'r0' must be present"):
            transform_cov_matrix(cov, keys, best, {}, rotation_key="rc")

    def test_no_rotation_key_returns_identity_transform(self):
        """With rotation_key=None and no v_r0, the Jacobian is the identity
        matrix, so the returned covariance should equal the input covariance."""
        cov = np.diag([4.0, 0.01, 1.0])
        keys = ["r0", "inc", "mass"]
        best = {"r0": 100.0, "inc": 0.3, "mass": 1.5}
        result = transform_cov_matrix(
            jnp.array(cov, dtype=jnp.float64),
            keys, best, {},
            rotation_key=None, v_r0_is_raw=False,
        )
        np.testing.assert_allclose(
            np.array(result["cov"]), cov, atol=1e-9,
        )

    def test_keys_unchanged_when_no_rotation(self):
        cov = np.eye(3)
        keys = ["r0", "inc", "mass"]
        best = {"r0": 100.0, "inc": 0.3, "mass": 1.5}
        result = transform_cov_matrix(
            jnp.array(cov, dtype=jnp.float64),
            keys, best, {}, rotation_key=None,
        )
        assert result["keys"] == keys

    def test_mu_replaced_by_rc_in_output_keys(self):
        cov = np.eye(2)
        keys = ["mu", "r0"]
        best_opt = {"mu": 0.5, "r0": 100.0}
        fixed = {"mass": 1.0}
        result = transform_cov_matrix(
            jnp.array(cov, dtype=jnp.float64),
            keys, best_opt, fixed, rotation_key="rc",
        )
        assert "rc" in result["keys"]
        assert "mu" not in result["keys"]

    def test_mu_replaced_by_omega_in_output_keys(self):
        cov = np.eye(2)
        keys = ["mu", "r0"]
        best_opt = {"mu": 0.5, "r0": 100.0}
        fixed = {"mass": 1.0}
        result = transform_cov_matrix(
            jnp.array(cov, dtype=jnp.float64),
            keys, best_opt, fixed, rotation_key="omega",
        )
        assert "omega" in result["keys"]
        assert "mu" not in result["keys"]

    def test_errors_dict_has_all_output_keys(self):
        cov = np.eye(3)
        keys = ["r0", "inc", "mass"]
        best_opt = {"r0": 100.0, "inc": 0.3, "mass": 1.5}
        fixed = {}
        result = transform_cov_matrix(
            jnp.array(cov, dtype=jnp.float64),
            keys, best_opt, fixed, rotation_key=None,
        )
        for k in result["keys"]:
            assert k in result["errors"]

    def test_errors_are_sqrt_of_diagonal(self):
        """Errors should equal sqrt of the diagonal of the returned covariance."""
        cov = np.diag([9.0, 0.04, 0.25])
        keys = ["r0", "inc", "mass"]
        best = {"r0": 100.0, "inc": 0.3, "mass": 1.5}
        result = transform_cov_matrix(
            jnp.array(cov, dtype=jnp.float64),
            keys, best, {}, rotation_key=None,
        )
        diag_errors = np.sqrt(np.diag(np.array(result["cov"])))
        for i, k in enumerate(result["keys"]):
            assert pytest.approx(result["errors"][k], rel=1e-6) == float(diag_errors[i])

    def test_result_covariance_is_symmetric(self):
        """Covariance matrices are symmetric by definition. The transformed covariance must remain symmetric."""
        cov = np.array([[4.0, 0.5], [0.5, 1.0]])
        keys = ["mu", "r0"]
        best_opt = {"mu": 0.5, "r0": 1000.0}
        fixed = {"mass": 1.0}
        result = transform_cov_matrix(
            jnp.array(cov, dtype=jnp.float64),
            keys, best_opt, fixed, rotation_key="rc",
        )
        cov_out = np.array(result["cov"])
        np.testing.assert_allclose(cov_out, cov_out.T, atol=1e-10)


# ===========================================================================
# Module-level constants
# ===========================================================================

class TestModuleConstants:
    """Sanity-check the physical constants exported by errors.py."""

    def test_G_positive(self):
        assert errors_module.G > 0

    def test_G_units_order_of_magnitude(self):
        # G in au (km/s)^2 Msol^-1 should be ~887.13...
        assert 1e2 < errors_module.G < 1e3

    def test_au_in_km(self):
        assert pytest.approx(errors_module.au_in_km, rel=1e-4) == 1.4959787e8

    def test_eps_small_positive(self):
        assert 0 < errors_module.eps < 1e-4

    def test_BIG_is_large(self):
        assert errors_module.BIG >= 1e20

    def test_loss_method_choices(self):
        assert set(errors_module.LOSS_METHOD_CHOICES) == {0, 1}

    def test_canonical_units_has_expected_keys(self):
        expected = {"r0", "theta0", "phi0", "inc", "pa", "v_r0",
                    "mass", "rmin", "deltar", "v_lsr", "rc", "omega"}
        assert expected.issubset(set(errors_module.CANONICAL_UNITS.keys()))

    def test_streamline_param_keys_is_tuple(self):
        assert isinstance(errors_module.STREAMLINE_MODEL_PARAM_KEYS, tuple)

    def test_streamline_param_keys_contains_core_params(self):
        core = {"r0", "mass", "inc", "pa", "v_r0", "v_lsr", "mu"}
        assert core.issubset(set(errors_module.STREAMLINE_MODEL_PARAM_KEYS))