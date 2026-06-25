"""
Tests for gradient_descent.py

Tests cover the pure utility and logic functions that can be exercised without
running the full Adam optimisation loop or requiring real observational data:
  - Module-level constants and collections (LOSS_METHOD_CHOICES, ANGLE_KEYS,
    DISPLAY_UNITS, STREAMLINE_MODEL_PARAM_KEYS, CANONICAL_UNITS)
  - to_float64: dtype and value preservation
  - softplus / inv_softplus: positivity, monotonicity, round-trip
  - is_numeric_value: type dispatch
  - check_loss_method: valid/invalid method validation
  - trace_fieldnames_for_loss_method: structure and method-specific keys
  - convert_and_strip_bound_units: unit stripping and passthrough
  - clean_model_param_dict: type conversion, theta0 edge nudging, unknown key rejection
  - check_param_types: numeric enforcement and rmin=None special case
  - sanitize_param_partition: overlap detection, rotation-key uniqueness,
    missing-parameter detection, require_nonempty_opt flag
  - standardise_param_bounds: unknown key rejection, None passthrough
  - build_normalisation_spec: missing bounds, invalid bounds, value-out-of-range
  - normalise_opt_params / denormalise_opt_params: round-trip and v_r0 softplus path
  - get_rotation_param_key: correct key identification and missing-key error
  - mu_from_rotation_param / rotation_param_from_mu: round-trips for all three keys
  - with_mu_substituted: mu substitution in opt and fixed, bound replacement
  - format_param: angle conversion, unit suffix, unknown key
  - build_trace_row: structure, component keys for each loss method
  - trace_tree_to_python: scalar conversion, nested structure, None preservation
  - gradient_l2_norm: known values, zero gradient, dtype

Heavy optimisation-loop paths (fit_streamline, evaluate_initial_guess,
forward_model, chi2_loss) are left for integration tests that require a
full physics environment.

Run with:
    pytest test_gradient_descent.py -v
"""

import math
import sys
import types

import jax.numpy as jnp
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so gradient_descent.py can be imported without the full
# package physics (stream_lines_grad, extract_streamline).
# ---------------------------------------------------------------------------

import sting

_slg = types.ModuleType("sting.stream_lines_grad")

def _mu_from_omega(omega, mass, r0):
    G = 6.67430e-11 * (1e-3)**2 * (1.988416e30) / (1.4959787e11) # in au (km/s)^2 * Msol^-1
    au_to_km = 1.4959787e8 #km
    # return float(omega) / math.sqrt(G * float(mass) / float(r0) ** 3)
    rc = (au_to_km**2) * float(r0)**4 * float(omega)**2 / (G * float(mass))
    mu = rc / float(r0)
    return float(mu)

def _omega_from_mu(mu, mass, r0):
    G = 6.67430e-11 * (1e-3)**2 * (1.988416e30) / (1.4959787e11) # in au (km/s)^2 * Msol^-1
    au_to_km = 1.4959787e8 #km
    omega = (1/au_to_km) *math.sqrt(mu * G * float(mass) / float(r0) ** 3)
    return float(omega)

_slg.mu_from_omega = _mu_from_omega
_slg.omega_from_mu = _omega_from_mu
_slg.checked_xyz_stream = None  # not needed for unit tests

_es = types.ModuleType("sting.extract_streamline")
_es.get_distance_metric = None
_es.cartesian_to_polar = None
_es.wrap_to_pi = None
_es.prepare_data = None

sys.modules["sting.stream_lines_grad"] = _slg
sys.modules["sting.extract_streamline"] = _es

import sting.gradient_descent as gd

# Convenient aliases
BIG = gd.BIG
BIG_NEG = gd.BIG_NEG


# ===========================================================================
# Helpers
# ===========================================================================

def _full_opt_params(rotation_key="mu"):
    """Return a minimal complete set of optimisable params (all required keys
    except those in fixed_params below)."""
    base = {
        "r0": 1000.0,
        "theta0": math.radians(30),
        "phi0": math.radians(15),
        rotation_key: 0.3,
        "v_r0": 2.0,
        "mass": 1.0,
        "inc": math.radians(10),
        "pa": math.radians(5),
        "deltar": 50.0,
    }
    return base


def _full_fixed_params():
    """Return minimal fixed params that complement _full_opt_params."""
    return {"rmin": None, "v_lsr": 0.0}


def _make_bounds(opt_params):
    """Return simple (value*0.5, value*2) bounds for every non-v_r0 key."""
    bounds = {}
    for k, v in opt_params.items():
        if k == "v_r0":
            continue
        lo = float(v) * 0.5 if float(v) > 0 else -abs(float(v)) * 2
        hi = float(v) * 2.0 if float(v) > 0 else abs(float(v)) * 0.5
        if lo >= hi:
            lo, hi = -1.0, 1.0
        bounds[k] = (lo, hi)
    return bounds


# ===========================================================================
# Module-level constants
# ===========================================================================

class TestModuleConstants:
    """Sanity-check the constants and collections exported by gradient_descent.py."""

    def test_BIG_is_large(self):
        assert gd.BIG >= 1e20

    def test_BIG_NEG_is_large_negative(self):
        assert gd.BIG_NEG <= -1e20

    def test_VR0_MIN_is_small_positive(self):
        assert 0 < gd.VR0_MIN < 1e-3

    def test_loss_method_choices_contains_0_and_1(self):
        assert set(gd.LOSS_METHOD_CHOICES) == {0, 1}

    def test_angle_keys_are_correct(self):
        assert gd.ANGLE_KEYS == {"theta0", "phi0", "inc", "pa"}

    def test_display_units_has_expected_keys(self):
        expected = {"r0", "v_r0", "mass", "rmin", "deltar", "v_lsr", "rc", "omega"}
        assert expected.issubset(set(gd.DISPLAY_UNITS.keys()))

    def test_streamline_param_keys_is_tuple(self):
        assert isinstance(gd.STREAMLINE_MODEL_PARAM_KEYS, tuple)

    def test_streamline_param_keys_contains_required(self):
        required = {"r0", "theta0", "phi0", "mu", "v_r0", "mass",
                    "inc", "pa", "rmin", "deltar", "v_lsr"}
        assert required.issubset(set(gd.STREAMLINE_MODEL_PARAM_KEYS))

    def test_canonical_units_has_angle_keys(self):
        for k in gd.ANGLE_KEYS:
            assert k in gd.CANONICAL_UNITS

    def test_loss_method_component_keys_has_both_methods(self):
        assert 0 in gd.LOSS_METHOD_COMPONENT_KEYS
        assert 1 in gd.LOSS_METHOD_COMPONENT_KEYS

    def test_method_0_components_are_radecvelprior(self):
        assert set(gd.LOSS_METHOD_COMPONENT_KEYS[0]) == {"chi2_ra", "chi2_dec", "chi2_v", "chi2_prior"}

    def test_method_1_components_are_rthetavelprior(self):
        assert set(gd.LOSS_METHOD_COMPONENT_KEYS[1]) == {"chi2_r", "chi2_theta", "chi2_v", "chi2_prior"}


# ===========================================================================
# to_float64
# ===========================================================================

class TestToFloat64:
    """Unit tests for the to_float64 conversion helper."""

    def test_scalar_becomes_float64(self):
        result = gd.to_float64(1.0)
        assert result.dtype == jnp.float64

    def test_value_preserved(self):
        assert pytest.approx(float(gd.to_float64(3.14)), rel=1e-9) == 3.14

    def test_integer_input_accepted(self):
        result = gd.to_float64(5)
        assert result.dtype == jnp.float64
        assert pytest.approx(float(result)) == 5.0

    def test_array_input_accepted(self):
        result = gd.to_float64(np.array([1.0, 2.0]))
        assert result.dtype == jnp.float64
        assert result.shape == (2,)

    def test_float32_upcast(self):
        result = gd.to_float64(np.float32(1.5))
        assert result.dtype == jnp.float64


# ===========================================================================
# softplus / inv_softplus
# ===========================================================================

class TestSoftplus:
    """Unit tests for the softplus activation."""

    def test_positive_output_for_any_input(self):
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            assert float(gd.softplus(jnp.array(x))) > 0.0

    def test_monotonically_increasing(self):
        xs = jnp.linspace(-3.0, 3.0, 20)
        ys = jnp.array([gd.softplus(x) for x in xs])
        assert np.all(np.diff(np.array(ys)) > 0)

    def test_large_positive_approx_identity(self):
        x = 20.0
        assert pytest.approx(float(gd.softplus(jnp.array(x))), rel=1e-4) == x

    def test_output_dtype_is_float64(self):
        result = gd.softplus(jnp.array(1.0, dtype=jnp.float64))
        assert result.dtype == jnp.float64

    def test_zero_gives_log_two(self):
        assert pytest.approx(float(gd.softplus(jnp.array(0.0))), rel=1e-6) == math.log(2)


class TestInvSoftplus:
    """Unit tests for the inverse softplus."""

    def test_roundtrip_softplus_then_inv(self):
        for y in [0.5, 1.0, 2.0, 5.0]:
            y_arr = jnp.array(y, dtype=jnp.float64)
            x = gd.inv_softplus(y_arr)
            recovered = gd.softplus(x)
            assert pytest.approx(float(recovered), rel=1e-6) == y

    def test_roundtrip_inv_then_softplus(self):
        for x in [-2.0, 0.0, 1.0, 3.0]:
            x_arr = jnp.array(x, dtype=jnp.float64)
            y = gd.softplus(x_arr)
            recovered = gd.inv_softplus(y)
            assert pytest.approx(float(recovered), rel=1e-6) == x

    def test_output_dtype_is_float64(self):
        result = gd.inv_softplus(jnp.array(1.0, dtype=jnp.float64))
        assert result.dtype == jnp.float64


# ===========================================================================
# is_numeric_value
# ===========================================================================

class TestIsNumericValue:
    """Unit tests for the numeric-type guard."""

    def test_int_is_numeric(self):
        assert gd.is_numeric_value(1)

    def test_float_is_numeric(self):
        assert gd.is_numeric_value(3.14)

    def test_numpy_float_is_numeric(self):
        assert gd.is_numeric_value(np.float64(1.0))

    def test_jax_array_is_numeric(self):
        assert gd.is_numeric_value(jnp.array(1.0))

    def test_string_is_not_numeric(self):
        assert not gd.is_numeric_value("hello")

    def test_none_is_not_numeric(self):
        assert not gd.is_numeric_value(None)

    def test_bool_is_not_numeric(self):
        """Booleans are excluded even though bool is a subtype of int."""
        assert not gd.is_numeric_value(True)

    def test_numeric_list_is_numeric(self):
        assert gd.is_numeric_value([1.0, 2.0])


# ===========================================================================
# check_loss_method
# ===========================================================================

class TestCheckLossMethod:
    """Unit tests for the loss method validator."""

    def test_method_0_accepted(self):
        assert gd.check_loss_method(0) == 0

    def test_method_1_accepted(self):
        assert gd.check_loss_method(1) == 1

    def test_method_2_raises(self):
        with pytest.raises(ValueError, match="Unknown loss_method"):
            gd.check_loss_method(2)

    def test_method_minus_1_raises(self):
        with pytest.raises(ValueError, match="Unknown loss_method"):
            gd.check_loss_method(-1)

    def test_string_raises(self):
        with pytest.raises((ValueError, TypeError)):
            gd.check_loss_method("radecvel")

    def test_return_value_is_integer(self):
        result = gd.check_loss_method(0)
        assert isinstance(result, int)


# ===========================================================================
# trace_fieldnames_for_loss_method
# ===========================================================================

class TestTraceFieldnamesForLossMethod:
    """Tests for the trace CSV header builder."""

    def test_returns_list(self):
        assert isinstance(gd.trace_fieldnames_for_loss_method(0), list)

    def test_epoch_and_loss_always_present(self):
        for method in [0, 1]:
            fields = gd.trace_fieldnames_for_loss_method(method)
            assert "epoch" in fields
            assert "loss" in fields

    def test_method_0_has_chi2_ra_dec_v(self):
        fields = gd.trace_fieldnames_for_loss_method(0)
        assert "chi2_ra" in fields
        assert "chi2_dec" in fields
        assert "chi2_v" in fields

    def test_method_1_has_chi2_r_theta_v(self):
        fields = gd.trace_fieldnames_for_loss_method(1)
        assert "chi2_r" in fields
        assert "chi2_theta" in fields
        assert "chi2_v" in fields

    def test_method_0_does_not_have_chi2_r_theta(self):
        fields = gd.trace_fieldnames_for_loss_method(0)
        assert "chi2_r" not in fields
        assert "chi2_theta" not in fields

    def test_method_1_does_not_have_chi2_ra_dec(self):
        fields = gd.trace_fieldnames_for_loss_method(1)
        assert "chi2_ra" not in fields
        assert "chi2_dec" not in fields

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            gd.trace_fieldnames_for_loss_method(99)

    def test_common_trace_fields_present(self):
        """Fields from TRACE_COMMON_FIELDNAMES (minus epoch/loss) should appear."""
        fields = gd.trace_fieldnames_for_loss_method(0)
        for f in ("grad_norm", "model_points_total", "data_points_total"):
            assert f in fields

    def test_chi2_prior_present_in_method_0(self):
        fields = gd.trace_fieldnames_for_loss_method(0)
        assert "chi2_prior" in fields

    def test_chi2_prior_present_in_method_1(self):
        fields = gd.trace_fieldnames_for_loss_method(1)
        assert "chi2_prior" in fields


# ===========================================================================
# convert_and_strip_bound_units
# ===========================================================================

class TestConvertAndStripBoundUnits:
    """Unit tests for bound unit conversion."""

    def test_plain_tuple_passthrough(self):
        bounds = {"r0": (100.0, 500.0)}
        result = gd.convert_and_strip_bound_units(bounds)
        assert result["r0"] == (100.0, 500.0)

    def test_astropy_quantity_stripped(self):
        import astropy.units as u
        bounds = {"r0": np.array([100.0, 500.0]) * u.au}
        result = gd.convert_and_strip_bound_units(bounds)
        lo, hi = result["r0"]
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert pytest.approx(lo) == 100.0
        assert pytest.approx(hi) == 500.0

    def test_unknown_key_with_quantity_raises(self):
        import astropy.units as u
        bounds = {"not_a_real_param": np.array([1.0, 2.0]) * u.au}
        with pytest.raises(ValueError):
            gd.convert_and_strip_bound_units(bounds)

    def test_output_values_are_floats(self):
        bounds = {"mass": (0.5, 2.0)}
        result = gd.convert_and_strip_bound_units(bounds)
        lo, hi = result["mass"]
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_empty_bounds_returns_empty(self):
        result = gd.convert_and_strip_bound_units({})
        assert result == {}

    def test_none_input_returns_empy_dict(self):
        result = gd.convert_and_strip_bound_units(None)
        assert result == {}


# ===========================================================================
# clean_model_param_dict
# ===========================================================================

class TestCleanModelParamDict:
    """Unit tests for the parameter dictionary sanitizer."""

    def test_returns_dict(self):
        result = gd.clean_model_param_dict({"r0": 100.0}, "test")
        assert isinstance(result, dict)

    def test_numeric_value_converted_to_float64(self):
        result = gd.clean_model_param_dict({"r0": 100}, "test")
        assert result["r0"].dtype == jnp.float64

    def test_none_value_preserved(self):
        result = gd.clean_model_param_dict({"rmin": None}, "test")
        assert result["rmin"] is None

    def test_non_dict_raises_type_error(self):
        with pytest.raises(TypeError):
            gd.clean_model_param_dict([1.0, 2.0], "test")

    def test_none_input_treated_as_empty_dict(self):
        result = gd.clean_model_param_dict(None, "test")
        assert result == {}

    def test_unknown_key_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown parameter keys"):
            gd.clean_model_param_dict({"not_a_param": 1.0}, "test")

    def test_theta0_zero_is_nudged(self):
        """theta0=0 should be nudged to a small positive value."""
        result = gd.clean_model_param_dict({"theta0": 0.0}, "test")
        assert float(result["theta0"]) > 0.0

    def test_theta0_pi_is_nudged(self):
        """theta0=pi should be nudged below pi."""
        result = gd.clean_model_param_dict({"theta0": math.pi}, "test")
        assert float(result["theta0"]) < math.pi

    def test_theta0_mid_value_unchanged(self):
        """A non-edge theta0 value should remain unchanged."""
        result = gd.clean_model_param_dict({"theta0": math.radians(45)}, "test")
        assert pytest.approx(float(result["theta0"]), rel=1e-9) == math.radians(45)

    def test_astropy_quantity_stripped(self):
        import astropy.units as u
        result = gd.clean_model_param_dict({"r0": 500.0 * u.au}, "test")
        assert pytest.approx(float(result["r0"]), rel=1e-9) == 500.0

    def test_multiple_valid_keys_accepted(self):
        params = {"r0": 1000.0, "mass": 1.0, "v_r0": 2.0}
        result = gd.clean_model_param_dict(params, "test")
        assert set(result.keys()) == {"r0", "mass", "v_r0"}


# ===========================================================================
# check_param_types
# ===========================================================================

class TestCheckParamTypes:
    """Unit tests for the parameter type checker."""

    def test_valid_numeric_opt_params_pass(self):
        gd.check_param_types({"r0": 100.0, "mass": 1.0}, {})

    def test_string_opt_param_raises_type_error(self):
        with pytest.raises(TypeError, match="must be numeric"):
            gd.check_param_types({"r0": "big"}, {})

    def test_bool_opt_param_raises_type_error(self):
        with pytest.raises(TypeError, match="must be numeric"):
            gd.check_param_types({"r0": True}, {})

    def test_none_opt_rmin_raises_value_error(self):
        """rmin=None is forbidden in opt_params."""
        with pytest.raises(ValueError, match="'rmin' cannot be None"):
            gd.check_param_types({"rmin": None}, {})

    def test_none_fixed_rmin_is_allowed(self):
        """rmin=None is allowed only in fixed_params."""
        gd.check_param_types({}, {"rmin": None})  # should not raise

    def test_string_fixed_param_raises_type_error(self):
        with pytest.raises(TypeError, match="must be numeric"):
            gd.check_param_types({}, {"mass": "one"})

    def test_bool_fixed_param_raises_type_error(self):
        with pytest.raises(TypeError, match="must be numeric"):
            gd.check_param_types({}, {"mass": False})

    def test_valid_fixed_none_non_rmin_raises(self):
        """None for a non-rmin fixed param should raise."""
        with pytest.raises(TypeError):
            gd.check_param_types({}, {"mass": None})


# ===========================================================================
# sanitize_param_partition
# ===========================================================================

class TestSanitizeParamPartition:
    """Unit tests for the opt/fixed partition validator."""

    def _make_full_partition(self, rotation_key="mu"):
        opt = _full_opt_params(rotation_key)
        fixed = _full_fixed_params()
        return opt, fixed

    def test_valid_partition_returns_dicts(self):
        opt, fixed = self._make_full_partition()
        result_opt, result_fixed = gd.sanitize_param_partition(opt, fixed)
        assert isinstance(result_opt, dict)
        assert isinstance(result_fixed, dict)

    def test_overlap_raises_key_error(self):
        opt, fixed = self._make_full_partition()
        fixed["r0"] = 999.0  # duplicate key
        with pytest.raises(KeyError, match="Overlap"):
            gd.sanitize_param_partition(opt, fixed)

    def test_missing_required_key_raises(self):
        opt, fixed = self._make_full_partition()
        del opt["mass"]
        with pytest.raises(KeyError, match="Missing required"):
            gd.sanitize_param_partition(opt, fixed)

    def test_no_rotation_key_raises(self):
        opt, fixed = self._make_full_partition()
        del opt["mu"]
        with pytest.raises(KeyError, match="Exactly one of"):
            gd.sanitize_param_partition(opt, fixed)

    def test_two_rotation_keys_raises(self):
        opt, fixed = self._make_full_partition()
        opt["rc"] = 300.0  # mu already present
        with pytest.raises(KeyError, match="Exactly one of"):
            gd.sanitize_param_partition(opt, fixed)

    def test_rc_rotation_key_accepted(self):
        opt, fixed = self._make_full_partition(rotation_key="rc")
        result_opt, _ = gd.sanitize_param_partition(opt, fixed)
        assert "rc" in result_opt

    def test_omega_rotation_key_accepted(self):
        opt, fixed = self._make_full_partition(rotation_key="omega")
        result_opt, _ = gd.sanitize_param_partition(opt, fixed)
        assert "omega" in result_opt

    def test_require_nonempty_opt_raises_when_empty(self):
        """Moving all params to fixed, nothing in opt, should raise when flag is True."""
        opt, fixed = self._make_full_partition()
        # move everything into fixed
        full_fixed = {**fixed, **opt}
        opt_empty = {}
        with pytest.raises(ValueError, match="at least one"):
            gd.sanitize_param_partition(opt_empty, full_fixed, require_nonempty_opt=True)

    def test_require_nonempty_opt_false_allows_empty_opt(self):
        """Moving all params to fixed should be valid when flag is False."""
        opt, fixed = self._make_full_partition()
        # move everything into fixed
        full_fixed = {**fixed, **opt}
        opt_empty = {}
        result_opt, _ = gd.sanitize_param_partition(opt_empty, full_fixed, require_nonempty_opt=False)
        assert result_opt == {}

    def test_non_numeric_opt_param_raises_error(self):
        opt, fixed = self._make_full_partition()
        opt["r0"] = "not_a_number"
        with pytest.raises((ValueError, TypeError)):
            gd.sanitize_param_partition(opt, fixed)

    def test_unknown_key_raises_key_error(self):
        opt, fixed = self._make_full_partition()
        opt["unknown_param"] = 1.0
        with pytest.raises(KeyError):
            gd.sanitize_param_partition(opt, fixed)


# ===========================================================================
# standardise_param_bounds
# ===========================================================================

class TestStandardiseParamBounds:
    """Unit tests for the bounds standardiser."""

    def test_none_returns_none(self):
        assert gd.standardise_param_bounds(None) is None

    def test_valid_bounds_returned_unchanged(self):
        bounds = {"r0": (100.0, 500.0), "mass": (0.5, 2.0)}
        result = gd.standardise_param_bounds(bounds)
        assert result["r0"] == (100.0, 500.0)

    def test_unknown_key_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown params"):
            gd.standardise_param_bounds({"not_a_param": (0, 1)})

    def test_returns_dict(self):
        result = gd.standardise_param_bounds({"r0": (100.0, 500.0)})
        assert isinstance(result, dict)

    def test_empty_bounds_accepted(self):
        result = gd.standardise_param_bounds({})
        assert result == {}


# ===========================================================================
# build_normalisation_spec
# ===========================================================================

class TestBuildNormalisationSpec:
    """Unit tests for the normalisation spec builder."""

    def test_none_param_bounds_raises(self):
        with pytest.raises(ValueError, match="param_bounds is required"):
            gd.build_normalisation_spec({"r0": 100.0}, None)

    def test_missing_bound_for_opt_param_raises(self):
        with pytest.raises(ValueError, match="Missing bounds"):
            gd.build_normalisation_spec({"r0": 100.0}, {})

    def test_invalid_bounds_tuple_raises(self):
        with pytest.raises(ValueError, match="must be a 2-element"):
            gd.build_normalisation_spec({"r0": 100.0}, {"r0": (100.0,)})

    def test_non_finite_bounds_raises(self):
        with pytest.raises(ValueError, match="finite"):
            gd.build_normalisation_spec({"r0": 100.0}, {"r0": (0.0, float("inf"))})

    def test_min_not_less_than_max_raises(self):
        with pytest.raises(ValueError, match="min < max"):
            gd.build_normalisation_spec({"r0": 100.0}, {"r0": (500.0, 100.0)})

    def test_value_outside_bounds_raises(self):
        with pytest.raises(ValueError, match="outside bounds"):
            gd.build_normalisation_spec({"r0": 1000.0}, {"r0": (100.0, 500.0)})

    def test_valid_spec_has_offset_and_scale(self):
        spec = gd.build_normalisation_spec({"r0": 300.0}, {"r0": (100.0, 500.0)})
        assert "offset" in spec["r0"]
        assert "scale" in spec["r0"]

    def test_scale_equals_hi_minus_lo(self):
        spec = gd.build_normalisation_spec({"r0": 300.0}, {"r0": (100.0, 500.0)})
        assert pytest.approx(float(spec["r0"]["scale"])) == 400.0

    def test_offset_equals_lower_bound(self):
        spec = gd.build_normalisation_spec({"r0": 300.0}, {"r0": (100.0, 500.0)})
        assert pytest.approx(float(spec["r0"]["offset"])) == 100.0

    def test_v_r0_skipped_even_without_bounds(self):
        """v_r0 uses softplus instead of normalisation, so no bounds are needed."""
        spec = gd.build_normalisation_spec(
            {"r0": 300.0, "v_r0": 2.0},
            {"r0": (100.0, 500.0)},  # no v_r0 bounds
        )
        assert "v_r0" not in spec
        assert "r0" in spec

    def test_multiple_params_all_in_spec(self):
        opt = {"r0": 300.0, "mass": 1.0}
        bounds = {"r0": (100.0, 500.0), "mass": (0.5, 2.0)}
        spec = gd.build_normalisation_spec(opt, bounds)
        assert "r0" in spec
        assert "mass" in spec


# ===========================================================================
# normalise_opt_params / denormalise_opt_params
# ===========================================================================

class TestNormaliseDenormaliseRoundTrip:
    """Unit tests for parameter normalisation and its inverse."""

    def _make_spec_and_params(self):
        opt = {"r0": 300.0, "mass": 1.0}
        bounds = {"r0": (100.0, 500.0), "mass": (0.5, 2.0)}
        spec = gd.build_normalisation_spec(opt, bounds)
        return opt, spec

    def test_normalised_values_in_zero_one(self):
        opt, spec = self._make_spec_and_params()
        norm = gd.normalise_opt_params(opt, spec)
        for k, v in norm.items():
            assert 0.0 <= float(v) <= 1.0, f"{k} not in [0,1]: {v}"

    def test_roundtrip_norm_denorm(self):
        opt, spec = self._make_spec_and_params()
        norm = gd.normalise_opt_params(opt, spec)
        recovered = gd.denormalise_opt_params(norm, spec)
        for k in opt:
            assert pytest.approx(float(recovered[k]), rel=1e-9) == float(opt[k])

    def test_v_r0_normalised_via_softplus(self):
        """v_r0 is stored as inv_softplus(v_r0) in normalised space."""
        opt = {"v_r0": 2.0}
        spec = {}  # no spec needed for v_r0
        norm = gd.normalise_opt_params(opt, spec)
        expected_raw = float(gd.inv_softplus(gd.to_float64(2.0)))
        assert pytest.approx(float(norm["v_r0"]), rel=1e-6) == expected_raw

    def test_v_r0_denormalised_via_softplus(self):
        """Denormalising v_r0 applies softplus."""
        raw = gd.inv_softplus(gd.to_float64(3.0))
        norm = {"v_r0": raw}
        recovered = gd.denormalise_opt_params(norm, {})
        assert pytest.approx(float(recovered["v_r0"]), rel=1e-6) == 3.0

    def test_negative_v_r0_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            gd.normalise_opt_params({"v_r0": -1.0}, {})

    def test_phi0_wraps_modulo_2pi(self):
        """phi0 uses modular arithmetic; check the result is in [0, 2pi)."""
        opt = {"phi0": math.radians(45)}
        spec = gd.build_normalisation_spec(opt, {"phi0": (0.0, 2 * math.pi)})
        norm = gd.normalise_opt_params(opt, spec)
        denorm = gd.denormalise_opt_params(norm, spec)
        result = float(denorm["phi0"])
        assert 0.0 <= result < 2 * math.pi + 1e-9


# ===========================================================================
# get_rotation_param_key
# ===========================================================================

class TestGetRotationParamKey:
    """Unit tests for the rotation-key finder."""

    def test_finds_mu(self):
        assert gd.get_rotation_param_key({"mu": 0.3}, {}) == "mu"

    def test_finds_rc_in_fixed(self):
        assert gd.get_rotation_param_key({}, {"rc": 300.0}) == "rc"

    def test_finds_omega_in_opt(self):
        assert gd.get_rotation_param_key({"omega": 1e-14}, {}) == "omega"

    def test_missing_all_raises_key_error(self):
        with pytest.raises(KeyError, match="None of the parameters"):
            gd.get_rotation_param_key({"r0": 1000.0}, {"mass": 1.0})


# ===========================================================================
# mu_from_rotation_param / rotation_param_from_mu
# ===========================================================================

class TestMuRotationParamConversions:
    """Unit tests for the mu <-> rotation parameter conversions."""

    def test_mu_key_returns_mu_unchanged(self):
        result = gd.mu_from_rotation_param("mu", 0.3, mass=1.0, r0=1000.0)
        assert pytest.approx(result, rel=1e-9) == 0.3

    def test_rc_key_computes_mu_as_rc_over_r0(self):
        result = gd.mu_from_rotation_param("rc", 300.0, mass=1.0, r0=1000.0)
        assert pytest.approx(result, rel=1e-9) == 0.3

    def test_omega_key_round_trips_with_omega_from_mu(self):
        mu_in = 0.3
        omega = _slg.omega_from_mu(mu=mu_in, mass=1.0, r0=1000.0)
        mu_out = gd.mu_from_rotation_param("omega", omega, mass=1.0, r0=1000.0)
        assert mu_out == pytest.approx(mu_in, rel=1e-5)

    def test_rotation_param_from_mu_mu_key(self):
        result = gd.rotation_param_from_mu("mu", 0.4, mass=1.0, r0=1000.0)
        assert 0.4 == pytest.approx(result, rel=1e-9)

    def test_rotation_param_from_mu_rc_key(self):
        result = gd.rotation_param_from_mu("rc", 0.3, mass=1.0, r0=1000.0)
        assert 300.0 == pytest.approx(result, rel=1e-9)  # 0.3 * 1000

    def test_rotation_param_from_mu_omega_key(self):
        mu = 0.3
        expected_omega = _slg.omega_from_mu(mu=mu, mass=1.0, r0=1000.0)
        result = gd.rotation_param_from_mu("omega", mu, mass=1.0, r0=1000.0)
        assert expected_omega == pytest.approx(result, rel=1e-5)

    def test_roundtrip_mu_from_rc_then_back(self):
        rc_in = 350.0
        r0 = 1000.0
        mu = gd.mu_from_rotation_param("rc", rc_in, mass=1.0, r0=r0)
        rc_out = gd.rotation_param_from_mu("rc", mu, mass=1.0, r0=r0)
        assert pytest.approx(rc_out, rel=1e-9) == rc_in


# ===========================================================================
# with_mu_substituted
# ===========================================================================

class TestWithMuSubstituted:
    """Unit tests for the rc/omega → mu substitution."""

    def _base_params(self, rotation_key):
        opt = _full_opt_params(rotation_key)
        fixed = _full_fixed_params()
        return opt, fixed

    def test_rc_in_opt_becomes_mu_in_opt(self):
        opt, fixed = self._base_params("rc")
        result_opt, result_fixed, _, _ = gd.with_mu_substituted(opt, fixed)
        assert "mu" in result_opt
        assert "rc" not in result_opt

    def test_omega_in_opt_becomes_mu_in_opt(self):
        opt, fixed = self._base_params("omega")
        result_opt, result_fixed, _, _ = gd.with_mu_substituted(opt, fixed)
        assert "mu" in result_opt
        assert "omega" not in result_opt

    def test_mu_in_opt_stays_mu(self):
        opt, fixed = self._base_params("mu")
        result_opt, _, _, rotation_key = gd.with_mu_substituted(opt, fixed)
        assert "mu" in result_opt
        assert rotation_key == "mu"

    def test_rotation_key_returned_correctly_for_rc(self):
        opt, fixed = self._base_params("rc")
        _, _, _, rotation_key = gd.with_mu_substituted(opt, fixed)
        assert rotation_key == "rc"

    def test_rotation_key_returned_correctly_for_omega(self):
        opt, fixed = self._base_params("omega")
        _, _, _, rotation_key = gd.with_mu_substituted(opt, fixed)
        assert rotation_key == "omega"

    def test_rc_in_fixed_becomes_mu_in_fixed(self):
        opt = _full_opt_params("mu")
        del opt["mu"]
        fixed = _full_fixed_params()
        fixed["rc"] = 300.0
        result_opt, result_fixed, _, _ = gd.with_mu_substituted(opt, fixed)
        assert "mu" in result_fixed
        assert "rc" not in result_fixed

    def test_mu_bounds_added_when_rc_in_opt(self):
        opt, fixed = self._base_params("rc")
        bounds = {"rc": (100.0, 500.0)}
        _, _, result_bounds, _ = gd.with_mu_substituted(opt, fixed, param_bounds=bounds)
        assert "mu" in result_bounds
        assert "rc" not in result_bounds

    def test_mu_value_correct_for_rc_substitution(self):
        opt, fixed = self._base_params("rc")
        rc_val = opt["rc"]
        r0_val = opt["r0"]
        result_opt, _, _, _ = gd.with_mu_substituted(opt, fixed)
        expected_mu = rc_val / r0_val
        assert pytest.approx(float(result_opt["mu"]), rel=1e-9) == expected_mu


# ==========================================================================
# auto_fill_angle_bounds
# =========================================================================

class TestAutoFillAngleBounds:
    """Units tests for the autoatic theta0/phi0 bound filler."""

    def test_theta0_bounds_added_when_optimised(self):
        opt_params = {"theta0": math.radians(30), "r0": 1000.0}
        result = gd.auto_fill_angle_bounds(opt_params, {"r0": (500.0, 2000.0)})
        assert result["theta0"] == (0.0, math.pi)
    
    def test_phi0_bounds_added_when_optimised(self):
        opt_params = {"phi0": math.radians(200), "r0": 1000.0}
        result = gd.auto_fill_angle_bounds(opt_params, {"r0": (500.0, 2000.0)})
        assert result["phi0"] == (0.0, 2 * math.pi)

    def test_not_added_when_fixed(self):
        opt_params = {"r0": 1000.0}
        result = gd.auto_fill_angle_bounds(opt_params, {"r0": (500.0, 2000.0)})
        assert "theta0" not in result
        assert "phi0" not in result

    def test_other_bounds_left_untouched(self):
        opt_params = {"theta0": math.radians(30), "r0": 1000.0}
        result = gd.auto_fill_angle_bounds(opt_params, {"r0": (500.0, 2000.0)})
        assert result["r0"] == (500.0, 2000.0)
    
    def test_none_param_bounds_treated_as_empty(self):
        opt_params = {"theta0": math.radians(30), "phi0": math.radians(200)}
        result = gd.auto_fill_angle_bounds(opt_params, None)
        assert result == {"theta0": (0.0, math.pi), "phi0": (0.0, 2 * math.pi)}

    def test_user_supplied_theta0_bounds_overriden(self):
        opt_params = {"theta0": math.radians(30)}
        user_bounds = {"theta0": (math.radians(10), math.radians(80))}
        result = gd.auto_fill_angle_bounds(opt_params, user_bounds)
        assert result["theta0"] == (0.0, math.pi)  # overridden

    def test_user_supplied_phi0_bounds_overriden(self):
        opt_params = {"phi0": math.radians(200)}
        user_bounds = {"phi0": (math.radians(100), math.radians(300))}
        result = gd.auto_fill_angle_bounds(opt_params, user_bounds)
        assert result["phi0"] == (0.0, 2 * math.pi)  # overridden

    def test_does_not_mutate_input_param_bounds(self):
        """ should return a new dict rather than mutating the user's bounds dict"""
        opt_params = {"theta0": math.radians(30), "phi0": math.radians(200)}
        user_bounds = {"r0": (500.0, 2000.0)}
        result = gd.auto_fill_angle_bounds(opt_params, user_bounds)
        assert "theta0" not in user_bounds
        assert "phi0" not in user_bounds
        assert result is not user_bounds

    def test_accepts_set_of_keys(self):
        """since the function only checks membership, a set of keys should work as well as a dict"""
        opt_keys = {"theta0", "phi0", "r0"}
        result = gd.auto_fill_angle_bounds(opt_keys, None)
        assert result["theta0"] == (0.0, math.pi)
        assert result["phi0"] == (0.0, 2 * math.pi)

    def test_bounds_in_radians(self):
        """The bounds returned should be plain floats in radians, not degrees."""
        opt_params = {"theta0": math.radians(30), "phi0": math.radians(200)}
        result = gd.auto_fill_angle_bounds(opt_params, None)
        for key in ("theta0", "phi0"):
            lo, hi = result[key]
            assert isinstance(lo, float)
            assert isinstance(hi, float)


# ===========================================================================
# validate_priors
# ===========================================================================

class TestValidatePriors:
    """Unit tests for the prior validation helper."""

    def _opt(self):
        return {"mass": 1.0, "inc": 0.3, "pa": 0.1, "r0": 1000.0}

    def _fixed(self):
        return {"v_lsr": 0.0}

    def test_none_returns_empty_dict(self):
        result = gd.validate_priors(None, self._opt(), self._fixed())
        assert result == {}

    def test_empty_dict_returns_empty_dict(self):
        result = gd.validate_priors({}, self._opt(), self._fixed())
        assert result == {}

    def test_valid_single_prior_accepted(self):
        result = gd.validate_priors({"mass": (1.0, 0.1)}, self._opt(), self._fixed())
        assert "mass" in result
        assert pytest.approx(result["mass"][0]) == 1.0
        assert pytest.approx(result["mass"][1]) == 0.1

    def test_valid_multiple_priors_accepted(self):
        result = gd.validate_priors(
            {"mass": (1.0, 0.1), "inc": (0.3, 0.05)},
            self._opt(), self._fixed(),
        )
        assert set(result.keys()) == {"mass", "inc"}

    def test_non_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="dict"):
            gd.validate_priors([(1.0, 0.1)], self._opt(), self._fixed())

    def test_unknown_key_raises_key_error(self):
        with pytest.raises(KeyError):
            gd.validate_priors({"not_a_param": (1.0, 0.1)}, self._opt(), self._fixed())

    def test_fixed_param_raises_value_error(self):
        with pytest.raises(ValueError, match="fixed"):
            gd.validate_priors({"v_lsr": (0.0, 0.1)}, self._opt(), self._fixed())

    def test_param_not_in_opt_raises_value_error(self):
        """A prior for a param that is neither opt nor fixed (just absent) should raise."""
        with pytest.raises(ValueError):
            gd.validate_priors({"v_lsr": (0.0, 0.1)}, self._opt(), self._fixed())

    def test_wrong_tuple_length_raises_value_error(self):
        with pytest.raises(ValueError, match="2-element"):
            gd.validate_priors({"mass": (1.0,)}, self._opt(), self._fixed())

    def test_three_element_tuple_raises_value_error(self):
        with pytest.raises(ValueError, match="2-element"):
            gd.validate_priors({"mass": (1.0, 0.1, 0.0)}, self._opt(), self._fixed())

    def test_zero_sigma_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            gd.validate_priors({"mass": (1.0, 0.0)}, self._opt(), self._fixed())

    def test_negative_sigma_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            gd.validate_priors({"mass": (1.0, -0.5)}, self._opt(), self._fixed())

    def test_astropy_quantity_stripped(self):
        import astropy.units as u
        result = gd.validate_priors(
            {"mass": (1.0 * u.Msun, 0.1 * u.Msun)},
            self._opt(), self._fixed(),
        )
        mean, sigma = result["mass"]
        assert isinstance(mean, float)
        assert isinstance(sigma, float)
        assert pytest.approx(mean) == 1.0
        assert pytest.approx(sigma) == 0.1

    def test_astropy_quantity_unit_conversion(self):
        """A prior given in non-canonical units should be converted correctly."""
        import astropy.units as u
        # r0 canonical unit is au; supply in au directly for simplicity
        opt = {"r0": 500.0, "mass": 1.0, "inc": 0.3, "pa": 0.1}
        result = gd.validate_priors({"r0": (500.0 * u.au, 50.0 * u.au)}, opt, self._fixed())
        assert pytest.approx(result["r0"][0]) == 500.0
        assert pytest.approx(result["r0"][1]) == 50.0

    def test_returns_float_values(self):
        result = gd.validate_priors({"mass": (1.0, 0.1)}, self._opt(), self._fixed())
        mean, sigma = result["mass"]
        assert isinstance(mean, float)
        assert isinstance(sigma, float)

    def test_param_in_opt_but_not_present_in_priors_is_not_validated(self):
        """Parameters without a prior are simply not in the returned dict."""
        result = gd.validate_priors({"mass": (1.0, 0.1)}, self._opt(), self._fixed())
        assert "inc" not in result
        assert "r0" not in result


# ===========================================================================
# compute_prior_penalty
# ===========================================================================

class TestComputePriorPenalty:
    """Unit tests for the Gaussian prior penalty computation.

    compute_prior_penalty is decorated with @jax.jit and uses priors_keys
    (a tuple of strings) as a static argument via zip.  We call it inside
    jax.disable_jit() so we can pass plain Python strings without needing to
    mark them static at the call site.
    """

    def _call(self, params, means, sigmas, keys):
        import jax
        with jax.disable_jit():
            return gd.compute_prior_penalty(params, means, sigmas, keys)

    def test_empty_priors_returns_zero(self):
        params = {"mass": 1.0, "inc": 0.3}
        result = float(self._call(params, (), (), ()))
        assert pytest.approx(result) == 0.0

    def test_single_prior_at_mean_gives_zero(self):
        """When the parameter equals its prior mean the penalty is zero."""
        params = {"mass": jnp.asarray(1.0, dtype=jnp.float64)}
        result = float(self._call(params, (1.0,), (0.1,), ("mass",)))
        assert pytest.approx(result, abs=1e-12) == 0.0

    def test_single_prior_one_sigma_away_gives_one(self):
        """((mean+sigma - mean)/sigma)^2 = 1."""
        params = {"mass": jnp.asarray(1.1, dtype=jnp.float64)}
        result = float(self._call(params, (1.0,), (0.1,), ("mass",)))
        assert pytest.approx(result, rel=1e-9) == 1.0

    def test_single_prior_two_sigma_away_gives_four(self):
        params = {"mass": jnp.asarray(1.2, dtype=jnp.float64)}
        result = float(self._call(params, (1.0,), (0.1,), ("mass",)))
        assert pytest.approx(result, rel=1e-9) == 4.0

    def test_multiple_priors_sum_correctly(self):
        """Two independent 1-sigma deviations should give penalty = 2."""
        params = {
            "mass": jnp.asarray(1.1, dtype=jnp.float64),
            "inc": jnp.asarray(0.35, dtype=jnp.float64),
        }
        result = float(self._call(
            params,
            (1.0, 0.3),     # means
            (0.1, 0.05),    # sigmas
            ("mass", "inc"),
        ))
        # mass: ((1.1-1.0)/0.1)^2 = 1.0
        # inc:  ((0.35-0.3)/0.05)^2 = 1.0
        assert pytest.approx(result, rel=1e-9) == 2.0

    def test_negative_deviation_gives_positive_penalty(self):
        """Penalty is squared so negative deviation still gives positive result."""
        params = {"mass": jnp.asarray(0.9, dtype=jnp.float64)}
        result = float(self._call(params, (1.0,), (0.1,), ("mass",)))
        assert pytest.approx(result, rel=1e-9) == 1.0

    def test_output_is_non_negative(self):
        params = {"mass": jnp.asarray(0.5, dtype=jnp.float64)}
        result = float(self._call(params, (1.0,), (0.2,), ("mass",)))
        assert result >= 0.0

    def test_larger_sigma_gives_smaller_penalty(self):
        """The same deviation with a larger sigma should give a smaller penalty."""
        params = {"mass": jnp.asarray(1.2, dtype=jnp.float64)}
        penalty_tight = float(self._call(params, (1.0,), (0.1,), ("mass",)))
        penalty_loose = float(self._call(params, (1.0,), (1.0,), ("mass",)))
        assert penalty_tight > penalty_loose

    def test_output_dtype_float64(self):
        params = {"mass": jnp.asarray(1.0, dtype=jnp.float64)}
        result = self._call(params, (1.0,), (0.1,), ("mass",))
        assert result.dtype == jnp.float64


# ===========================================================================
# format_param
# ===========================================================================

class TestFormatParam:
    """Unit tests for the parameter display formatter."""

    def test_angle_key_converted_to_degrees(self):
        for key in ("theta0", "phi0", "inc", "pa"):
            result = gd.format_param(key, math.pi)
            assert "deg" in result
            assert "180" in result

    def test_non_angle_key_has_unit_suffix(self):
        result = gd.format_param("r0", 100.0)
        assert "au" in result

    def test_v_r0_has_km_s(self):
        result = gd.format_param("v_r0", 2.0)
        assert "km/s" in result

    def test_mu_has_no_unit(self):
        result = gd.format_param("mu", 0.3)
        assert "au" not in result
        assert "km" not in result

    def test_omega_has_1_over_s(self):
        result = gd.format_param("omega", 1e-14)
        assert "1/s" in result

    def test_unknown_key_returns_string(self):
        result = gd.format_param("unknown", 42.0)
        assert "42" in result

    def test_returns_string(self):
        assert isinstance(gd.format_param("r0", 100.0), str)

    def test_zero_angle_gives_zero_degrees(self):
        result = gd.format_param("inc", 0.0)
        assert "0" in result


# ===========================================================================
# build_trace_row
# ===========================================================================

class TestBuildTraceRow:
    """Unit tests for the trace row builder."""

    def _make_loss_trace(self, method=0):
        if method == 0:
            chi2_components = {
                "chi2_ra": 1.0, "chi2_dec": 2.0, "chi2_v": 3.0, "chi2_prior": 0.5,
                "chi2_total": 6.5, "overlap_width": 0.5,
            }
        else:
            chi2_components = {
                "chi2_r": 1.0, "chi2_theta": 2.0, "chi2_v": 3.0, "chi2_prior": 0.5,
                "chi2_total": 6.5, "overlap_width": 0.5,
            }
        matching = {
            "model_points_total": 100,
            "model_nan_count": 5,
            "model_valid_points": 95,
            "model_metric_span": 4.5,
            "data_points_total": 10,
            "data_valid_points": 10,
            "data_retained_count": 8,
            "model_retained_count": 90,
            "overlap_metric_min": 0.1,
            "overlap_metric_max": 4.6,
            "distance_metric_model": {"inner_count": 80},
            "distance_metric_data": {"inner_count": 8},
        }
        return {"chi2_components": chi2_components, "matching": matching}

    def test_returns_dict(self):
        trace = self._make_loss_trace()
        row = gd.build_trace_row(0, 6.0, trace, 0.01, loss_method=0)
        assert isinstance(row, dict)

    def test_epoch_and_loss_present(self):
        trace = self._make_loss_trace()
        row = gd.build_trace_row(5, 6.0, trace, 0.01, loss_method=0)
        assert row["epoch"] == 5
        assert pytest.approx(row["loss"]) == 6.0

    def test_method_0_component_keys_present(self):
        trace = self._make_loss_trace(method=0)
        row = gd.build_trace_row(0, 6.0, trace, 0.01, loss_method=0)
        assert "chi2_ra" in row
        assert "chi2_dec" in row
        assert "chi2_v" in row

    def test_method_1_component_keys_present(self):
        trace = self._make_loss_trace(method=1)
        row = gd.build_trace_row(0, 6.0, trace, 0.01, loss_method=1)
        assert "chi2_r" in row
        assert "chi2_theta" in row
        assert "chi2_v" in row

    def test_grad_norm_in_row(self):
        trace = self._make_loss_trace()
        row = gd.build_trace_row(0, 6.0, trace, 0.123, loss_method=0)
        assert pytest.approx(row["grad_norm"]) == 0.123

    def test_model_points_total_in_row(self):
        trace = self._make_loss_trace()
        row = gd.build_trace_row(0, 6.0, trace, 0.01, loss_method=0)
        assert row["model_points_total"] == 100

    def test_invalid_loss_method_raises(self):
        trace = self._make_loss_trace()
        with pytest.raises(ValueError):
            gd.build_trace_row(0, 1.0, trace, 0.0, loss_method=99)

    def test_chi2_prior_in_method_0_row(self):
        trace = self._make_loss_trace(method=0)
        row = gd.build_trace_row(0, 6.5, trace, 0.01, loss_method=0)
        assert "chi2_prior" in row
        assert pytest.approx(row["chi2_prior"]) == 0.5

    def test_chi2_prior_in_method_1_row(self):
        trace = self._make_loss_trace(method=1)
        row = gd.build_trace_row(0, 6.5, trace, 0.01, loss_method=1)
        assert "chi2_prior" in row
        assert pytest.approx(row["chi2_prior"]) == 0.5

    def test_chi2_prior_zero_when_not_in_trace(self):
        """If chi2_prior is absent from the trace components it should default to NaN."""
        trace = self._make_loss_trace(method=0)
        del trace["chi2_components"]["chi2_prior"]
        row = gd.build_trace_row(0, 6.0, trace, 0.01, loss_method=0)
        # build_trace_row uses .get(..., nan) so the key is still present
        assert "chi2_prior" in row
        assert math.isnan(row["chi2_prior"])


# ===========================================================================
# trace_tree_to_python
# ===========================================================================

class TestTraceTreeToPython:
    """Unit tests for the JAX → Python scalar converter."""

    def test_scalar_jax_array_becomes_python_scalar(self):
        arr = jnp.array(3.14, dtype=jnp.float64)
        result = gd.trace_tree_to_python(arr)
        assert isinstance(result, float)
        assert pytest.approx(result) == 3.14

    def test_dict_values_converted(self):
        d = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
        result = gd.trace_tree_to_python(d)
        assert isinstance(result["a"], float)
        assert isinstance(result["b"], float)

    def test_nested_dict_converted(self):
        d = {"outer": {"inner": jnp.array(42.0)}}
        result = gd.trace_tree_to_python(d)
        assert isinstance(result["outer"]["inner"], float)

    def test_none_preserved(self):
        assert gd.trace_tree_to_python(None) is None

    def test_none_in_dict_preserved(self):
        result = gd.trace_tree_to_python({"key": None})
        assert result["key"] is None

    def test_list_converted(self):
        lst = [jnp.array(1.0), jnp.array(2.0)]
        result = gd.trace_tree_to_python(lst)
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_tuple_converted(self):
        t = (jnp.array(1.0), jnp.array(2.0))
        result = gd.trace_tree_to_python(t)
        assert isinstance(result[0], float)

    def test_multidim_array_not_scalar_stays_as_array(self):
        """Arrays with ndim > 0 should be left as-is."""
        arr = jnp.array([1.0, 2.0, 3.0])
        result = gd.trace_tree_to_python(arr)
        # Should not be a plain Python float
        assert not isinstance(result, float)

    def test_plain_python_float_passthrough(self):
        result = gd.trace_tree_to_python(2.71)
        assert pytest.approx(result) == 2.71

    def test_plain_string_passthrough(self):
        result = gd.trace_tree_to_python("hello")
        assert result == "hello"


# ===========================================================================
# gradient_l2_norm
# ===========================================================================

class TestGradientL2Norm:
    """Unit tests for the gradient L2 norm helper."""

    def test_single_leaf_known_value(self):
        """For a single array [3, 4], the L2 norm is 5."""
        grad = {"r0": jnp.array([3.0, 4.0], dtype=jnp.float64)}
        result = float(gd.gradient_l2_norm(grad))
        assert pytest.approx(result, rel=1e-9) == 5.0

    def test_zero_gradient_gives_zero(self):
        grad = {"r0": jnp.zeros(5, dtype=jnp.float64)}
        result = float(gd.gradient_l2_norm(grad))
        assert pytest.approx(result, abs=1e-12) == 0.0

    def test_multiple_leaves_combined(self):
        """||[3,0]|| + ||[0,4]|| as separate leaves should give norm 5."""
        grad = {
            "r0": jnp.array([3.0, 0.0], dtype=jnp.float64),
            "mass": jnp.array([0.0, 4.0], dtype=jnp.float64),
        }
        result = float(gd.gradient_l2_norm(grad))
        assert pytest.approx(result, rel=1e-9) == 5.0

    def test_output_is_non_negative(self):
        grad = {"r0": jnp.array([-1.0, -2.0, -3.0], dtype=jnp.float64)}
        assert float(gd.gradient_l2_norm(grad)) >= 0.0

    def test_output_dtype_is_float64(self):
        grad = {"r0": jnp.array([1.0], dtype=jnp.float64)}
        result = gd.gradient_l2_norm(grad)
        assert result.dtype == jnp.float64

    def test_scalar_leaf(self):
        grad = {"r0": jnp.array(5.0, dtype=jnp.float64)}
        result = float(gd.gradient_l2_norm(grad))
        assert pytest.approx(result, rel=1e-9) == 5.0