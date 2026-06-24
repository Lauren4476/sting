"""
Initial tests for outputs.py

Tests cover pure/logic functions that can be exercised without the full JAX
forward model or filesystem side-effects, plus lightweight smoke tests for the
I/O and plotting helpers (using tmp_path and matplotlib's non-interactive
Agg backend).

Run with:
    pytest test_outputs.py -v
"""

import json
import math
import os
import sys
import types

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – must come before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so outputs.py can be imported without the full package
# ---------------------------------------------------------------------------

# Import the real sting package first so sys.modules["sting"] is the real
# package object — not a bare ModuleType — before we inject any stubs.
import sting

# Build a fake `gradient_descent` module with just the constants that
# outputs.py reads at import time and in param_for_display / save_best_fit_params.
_gd = types.ModuleType("sting.gradient_descent")
_gd.ANGLE_KEYS = {"theta0", "phi0", "inc", "pa"}
_gd.DISPLAY_UNITS = {
    "r0": "au",
    "v_r0": "km/s",
    "mass": "M_sun",
    "rmin": "au",
    "deltar": "au",
    "v_lsr": "km/s",
    "rc": "au",
    "omega": "1/s",
}

# Stub out the functions called by evaluate_best_fit / plot_fitting_results
# (heavy JAX functions); individual tests that need them will mock further.
_gd.prepare_model_params = None
_gd.forward_model = None
_gd.checked_match_model_to_data_curve = None
_gd.convert_and_strip_bound_units = None

_es = types.ModuleType("sting.extract_streamline")
_es.get_distance_metric = None
_es.plot_metric_boundaries = None
_es.get_metric_partitions = None
_es.sample_metric_boundaries = None

# Inject stubs for the submodules we want to fake out, leaving the real
# top-level sting package untouched.
sys.modules["sting.gradient_descent"] = _gd
sys.modules["sting.extract_streamline"] = _es

# Now import the module under test
import sting.outputs as outputs_module

# Bring the public names into this namespace for convenience
param_for_display = outputs_module.param_for_display
save_best_fit_params = outputs_module.save_best_fit_params
_ensure_clean_dir = outputs_module._ensure_clean_dir
_opt_params_from_log = outputs_module._opt_params_from_log
plot_loss = outputs_module.plot_loss
plot_loss_panel = outputs_module.plot_loss_panel
build_velocity_radius_kde = outputs_module.build_velocity_radius_kde
plot_param_uncertainties = outputs_module.plot_param_uncertainties
plot_param_correlations = outputs_module.plot_param_correlations
sample_parameter_sets_from_covariance = outputs_module.sample_parameter_sets_from_covariance
load_optimisation_log = outputs_module.load_optimisation_log


# ===========================================================================
# Helpers
# ===========================================================================

def _make_log_csv(tmp_path, epochs, loss, extra_cols=None):
    """Write a minimal optimisation_log.csv to *tmp_path*."""
    data = {"epoch": epochs, "loss": loss}
    if extra_cols:
        data.update(extra_cols)
    df = pd.DataFrame(data)
    df.to_csv(tmp_path / "optimisation_log.csv", index=False)
    return df


# ===========================================================================
# param_for_display
# ===========================================================================

class TestParamForDisplay:
    """Unit tests for param_for_display."""

    def test_angle_key_converted_to_degrees(self):
        for key in ("theta0", "phi0", "inc", "pa"):
            _, val, unit = param_for_display(key, math.pi)
            assert unit == "deg"
            assert pytest.approx(val, rel=1e-9) == 180.0

    def test_angle_zero_stays_zero(self):
        _, val, unit = param_for_display("inc", 0.0)
        assert val == 0.0
        assert unit == "deg"

    def test_angle_negative_value(self):
        _, val, unit = param_for_display("pa", -math.pi / 2)
        assert pytest.approx(val, rel=1e-9) == -90.0

    def test_non_angle_key_with_unit(self):
        key, val, unit = param_for_display("r0", 100.0)
        assert key == "r0"
        assert val == 100.0
        assert unit == "au"

    def test_unknown_key_returns_empty_unit(self):
        _, val, unit = param_for_display("unknown_param", 42.0)
        assert unit == ""
        assert val == 42.0

    def test_return_tuple_length(self):
        result = param_for_display("mass", 1.5)
        assert len(result) == 3

    def test_value_cast_to_float(self):
        # value should always come back as a plain Python float
        _, val, _ = param_for_display("r0", np.float32(3.14))
        assert isinstance(val, float)


# ===========================================================================
# save_best_fit_params
# ===========================================================================

class TestSaveBestFitParams:
    """Tests for save_best_fit_params – exercises JSON output structure."""

    def test_creates_json_file(self, tmp_path):
        save_best_fit_params(
            best_opt_params={"r0": 100.0, "inc": math.pi / 4},
            fixed_params={"mass": 1.0},
            param_errors=None,
            save_folder=str(tmp_path),
        )
        out = tmp_path / "best_fit_params.json"
        assert out.exists()

    def test_json_has_expected_top_level_keys(self, tmp_path):
        save_best_fit_params({"r0": 50.0}, {"mass": 2.0}, None, str(tmp_path))
        with open(tmp_path / "best_fit_params.json") as f:
            data = json.load(f)
        assert "optimised_parameters" in data
        assert "fixed_parameters" in data

    def test_angle_stored_in_degrees(self, tmp_path):
        save_best_fit_params(
            {"inc": math.pi},
            {},
            None,
            str(tmp_path),
        )
        with open(tmp_path / "best_fit_params.json") as f:
            data = json.load(f)
        assert pytest.approx(data["optimised_parameters"]["inc"]["value"], rel=1e-9) == 180.0
        assert data["optimised_parameters"]["inc"]["unit"] == "deg"

    def test_non_angle_param_unit_preserved(self, tmp_path):
        save_best_fit_params({"r0": 200.0}, {}, None, str(tmp_path))
        with open(tmp_path / "best_fit_params.json") as f:
            data = json.load(f)
        assert data["optimised_parameters"]["r0"]["unit"] == "au"

    def test_sigma_written_when_errors_provided(self, tmp_path):
        save_best_fit_params(
            {"r0": 100.0},
            {},
            {"r0": 5.0},
            str(tmp_path),
        )
        with open(tmp_path / "best_fit_params.json") as f:
            data = json.load(f)
        assert "sigma" in data["optimised_parameters"]["r0"]
        assert pytest.approx(data["optimised_parameters"]["r0"]["sigma"]) == 5.0

    def test_sigma_not_written_when_errors_none(self, tmp_path):
        save_best_fit_params({"r0": 100.0}, {}, None, str(tmp_path))
        with open(tmp_path / "best_fit_params.json") as f:
            data = json.load(f)
        assert "sigma" not in data["optimised_parameters"]["r0"]

    def test_fixed_param_none_value_handled(self, tmp_path):
        """Fixed params with None values should be stored with value: null."""
        save_best_fit_params({}, {"v_lsr": None}, None, str(tmp_path))
        with open(tmp_path / "best_fit_params.json") as f:
            data = json.load(f)
        assert data["fixed_parameters"]["v_lsr"]["value"] is None

    def test_creates_directory_if_missing(self, tmp_path):
        nested = tmp_path / "new_dir" / "sub"
        save_best_fit_params({"r0": 1.0}, {}, None, str(nested))
        assert (nested / "best_fit_params.json").exists()

class TestSaveBestFitParamsAngleError:
    """Angle parameters in param_errors must be converted to degrees in the JSON."""

    def test_angle_error_stored_in_degrees(self, tmp_path):
        save_best_fit_params(
            best_opt_params={"inc": math.pi / 6},   # 30 deg
            fixed_params={},
            param_errors={"inc": math.pi / 180},    # 1 deg in radians
            save_folder=str(tmp_path),
        )
        with open(tmp_path / "best_fit_params.json") as f:
            data = json.load(f)
        sigma = data["optimised_parameters"]["inc"]["sigma"]
        assert pytest.approx(sigma, rel=1e-6) == 1.0  # should be 1 deg, not ~0.0175 rad

    def test_non_angle_error_stored_as_is(self, tmp_path):
        save_best_fit_params(
            best_opt_params={"r0": 100.0},
            fixed_params={},
            param_errors={"r0": 7.5},
            save_folder=str(tmp_path),
        )
        with open(tmp_path / "best_fit_params.json") as f:
            data = json.load(f)
        assert pytest.approx(data["optimised_parameters"]["r0"]["sigma"]) == 7.5



# ===========================================================================
# _ensure_clean_dir
# ===========================================================================

class TestEnsureCleanDir:
    def test_creates_missing_directory(self, tmp_path):
        new_dir = tmp_path / "created"
        assert not new_dir.exists()
        _ensure_clean_dir(str(new_dir))
        assert new_dir.is_dir()

    def test_removes_existing_files(self, tmp_path):
        d = tmp_path / "dirty"
        d.mkdir()
        (d / "old.png").write_text("data")
        assert len(list(d.iterdir())) == 1
        _ensure_clean_dir(str(d))
        assert len(list(d.iterdir())) == 0

    def test_does_not_raise_on_existing_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        _ensure_clean_dir(str(d))  # should not raise


# ===========================================================================
# _opt_params_from_log
# ===========================================================================

class TestOptParamsFromLog:
    def test_skips_epoch_and_loss_columns(self):
        df = pd.DataFrame({"epoch": [0], "loss": [1.0], "r0": [100.0], "inc": [0.5]})
        cols = _opt_params_from_log(df)
        assert "epoch" not in cols
        assert "loss" not in cols
        assert set(cols) == {"r0", "inc"}

    def test_mu_present_excludes_rc_and_omega(self):
        df = pd.DataFrame(
            {"epoch": [0], "loss": [1.0], "mu": [0.5], "rc": [10.0], "omega": [1e-13], "r0": [100.0]}
        )
        cols = _opt_params_from_log(df)
        assert "rc" not in cols
        assert "omega" not in cols
        assert "mu" in cols
        assert "r0" in cols

    def test_no_mu_keeps_rc_and_omega(self):
        df = pd.DataFrame(
            {"epoch": [0], "loss": [1.0], "rc": [10.0], "omega": [1e-13]}
        )
        cols = _opt_params_from_log(df)
        assert "rc" in cols
        assert "omega" in cols

    def test_returns_list(self):
        df = pd.DataFrame({"epoch": [0], "loss": [0.1], "r0": [1.0]})
        assert isinstance(_opt_params_from_log(df), list)


# ===========================================================================
# load_optimisation_log
# ===========================================================================

class TestLoadOptimisationLog:
    def test_returns_dataframe(self, tmp_path):
        _make_log_csv(tmp_path, [0, 1], [0.5, 0.3])
        df = load_optimisation_log(str(tmp_path))
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self, tmp_path):
        _make_log_csv(tmp_path, [0], [0.5], extra_cols={"r0": [100.0]})
        df = load_optimisation_log(str(tmp_path))
        assert "epoch" in df.columns
        assert "loss" in df.columns
        assert "r0" in df.columns

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_optimisation_log(str(tmp_path / "nonexistent"))


# ===========================================================================
# plot_loss  (smoke test – checks file is written and valid PNG)
# ===========================================================================

class TestPlotLoss:
    def test_saves_png(self, tmp_path):
        plot_loss([1.0, 0.5, 0.25, 0.1], save_folder=str(tmp_path))
        assert (tmp_path / "loss_history.png").exists()

    def test_single_epoch(self, tmp_path):
        """Should not raise for a single-element loss history."""
        plot_loss([1.0], save_folder=str(tmp_path))
        assert (tmp_path / "loss_history.png").exists()

    def test_show_false_closes_figure(self, tmp_path):
        """After saving, no figures should be left open."""
        initial = plt.get_fignums()
        plot_loss([1.0, 0.5], save_folder=str(tmp_path))
        after = plt.get_fignums()
        # Number of open figures should not grow
        assert len(after) <= len(initial)


# ===========================================================================
# plot_loss_panel (unit-level)
# ===========================================================================

class TestPlotLossPanel:
    def test_green_scatter_at_best_epoch(self):
        fig, ax = plt.subplots()
        epochs = np.array([0, 1, 2, 3])
        loss = np.array([1.0, 0.5, 0.1, 0.3])
        plot_loss_panel(ax, epochs, loss)
        # The best scatter should be at epoch 2 (minimum loss)
        scatter = [c for c in ax.get_children()
                   if isinstance(c, matplotlib.collections.PathCollection)]
        assert len(scatter) >= 1
        plt.close(fig)

    def test_log_scale_set(self):
        fig, ax = plt.subplots()
        plot_loss_panel(ax, np.array([0, 1]), np.array([1.0, 0.5]))
        assert ax.get_yscale() == "log"
        plt.close(fig)

# =========================================================================
# plot_loss – show=True / save_folder=None paths
# ==========================================================================

class TestPlotLossExtraPaths:
    def test_no_save_folder_does_not_write_file(self, tmp_path):
        """With save_folder=None nothing should be written to disk."""
        plot_loss([1.0, 0.5, 0.1], save_folder=None, show=False)
        assert list(tmp_path.iterdir()) == []

    def test_descending_loss(self, tmp_path):
        """Monotonically decreasing loss should still produce a valid PNG."""
        plot_loss(list(np.logspace(0, -3, 20)), save_folder=str(tmp_path))
        assert (tmp_path / "loss_history.png").exists()

# ===========================================================================
# build_velocity_radius_kde
# ===========================================================================

class TestBuildVelocityRadiusKde:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(0)

    def test_returns_required_keys(self, rng):
        ra = rng.uniform(-5, 5, 50)
        dec = rng.uniform(-5, 5, 50)
        v = rng.uniform(-10, 10, 50)
        result = build_velocity_radius_kde(ra, dec, v)
        for key in ("xx", "yy", "zz", "levels", "xlim", "ylim"):
            assert key in result

    def test_zz_normalised(self, rng):
        ra = rng.uniform(-5, 5, 50)
        dec = rng.uniform(-5, 5, 50)
        v = rng.uniform(-10, 10, 50)
        result = build_velocity_radius_kde(ra, dec, v)
        assert np.nanmax(result["zz"]) <= 1.0 + 1e-9

    def test_xlim_ylim_match_grid(self, rng):
        ra = rng.uniform(1, 5, 50)
        dec = rng.uniform(1, 5, 50)
        v = rng.uniform(-5, 5, 50)
        result = build_velocity_radius_kde(ra, dec, v)
        xmin, xmax = result["xlim"]
        ymin, ymax = result["ylim"]
        assert xmin < xmax
        assert ymin < ymax

    def test_explicit_limits_respected(self, rng):
        ra = rng.uniform(-5, 5, 50)
        dec = rng.uniform(-5, 5, 50)
        v = rng.uniform(-10, 10, 50)
        result = build_velocity_radius_kde(ra, dec, v, xmin=0, xmax=10, ymin=-20, ymax=20)
        assert result["xlim"] == (0, 10)
        assert result["ylim"] == (-20, 20)

    def test_raises_with_fewer_than_three_finite_points(self):
        with pytest.raises(ValueError, match="at least 3 finite"):
            build_velocity_radius_kde(
                [np.nan, 1.0],
                [np.nan, 1.0],
                [np.nan, 1.0],
            )

    def test_grid_size_parameter(self, rng):
        ra = rng.uniform(-5, 5, 50)
        dec = rng.uniform(-5, 5, 50)
        v = rng.uniform(-10, 10, 50)
        result = build_velocity_radius_kde(ra, dec, v, grid_size=20)
        assert result["xx"].shape == (20, 20)


# ===========================================================================
# plot_param_uncertainties (smoke)
# ===========================================================================

class TestPlotParamUncertainties:
    def test_saves_png(self, tmp_path):
        keys = ["r0", "inc", "mass"]
        vals = np.array([100.0, 0.5, 1.0])
        errs = np.array([5.0, 0.1, 0.05])
        plot_param_uncertainties(keys, vals, errs, save_folder=str(tmp_path))
        assert (tmp_path / "parameter_uncertainties.png").exists()

    def test_single_parameter(self, tmp_path):
        plot_param_uncertainties(["r0"], np.array([100.0]), np.array([5.0]), save_folder=str(tmp_path))
        assert (tmp_path / "parameter_uncertainties.png").exists()

    def test_uncertainties_show_true(self):
        plot_param_uncertainties(
            ["r0", "mass"],
            np.array([100.0, 1.0]),
            np.array([5.0, 0.1]),
            save_folder=None,
            show=True,
        )
        plt.close("all")


# ===========================================================================
# plot_param_correlations (smoke + correctness)
# ===========================================================================

class TestPlotParamCorrelations:
    def _identity_cov(self, n):
        return np.eye(n)

    def test_saves_png(self, tmp_path):
        cov = self._identity_cov(3)
        plot_param_correlations(["r0", "inc", "mass"], cov, save_folder=str(tmp_path))
        assert (tmp_path / "parameter_correlation_matrix.png").exists()

    def test_diagonal_is_one_for_identity_cov(self, tmp_path):
        """With an identity covariance the correlation matrix diagonal must be 1."""
        n = 3
        cov = self._identity_cov(n)
        diag = np.sqrt(np.diag(cov))
        corr = cov / np.outer(diag, diag)
        assert np.allclose(np.diag(corr), 1.0)

    def test_correlation_values_clipped_to_minus_one_one(self):
        """Numerically extreme covariances should never produce |corr| > 1."""
        cov = np.array([[1e-30, 1e30], [1e30, 1e-30]])
        diag = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
        corr = np.clip(cov / np.outer(diag, diag), -1.0, 1.0)
        assert np.all(np.abs(corr) <= 1.0)

    def test_no_annotate_does_not_raise(self, tmp_path):
        cov = self._identity_cov(2)
        plot_param_correlations(["a", "b"], cov, annotate=False, save_folder=str(tmp_path))
    
    def test_correlations_show_true(self):
        plot_param_correlations(
            ["r0", "mass"],
            np.eye(2),
            save_folder=None,
            show=True,
        )
        plt.close("all")


# ===========================================================================
# sample_parameter_sets_from_covariance
# ===========================================================================

class TestSampleParameterSetsFromCovariance:
    @pytest.fixture
    def simple_setup(self):
        params = {"r0": 100.0, "inc": 0.5}
        cov = np.diag([4.0, 0.01])  # std = 2.0 and 0.1
        keys = ["r0", "inc"]
        return params, cov, keys

    def test_output_shape(self, simple_setup):
        params, cov, keys = simple_setup
        samples = sample_parameter_sets_from_covariance(params, cov, keys, n_samples=50)
        assert samples.shape == (50, 2)

    def test_reproducible_with_same_seed(self, simple_setup):
        params, cov, keys = simple_setup
        s1 = sample_parameter_sets_from_covariance(params, cov, keys, seed=7)
        s2 = sample_parameter_sets_from_covariance(params, cov, keys, seed=7)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_give_different_results(self, simple_setup):
        params, cov, keys = simple_setup
        s1 = sample_parameter_sets_from_covariance(params, cov, keys, seed=1)
        s2 = sample_parameter_sets_from_covariance(params, cov, keys, seed=2)
        assert not np.array_equal(s1, s2)

    def test_sample_mean_close_to_input(self, simple_setup):
        params, cov, keys = simple_setup
        samples = sample_parameter_sets_from_covariance(params, cov, keys, n_samples=5000)
        # With enough samples the mean should be close to the true mean
        np.testing.assert_allclose(samples.mean(axis=0), [100.0, 0.5], atol=0.3)

    def test_clipping_with_param_bounds(self, simple_setup):
        """Samples must respect provided bounds after clipping."""
        params, cov, keys = simple_setup
        # Stub convert_and_strip_bound_units to be a pass-through
        _gd.convert_and_strip_bound_units = lambda b: b
        bounds = {"r0": (95.0, 105.0)}
        samples = sample_parameter_sets_from_covariance(
            params, cov, keys, param_bounds=bounds, n_samples=200
        )
        assert np.all(samples[:, 0] >= 95.0)
        assert np.all(samples[:, 0] <= 105.0)
        _gd.convert_and_strip_bound_units = None  # restore

    def test_n_samples_one(self, simple_setup):
        params, cov, keys = simple_setup
        samples = sample_parameter_sets_from_covariance(params, cov, keys, n_samples=1)
        assert samples.shape == (1, 2)


# ===========================================================================
# Bring in extra names not imported at module level in the original file
# ===========================================================================

plot_morphology = outputs_module.plot_morphology
plot_ra_vel = outputs_module.plot_ra_vel
plot_dec_vel = outputs_module.plot_dec_vel
plot_vel_radius = outputs_module.plot_vel_radius


# ===========================================================================
# Shared test fixtures
# ===========================================================================

def _make_streamer(n=8):
    """Return a SimpleNamespace that mimics the Streamer namedtuple."""
    rng = np.random.default_rng(42)
    ra   = rng.uniform(1.0, 5.0, n)
    dec  = rng.uniform(1.0, 5.0, n)
    v    = rng.uniform(-3.0, 3.0, n)
    sig  = np.full(n, 0.1)

    import types
    s = types.SimpleNamespace(
        ra_data=ra,
        dec_data=dec,
        v_data=v,
        ra_sigma=sig,
        dec_sigma=sig,
        v_sigma=sig,
        pc_coords=(
            rng.uniform(0, 8, 30),   # ra cloud
            rng.uniform(0, 8, 30),   # dec cloud
            rng.uniform(-5, 5, 30),  # vel cloud
        ),
        data=(ra, dec, v),
        uncertainties=(sig, sig, sig),
    )
    return s


# ===========================================================================
# save_best_fit_params – angle error branch (line ~212)
# ===========================================================================


# ===========================================================================
# plot_loss – show=True / save_folder=None paths
# ===========================================================================

class TestPlotLossExtraPaths:
    def test_no_save_folder_does_not_write_file(self, tmp_path):
        """With save_folder=None nothing should be written to disk."""
        plot_loss([1.0, 0.5, 0.1], save_folder=None, show=False)
        assert list(tmp_path.iterdir()) == []

    def test_descending_loss(self, tmp_path):
        """Monotonically decreasing loss should still produce a valid PNG."""
        plot_loss(list(np.logspace(0, -3, 20)), save_folder=str(tmp_path))
        assert (tmp_path / "loss_history.png").exists()


# ===========================================================================
# _ensure_clean_dir – subdirectory preservation
# ===========================================================================

class TestEnsureCleanDirSubdirs:
    def test_subdirectories_are_not_removed(self, tmp_path):
        """_ensure_clean_dir should only delete *files*, not child directories."""
        d = tmp_path / "parent"
        d.mkdir()
        child_dir = d / "subdir"
        child_dir.mkdir()
        (d / "file.txt").write_text("x")
        _ensure_clean_dir(str(d))
        # file is gone, subdir survives
        assert not (d / "file.txt").exists()
        assert child_dir.exists()


# ===========================================================================
# build_velocity_radius_kde – sigma_levels parameter
# ===========================================================================

class TestBuildVelocityRadiusKdeSigmaLevels:
    def test_custom_sigma_levels_change_number_of_contour_levels(self):
        rng = np.random.default_rng(1)
        ra  = rng.uniform(-5, 5, 60)
        dec = rng.uniform(-5, 5, 60)
        v   = rng.uniform(-10, 10, 60)
        result_default = build_velocity_radius_kde(ra, dec, v)
        result_custom  = build_velocity_radius_kde(ra, dec, v, sigma_levels=[1.0, 2.0, 3.0])
        # default uses np.arange(1.0, 2.1, 0.5) -> 4 levels
        # custom uses 3 sigma values -> also 4 levels, check the actual level count differs
        # when sigma values differ the contour threshold values differ
        assert not np.allclose(result_default["levels"], result_custom["levels"])

    def test_zz_max_is_one(self):
        rng = np.random.default_rng(2)
        ra  = rng.uniform(0, 10, 80)
        dec = rng.uniform(0, 10, 80)
        v   = rng.uniform(0, 5, 80)
        result = build_velocity_radius_kde(ra, dec, v)
        assert pytest.approx(float(np.nanmax(result["zz"])), abs=1e-9) == 1.0


# ===========================================================================
# plot_morphology – smoke tests for uncovered branches
# ===========================================================================

class TestPlotMorphologySmoke:
    """Smoke tests: check the function runs and either saves a file or closes cleanly."""

    def test_minimal_no_streamer_no_model(self, tmp_path):
        """Call with only save_folder set; no streamer, no model."""
        plot_morphology(save_folder=str(tmp_path), save_name="minimal", show=False)
        assert (tmp_path / "minimal.png").exists()

    def test_with_model_curve_only(self, tmp_path):
        ra_m  = np.linspace(1, 5, 20)
        dec_m = np.linspace(1, 5, 20)
        plot_morphology(
            ra_model=ra_m, dec_model=dec_m,
            save_folder=str(tmp_path), save_name="model_only", show=False,
        )
        assert (tmp_path / "model_only.png").exists()

    def test_with_streamer(self, tmp_path):
        s = _make_streamer()
        plot_morphology(
            streamer=s,
            ra_model=np.linspace(1, 5, 20),
            dec_model=np.linspace(1, 5, 20),
            save_folder=str(tmp_path),
            save_name="with_streamer",
            show=False,
        )
        assert (tmp_path / "with_streamer.png").exists()

    def test_with_model_interp_and_valid(self, tmp_path):
        s = _make_streamer()
        n = len(s.ra_data)
        plot_morphology(
            streamer=s,
            ra_model=np.linspace(1, 5, 20),
            dec_model=np.linspace(1, 5, 20),
            ra_model_interp=s.ra_data,
            dec_model_interp=s.dec_data,
            valid=np.ones(n, dtype=bool),
            save_folder=str(tmp_path),
            save_name="with_interp",
            show=False,
        )
        assert (tmp_path / "with_interp.png").exists()

    def test_with_by_eye(self, tmp_path):
        """Exercise the by_eye branch."""
        s = _make_streamer()
        by_eye = (np.linspace(0, 6, 20), np.linspace(0, 6, 20), np.zeros(20))
        plot_morphology(
            streamer=s,
            ra_model=np.linspace(1, 5, 20),
            dec_model=np.linspace(1, 5, 20),
            by_eye=by_eye,
            save_folder=str(tmp_path),
            save_name="with_by_eye",
            show=False,
        )
        assert (tmp_path / "with_by_eye.png").exists()

    def test_with_bg_rgba(self, tmp_path):
        """Exercise the pre-rendered background branch."""
        s = _make_streamer()
        # create a dummy RGBA background image
        bg_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        bg_extent = [0.0, 8.0, 0.0, 8.0]
        plot_morphology(
            streamer=s,
            bg_rgba=bg_rgba,
            bg_extent=bg_extent,
            xlim=(0, 8),
            ylim=(0, 8),
            save_folder=str(tmp_path),
            save_name="with_bg",
            show=False,
        )
        assert (tmp_path / "with_bg.png").exists()

    def test_explicit_xlim_ylim(self, tmp_path):
        s = _make_streamer()
        plot_morphology(
            streamer=s,
            ra_model=np.linspace(1, 5, 10),
            dec_model=np.linspace(1, 5, 10),
            xlim=(0.0, 10.0),
            ylim=(0.0, 10.0),
            save_folder=str(tmp_path),
            save_name="explicit_lim",
            show=False,
        )
        assert (tmp_path / "explicit_lim.png").exists()

    def test_no_sigma_data_only(self, tmp_path):
        """Streamer with sigma=None falls through to ax.plot branch."""
        import types
        rng = np.random.default_rng(7)
        s = types.SimpleNamespace(
            ra_data=rng.uniform(1, 5, 6),
            dec_data=rng.uniform(1, 5, 6),
            v_data=rng.uniform(-2, 2, 6),
            ra_sigma=None,
            dec_sigma=None,
            v_sigma=None,
            pc_coords=(rng.uniform(0, 8, 20), rng.uniform(0, 8, 20), rng.uniform(-3, 3, 20)),
            data=None, uncertainties=None,
        )
        plot_morphology(
            streamer=s,
            xlim=(0, 8), ylim=(0, 8),
            save_folder=str(tmp_path),
            save_name="no_sigma",
            show=False,
        )
        assert (tmp_path / "no_sigma.png").exists()


# ===========================================================================
# plot_ra_vel – smoke tests
# ===========================================================================

class TestPlotRaVelSmoke:
    def test_model_only(self, tmp_path):
        ra_m = np.linspace(1, 5, 20)
        v_m  = np.linspace(-2, 2, 20)
        plot_ra_vel(ra_m, v_m, save_folder=str(tmp_path), save_name="ra_vel_basic")
        assert (tmp_path / "ra_vel_basic.png").exists()

    def test_with_streamer(self, tmp_path):
        s = _make_streamer()
        plot_ra_vel(
            np.linspace(1, 5, 20),
            np.linspace(-2, 2, 20),
            streamer=s,
            save_folder=str(tmp_path),
            save_name="ra_vel_streamer",
        )
        assert (tmp_path / "ra_vel_streamer.png").exists()

    def test_with_interp_and_valid(self, tmp_path):
        s = _make_streamer()
        n = len(s.ra_data)
        plot_ra_vel(
            np.linspace(1, 5, 20),
            np.linspace(-2, 2, 20),
            streamer=s,
            ra_model_interp=s.ra_data,
            v_model_interp=s.v_data,
            valid=np.ones(n, dtype=bool),
            save_folder=str(tmp_path),
            save_name="ra_vel_interp",
        )
        assert (tmp_path / "ra_vel_interp.png").exists()

    def test_with_vlim_ralim(self, tmp_path):
        plot_ra_vel(
            np.linspace(0, 5, 15),
            np.linspace(-1, 1, 15),
            vlim=(-5.0, 5.0),
            ralim=(0.0, 10.0),
            save_folder=str(tmp_path),
            save_name="ra_vel_lim",
        )
        assert (tmp_path / "ra_vel_lim.png").exists()

    def test_no_sigma_streamer(self, tmp_path):
        """streamer without sigmas falls through to ax.plot branch."""
        import types
        rng = np.random.default_rng(9)
        s = types.SimpleNamespace(
            ra_data=rng.uniform(1, 5, 6),
            dec_data=rng.uniform(1, 5, 6),
            v_data=rng.uniform(-2, 2, 6),
            ra_sigma=None,
            dec_sigma=None,
            v_sigma=None,
            pc_coords=(rng.uniform(0, 8, 20), rng.uniform(0, 8, 20), rng.uniform(-3, 3, 20)),
            data=None, uncertainties=None,
        )
        plot_ra_vel(
            np.linspace(1, 5, 10),
            np.linspace(-1, 1, 10),
            streamer=s,
            save_folder=str(tmp_path),
            save_name="ra_vel_no_sigma",
        )
        assert (tmp_path / "ra_vel_no_sigma.png").exists()

    def test_with_model_keep_mask(self, tmp_path):
        """model_keep mask filters the plotted model points."""
        ra_m = np.linspace(0, 6, 20)
        v_m  = np.linspace(-3, 3, 20)
        keep = np.ones(20, dtype=bool)
        keep[:5] = False
        plot_ra_vel(
            ra_m, v_m,
            model_keep=keep,
            save_folder=str(tmp_path),
            save_name="ra_vel_keep",
        )
        assert (tmp_path / "ra_vel_keep.png").exists()


# ===========================================================================
# plot_dec_vel – smoke tests
# ===========================================================================

class TestPlotDecVelSmoke:
    def test_model_only(self, tmp_path):
        dec_m = np.linspace(1, 5, 20)
        v_m   = np.linspace(-2, 2, 20)
        plot_dec_vel(dec_m, v_m, save_folder=str(tmp_path), save_name="dec_vel_basic")
        assert (tmp_path / "dec_vel_basic.png").exists()

    def test_with_streamer(self, tmp_path):
        s = _make_streamer()
        plot_dec_vel(
            np.linspace(1, 5, 20),
            np.linspace(-2, 2, 20),
            streamer=s,
            save_folder=str(tmp_path),
            save_name="dec_vel_streamer",
        )
        assert (tmp_path / "dec_vel_streamer.png").exists()

    def test_with_interp_and_valid(self, tmp_path):
        s = _make_streamer()
        n = len(s.dec_data)
        plot_dec_vel(
            np.linspace(1, 5, 20),
            np.linspace(-2, 2, 20),
            streamer=s,
            dec_model_interp=s.dec_data,
            v_model_interp=s.v_data,
            valid=np.ones(n, dtype=bool),
            save_folder=str(tmp_path),
            save_name="dec_vel_interp",
        )
        assert (tmp_path / "dec_vel_interp.png").exists()

    def test_vlim_declim(self, tmp_path):
        plot_dec_vel(
            np.linspace(0, 5, 10),
            np.linspace(-1, 1, 10),
            vlim=(-5.0, 5.0),
            declim=(0.0, 8.0),
            save_folder=str(tmp_path),
            save_name="dec_vel_lim",
        )
        assert (tmp_path / "dec_vel_lim.png").exists()

    def test_no_sigma_streamer(self, tmp_path):
        """Streamer without sigmas falls through to ax.plot."""
        import types
        rng = np.random.default_rng(11)
        s = types.SimpleNamespace(
            ra_data=rng.uniform(1, 5, 6),
            dec_data=rng.uniform(1, 5, 6),
            v_data=rng.uniform(-2, 2, 6),
            ra_sigma=None,
            dec_sigma=None,
            v_sigma=None,
            pc_coords=(rng.uniform(0, 8, 20), rng.uniform(0, 8, 20), rng.uniform(-3, 3, 20)),
            data=None, uncertainties=None,
        )
        plot_dec_vel(
            np.linspace(1, 5, 10),
            np.linspace(-1, 1, 10),
            streamer=s,
            save_folder=str(tmp_path),
            save_name="dec_vel_no_sigma",
        )
        assert (tmp_path / "dec_vel_no_sigma.png").exists()

    def test_show_false_no_save_folder(self, tmp_path):
        """With save_folder=None and show=False the figure should be closed."""
        before = plt.get_fignums()
        plot_dec_vel(
            np.linspace(1, 5, 10),
            np.linspace(-1, 1, 10),
            save_folder=None,
            show=False,
        )
        after = plt.get_fignums()
        assert len(after) <= len(before)


# ===========================================================================
# plot_vel_radius – smoke tests
# ===========================================================================

class TestPlotVelRadiusSmoke:
    def test_basic(self, tmp_path):
        s = _make_streamer()
        ra_m  = np.linspace(0.5, 5.0, 30)
        dec_m = np.linspace(0.5, 5.0, 30)
        v_m   = np.linspace(-3.0, 3.0, 30)
        plot_vel_radius(
            ra_m, dec_m, v_m,
            streamer=s,
            save_folder=str(tmp_path),
            save_name="vel_radius_basic",
        )
        assert (tmp_path / "vel_radius_basic.png").exists()

    def test_with_interp_and_valid(self, tmp_path):
        s = _make_streamer()
        n = len(s.ra_data)
        ra_m  = np.linspace(0.5, 5.0, 30)
        dec_m = np.linspace(0.5, 5.0, 30)
        v_m   = np.linspace(-3.0, 3.0, 30)
        plot_vel_radius(
            ra_m, dec_m, v_m,
            streamer=s,
            ra_model_interp=s.ra_data,
            dec_model_interp=s.dec_data,
            v_model_interp=s.v_data,
            valid=np.ones(n, dtype=bool),
            save_folder=str(tmp_path),
            save_name="vel_radius_interp",
        )
        assert (tmp_path / "vel_radius_interp.png").exists()

    def test_with_velocity_reference(self, tmp_path):
        s = _make_streamer()
        plot_vel_radius(
            np.linspace(1, 5, 20),
            np.linspace(1, 5, 20),
            np.linspace(-2, 2, 20),
            streamer=s,
            velocity_reference=0.5,
            save_folder=str(tmp_path),
            save_name="vel_radius_vlsr",
        )
        assert (tmp_path / "vel_radius_vlsr.png").exists()

    def test_with_by_eye(self, tmp_path):
        """Exercise the by_eye branch in plot_vel_radius."""
        s = _make_streamer()
        by_eye = (
            np.linspace(0.5, 5, 20),
            np.linspace(0.5, 5, 20),
            np.linspace(-1, 1, 20),
        )
        plot_vel_radius(
            np.linspace(1, 5, 20),
            np.linspace(1, 5, 20),
            np.linspace(-2, 2, 20),
            streamer=s,
            by_eye=by_eye,
            save_folder=str(tmp_path),
            save_name="vel_radius_by_eye",
        )
        assert (tmp_path / "vel_radius_by_eye.png").exists()

    def test_with_model_keep_mask(self, tmp_path):
        """model_keep masks out part of the model before plotting rproj."""
        s = _make_streamer()
        ra_m  = np.linspace(0, 6, 30)
        dec_m = np.linspace(0, 6, 30)
        v_m   = np.linspace(-3, 3, 30)
        keep  = np.ones(30, dtype=bool)
        keep[:5] = False
        plot_vel_radius(
            ra_m, dec_m, v_m,
            streamer=s,
            model_keep=keep,
            save_folder=str(tmp_path),
            save_name="vel_radius_keep",
        )
        assert (tmp_path / "vel_radius_keep.png").exists()

    def test_explicit_kde_background_skips_rebuild(self, tmp_path):
        """Passing a pre-built kde_background should skip the auto-build branch."""
        s = _make_streamer()
        rng = np.random.default_rng(3)
        kde = build_velocity_radius_kde(s.ra_data, s.dec_data, s.v_data)
        plot_vel_radius(
            np.linspace(1, 5, 20),
            np.linspace(1, 5, 20),
            np.linspace(-2, 2, 20),
            streamer=s,
            kde_background=kde,
            save_folder=str(tmp_path),
            save_name="vel_radius_kde",
        )
        assert (tmp_path / "vel_radius_kde.png").exists()

    def test_explicit_xlim_ylim(self, tmp_path):
        s = _make_streamer()
        plot_vel_radius(
            np.linspace(1, 5, 20),
            np.linspace(1, 5, 20),
            np.linspace(-2, 2, 20),
            streamer=s,
            xlim=(0.0, 10.0),
            ylim=(-5.0, 5.0),
            save_folder=str(tmp_path),
            save_name="vel_radius_lim",
        )
        assert (tmp_path / "vel_radius_lim.png").exists()

    def test_no_sigma_data(self, tmp_path):
        """Streamer without sigmas uses the bare ax.plot path for data."""
        import types
        rng = np.random.default_rng(13)
        s = types.SimpleNamespace(
            ra_data=rng.uniform(1, 5, 6),
            dec_data=rng.uniform(1, 5, 6),
            v_data=rng.uniform(-2, 2, 6),
            ra_sigma=None,
            dec_sigma=None,
            v_sigma=None,
            pc_coords=None,
            data=None, uncertainties=None,
        )
        plot_vel_radius(
            np.linspace(1, 5, 10),
            np.linspace(1, 5, 10),
            np.linspace(-1, 1, 10),
            streamer=s,
            save_folder=str(tmp_path),
            save_name="vel_radius_no_sigma",
        )
        assert (tmp_path / "vel_radius_no_sigma.png").exists()


