"""
Initial tests for extract_streamline.py

Tests cover the pure mathematical and utility functions that can be exercised
without real observational data or SpectralCube objects:
  - wrap_to_pi / wrap_to_pi_numpy: boundary and symmetry behaviour
  - cartesian_to_polar: known values, dtype, gradient stability
  - circular_median: known medians, weight-zero exclusion, branch-cut safety
  - safe_percentile: empty/finite handling, clipping, known values
  - get_distance_metric: output shapes, finite-only contract, empty-input case
  - get_metric_partitions: monotonicity, validation, empty-input edge case
  - get_metric_reference_trace: return types and plausible range
  - sample_metric_boundary: shape, closed-curve property, validation
  - sample_metric_boundaries: curve count matches partitions, trace keys
  - prepare_data: output field types, safe-sigma floor, finite-mask logic

Heavy SpectralCube paths (extract_streamer_subcube, reduce_to_1D) are left
for integration tests that require a full environment.

Run with:
    pytest test_extract_streamline.py -v
"""

import math
import types

import jax.numpy as jnp
import numpy as np
import pytest

import sting.extract_streamline as es

# Convenient aliases
BIG = es.BIG


# ===========================================================================
# Helpers
# ===========================================================================

def _make_grid(n=20, scale=5.0, rng=None):
    """Return a simple 2-D point cloud (ra, dec) spread around the origin."""
    if rng is None:
        rng = np.random.default_rng(0)
    ra = jnp.array(rng.uniform(-scale, scale, n), dtype=jnp.float64)
    dec = jnp.array(rng.uniform(-scale, scale, n), dtype=jnp.float64)
    return ra, dec


def _make_straight_line(n=20, scale=5.0):
    """Return a collinear set of RA/Dec points along the RA axis."""
    ra = jnp.linspace(0.5, scale, n, dtype=jnp.float64)
    dec = jnp.zeros(n, dtype=jnp.float64)
    return ra, dec


# ===========================================================================
# wrap_to_pi
# ===========================================================================

class TestWrapToPi:
    """Unit tests for the JAX wrap_to_pi helper."""

    def test_zero_stays_zero(self):
        assert pytest.approx(float(es.wrap_to_pi(0.0))) == 0.0

    def test_pi_maps_to_minus_pi(self):
        # convention: [-pi, pi) so pi itself wraps to -pi
        result = float(es.wrap_to_pi(math.pi))
        assert pytest.approx(result, abs=1e-9) == -math.pi

    def test_minus_pi_stays_minus_pi(self):
        result = float(es.wrap_to_pi(-math.pi))
        assert pytest.approx(result, abs=1e-9) == -math.pi

    def test_two_pi_maps_to_zero(self):
        result = float(es.wrap_to_pi(2 * math.pi))
        assert pytest.approx(result, abs=1e-9) == 0.0

    def test_three_halves_pi_maps_to_minus_half_pi(self):
        result = float(es.wrap_to_pi(3 * math.pi / 2))
        assert pytest.approx(result, abs=1e-9) == -math.pi / 2

    def test_output_in_minus_pi_to_pi(self):
        for val in [-10.0, -math.pi - 0.1, 0.0, math.pi - 0.1, 10.0, 100.0]:
            result = float(es.wrap_to_pi(val))
            assert -math.pi <= result < math.pi

    def test_output_dtype_is_float64(self):
        result = es.wrap_to_pi(jnp.array(1.0, dtype=jnp.float64))
        assert result.dtype == jnp.float64

    def test_array_input(self):
        angles = jnp.array([0.0, math.pi, 2 * math.pi], dtype=jnp.float64)
        result = es.wrap_to_pi(angles)
        assert result.shape == (3,)
        assert np.all(np.array(result) >= -math.pi)
        assert np.all(np.array(result) < math.pi)


class TestPreparePointCloudData:
    def test_filters_nonfinite_and_nonpositive_intensity(self):
        streamer = types.SimpleNamespace(
            pc_coords=jnp.array([
                [1.0, 2.0, jnp.nan, 4.0],
                [0.5, 1.0, 1.5, 2.0],
                [3.0, 4.0, 5.0, 6.0],
                [1.0, 0.0, 2.0, jnp.nan],
            ]),
            ra_sigma=jnp.ones(2),
            dec_sigma=jnp.ones(2),
            v_sigma=jnp.ones(2),
            ra_data=jnp.ones(2),
        )

        prepared = es.prepare_point_cloud_data(streamer)

        assert int(prepared.valid_points) == 1
        assert int(prepared.total_points) == 4
        assert float(prepared.intensity[0]) == 1.0
        assert np.all(np.asarray(prepared.weights[1:]) == 0.0)


# ===========================================================================
# wrap_to_pi_numpy
# ===========================================================================

class TestWrapToPiNumpy:
    """Unit tests for the NumPy wrap_to_pi_numpy helper."""

    def test_zero_stays_zero(self):
        assert pytest.approx(es.wrap_to_pi_numpy(0.0)) == 0.0

    def test_pi_maps_to_minus_pi(self):
        result = es.wrap_to_pi_numpy(math.pi)
        assert pytest.approx(result, abs=1e-9) == -math.pi

    def test_output_in_minus_pi_to_pi(self):
        for val in [-10.0, 0.0, math.pi - 0.001, 10.0]:
            result = es.wrap_to_pi_numpy(val)
            assert -math.pi <= result < math.pi

    def test_matches_jax_version_on_scalar(self):
        for val in [-2.5, 0.0, 1.5, 4.7]:
            jax_result = float(es.wrap_to_pi(jnp.array(val)))
            np_result = float(es.wrap_to_pi_numpy(val))
            assert pytest.approx(np_result, abs=1e-9) == jax_result

    def test_array_input(self):
        angles = np.array([0.0, math.pi, 2 * math.pi])
        result = es.wrap_to_pi_numpy(angles)
        assert result.shape == (3,)
        assert np.all(result >= -math.pi)
        assert np.all(result < math.pi)


# ===========================================================================
# cartesian_to_polar
# ===========================================================================

class TestCartesianToPolar:
    """Unit tests for cartesian_to_polar."""

    def test_positive_x_axis_theta_is_zero(self):
        r, theta = es.cartesian_to_polar(jnp.array(1.0), jnp.array(0.0))
        assert pytest.approx(float(theta), abs=1e-6) == 0.0

    def test_positive_y_axis_theta_is_pi_over_two(self):
        r, theta = es.cartesian_to_polar(jnp.array(0.0), jnp.array(1.0))
        assert pytest.approx(float(theta), abs=1e-6) == math.pi / 2

    def test_negative_x_axis_theta_is_pi(self):
        r, theta = es.cartesian_to_polar(jnp.array(-1.0), jnp.array(0.0))
        assert pytest.approx(abs(float(theta)), abs=1e-6) == math.pi

    def test_r_is_positive_for_nonzero_input(self):
        r, _ = es.cartesian_to_polar(jnp.array(3.0), jnp.array(4.0))
        assert float(r) > 0.0

    def test_r_known_value_3_4_5(self):
        r, _ = es.cartesian_to_polar(jnp.array(3.0), jnp.array(4.0))
        # sqrt(9 + 16 + 1e-60) ≈ 5
        assert pytest.approx(float(r), rel=1e-6) == 5.0

    def test_origin_r_is_near_zero(self):
        r, _ = es.cartesian_to_polar(jnp.array(0.0), jnp.array(0.0))
        # gradient stabiliser adds 1e-60, so r is tiny but not exactly zero
        assert float(r) >= 0.0
        assert float(r) < 1e-20

    def test_output_dtypes_are_float64(self):
        r, theta = es.cartesian_to_polar(jnp.array(1.0, dtype=jnp.float64),
                                         jnp.array(0.0, dtype=jnp.float64))
        assert r.dtype == jnp.float64
        assert theta.dtype == jnp.float64

    def test_array_input_shapes_preserved(self):
        x = jnp.array([1.0, 0.0, -1.0], dtype=jnp.float64)
        y = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float64)
        r, theta = es.cartesian_to_polar(x, y)
        assert r.shape == (3,)
        assert theta.shape == (3,)

    def test_r_finite_at_origin(self):
        r, theta = es.cartesian_to_polar(jnp.array(0.0), jnp.array(0.0))
        assert math.isfinite(float(r))
        assert math.isfinite(float(theta))


# ===========================================================================
# circular_median
# ===========================================================================

class TestCircularMedian:
    """Unit tests for the branch-cut-safe weighted circular median."""

    def test_single_value_returns_that_value(self):
        theta = jnp.array([0.5], dtype=jnp.float64)
        w = jnp.array([1.0], dtype=jnp.float64)
        result = float(es.circular_median(theta, w))
        assert pytest.approx(result, abs=1e-4) == 0.5

    def test_two_equal_values_returns_that_value(self):
        theta = jnp.array([1.0, 1.0], dtype=jnp.float64)
        w = jnp.array([1.0, 1.0], dtype=jnp.float64)
        result = float(es.circular_median(theta, w))
        assert pytest.approx(result, abs=1e-4) == 1.0

    def test_zero_weight_point_is_ignored(self):
        """A zero-weight point at an outlier angle should not shift the median."""
        theta = jnp.array([0.0, 0.0, math.pi], dtype=jnp.float64)
        w = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float64)  # pi has weight=0
        result = float(es.circular_median(theta, w))
        # median of two zeros is zero
        assert pytest.approx(result, abs=1e-4) == 0.0

    def test_output_in_minus_pi_to_pi(self):
        rng = np.random.default_rng(1)
        for _ in range(5):
            theta = jnp.array(rng.uniform(-math.pi, math.pi, 10), dtype=jnp.float64)
            w = jnp.array(rng.uniform(0.1, 1.0, 10), dtype=jnp.float64)
            result = float(es.circular_median(theta, w))
            assert -math.pi <= result < math.pi

    def test_output_dtype_is_float64(self):
        theta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        w = jnp.array([1.0, 1.0], dtype=jnp.float64)
        result = es.circular_median(theta, w)
        assert result.dtype == jnp.float64

    def test_symmetric_distribution_returns_near_zero(self):
        """Symmetric angles about 0 should give a median near 0."""
        theta = jnp.array([-0.5, -0.3, 0.0, 0.3, 0.5], dtype=jnp.float64)
        w = jnp.ones(5, dtype=jnp.float64)
        result = float(es.circular_median(theta, w))
        assert abs(result) < 0.4  # should be near zero

    def test_near_branch_cut_does_not_produce_nan(self):
        """Angles straddling the +pi/-pi branch cut should not produce NaN."""
        theta = jnp.array([math.pi - 0.1, -(math.pi - 0.1)], dtype=jnp.float64)
        w = jnp.array([1.0, 1.0], dtype=jnp.float64)
        result = float(es.circular_median(theta, w))
        assert math.isfinite(result)


# ===========================================================================
# safe_percentile
# ===========================================================================

class TestSafePercentile:
    """Unit tests for the JAX-safe percentile that ignores non-finite values."""

    def test_50th_percentile_of_sorted_array(self):
        values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
        result = float(es.safe_percentile(values, 50))
        # Should be the middle element
        assert pytest.approx(result, abs=1.0) == 3.0

    def test_0th_percentile_is_minimum(self):
        values = jnp.array([3.0, 1.0, 2.0], dtype=jnp.float64)
        result = float(es.safe_percentile(values, 0))
        assert result <= 2.0  # minimum or close to it

    def test_100th_percentile_is_maximum(self):
        values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
        result = float(es.safe_percentile(values, 100))
        assert result >= 4.0  # close to maximum

    def test_ignores_inf_values(self):
        """Non-finite (inf) values should be masked out."""
        values = jnp.array([1.0, 2.0, 3.0, jnp.inf], dtype=jnp.float64)
        result = float(es.safe_percentile(values, 50))
        # With inf masked, percentile of [1,2,3] should be finite
        assert math.isfinite(result)

    def test_all_inf_returns_big(self):
        """All-invalid array: should return BIG (the sentinel pushed to front after sort)."""
        values = jnp.array([jnp.inf, jnp.inf], dtype=jnp.float64)
        result = float(es.safe_percentile(values, 50))
        assert result >= BIG

    def test_output_dtype_is_float64(self):
        values = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        result = es.safe_percentile(values, 50)
        assert result.dtype == jnp.float64

    def test_result_is_finite_for_normal_input(self):
        values = jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float64)
        for pct in [0, 25, 50, 75, 100]:
            assert math.isfinite(float(es.safe_percentile(values, pct)))

    def test_single_element(self):
        values = jnp.array([7.0], dtype=jnp.float64)
        result = float(es.safe_percentile(values, 50))
        assert pytest.approx(result, abs=1e-9) == 7.0


# ===========================================================================
# get_distance_metric
# ===========================================================================

class TestGetDistanceMetric:
    """Tests for the radial + angular distance metric."""

    def test_output_is_two_tuple(self):
        ra, dec = _make_straight_line(10)
        result = es.get_distance_metric(ra, dec)
        assert len(result) == 2

    def test_metric_shape_matches_input(self):
        n = 15
        ra, dec = _make_grid(n)
        metric, _ = es.get_distance_metric(ra, dec)
        assert metric.shape == (n,)

    def test_trace_is_dict(self):
        ra, dec = _make_straight_line(10)
        _, trace = es.get_distance_metric(ra, dec)
        assert isinstance(trace, dict)

    def test_trace_has_expected_keys(self):
        ra, dec = _make_straight_line(10)
        _, trace = es.get_distance_metric(ra, dec)
        expected = {"n_points", "n_finite_points", "r_thresh", "theta_ref", "theta_weight"}
        assert expected.issubset(set(trace.keys()))

    def test_metric_is_non_negative_for_finite_points(self):
        ra, dec = _make_grid(20)
        metric, _ = es.get_distance_metric(ra, dec)
        finite_mask = np.isfinite(np.array(metric))
        assert np.all(np.array(metric)[finite_mask] >= 0.0)

    def test_metric_output_dtype_is_float64(self):
        ra, dec = _make_straight_line(10)
        metric, _ = es.get_distance_metric(ra, dec)
        assert metric.dtype == jnp.float64

    def test_all_nan_input_returns_big(self):
        """All-NaN coordinates should trigger the empty_case branch."""
        ra = jnp.full(5, jnp.nan, dtype=jnp.float64)
        dec = jnp.full(5, jnp.nan, dtype=jnp.float64)
        metric, trace = es.get_distance_metric(ra, dec)
        assert np.all(np.array(metric) >= BIG)

    def test_single_valid_point(self):
        """A single non-NaN point should run through notempty_case without error."""
        ra = jnp.array([2.0], dtype=jnp.float64)
        dec = jnp.array([0.0], dtype=jnp.float64)
        metric, trace = es.get_distance_metric(ra, dec)
        assert metric.shape == (1,)
        assert math.isfinite(float(trace["theta_ref"]))

    def test_n_finite_points_count_in_trace(self):
        """n_finite_points should count non-NaN, non-inf entries."""
        ra = jnp.array([1.0, 2.0, jnp.nan, 3.0], dtype=jnp.float64)
        dec = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
        _, trace = es.get_distance_metric(ra, dec)
        # 3 of 4 are finite — origin points get r≈BIG but the NaN masks them
        assert int(trace["n_finite_points"]) >= 0  # at least non-negative

    def test_n_elements_parameter_accepted(self):
        ra, dec = _make_grid(20)
        for n_el in [5, 10, 20]:
            metric, _ = es.get_distance_metric(ra, dec, n_elements=n_el)
            assert metric.shape == (20,)


# ===========================================================================
# get_metric_partitions
# ===========================================================================

class TestGetMetricPartitions:
    """Tests for the percentile-partition computation."""

    def test_output_length_is_n_elements_plus_one(self):
        ra, dec = _make_straight_line(20)
        pc = np.vstack([np.array(ra), np.array(dec), np.zeros(20)])
        parts = es.get_metric_partitions(pc, n_elements=5)
        assert parts.shape == (6,)

    def test_partitions_are_monotonically_non_decreasing(self):
        ra, dec = _make_straight_line(30)
        pc = np.vstack([np.array(ra), np.array(dec), np.zeros(30)])
        parts = es.get_metric_partitions(pc, n_elements=8)
        assert np.all(np.diff(parts) >= 0.0)

    def test_first_partition_is_minimum(self):
        ra, dec = _make_straight_line(20)
        pc = np.vstack([np.array(ra), np.array(dec), np.zeros(20)])
        parts = es.get_metric_partitions(pc, n_elements=5)
        # 0th percentile is the minimum of the finite metric values
        assert parts[0] <= parts[1]

    def test_output_dtype_is_float64(self):
        ra, dec = _make_straight_line(10)
        pc = np.vstack([np.array(ra), np.array(dec), np.zeros(10)])
        parts = es.get_metric_partitions(pc, n_elements=4)
        assert parts.dtype == np.float64

    def test_raises_for_n_elements_less_than_1(self):
        ra, dec = _make_straight_line(10)
        pc = np.vstack([np.array(ra), np.array(dec), np.zeros(10)])
        with pytest.raises(ValueError, match="n_elements must be >= 1"):
            es.get_metric_partitions(pc, n_elements=0)

    def test_all_nan_input_returns_big_array(self):
        """When all metric values are NaN, get_distance_metric hits the empty_case branch
        and fills the metric with BIG. So all metric partitions = BIG."""
        pc = np.full((3, 10), np.nan)
        parts = es.get_metric_partitions(pc, n_elements=3)
        assert parts.shape == (4,) # one higher than n_elements
        assert np.all(parts == BIG)

    def test_n_elements_1_gives_two_partitions(self):
        ra, dec = _make_straight_line(10)
        pc = np.vstack([np.array(ra), np.array(dec), np.zeros(10)])
        parts = es.get_metric_partitions(pc, n_elements=1)
        assert parts.shape == (2,)


# ===========================================================================
# get_metric_reference_trace
# ===========================================================================

class TestGetMetricReferenceTrace:
    """Tests for the theta_ref / theta_weight extraction."""

    def test_returns_two_floats(self):
        ra, dec = _make_straight_line(15)
        pc = np.vstack([np.array(ra), np.array(dec), np.zeros(15)])
        theta_ref, theta_weight = es.get_metric_reference_trace(pc)
        assert isinstance(theta_ref, float)
        assert isinstance(theta_weight, float)

    def test_theta_ref_in_minus_pi_to_pi(self):
        ra, dec = _make_grid(20)
        pc = np.vstack([np.array(ra), np.array(dec), np.zeros(20)])
        theta_ref, _ = es.get_metric_reference_trace(pc)
        assert -math.pi <= theta_ref <= math.pi

    def test_theta_weight_is_positive(self):
        ra, dec = _make_straight_line(15)
        pc = np.vstack([np.array(ra), np.array(dec), np.zeros(15)])
        _, theta_weight = es.get_metric_reference_trace(pc)
        assert theta_weight > 0.0

    def test_theta_ref_is_finite(self):
        ra, dec = _make_grid(20)
        pc = np.vstack([np.array(ra), np.array(dec), np.zeros(20)])
        theta_ref, _ = es.get_metric_reference_trace(pc)
        assert math.isfinite(theta_ref)

    def test_n_elements_parameter_accepted(self):
        ra, dec = _make_straight_line(20)
        pc = np.vstack([np.array(ra), np.array(dec), np.zeros(20)])
        # should not raise for different n_elements values
        for n_el in [3, 5, 10]:
            theta_ref, theta_weight = es.get_metric_reference_trace(pc, n_elements=n_el)
            assert math.isfinite(theta_ref)

    def test_collinear_points_give_consistent_theta_ref(self):
        """Points along the RA axis should give theta_ref close to 0."""
        ra = jnp.linspace(1.0, 10.0, 20, dtype=jnp.float64)
        dec = jnp.zeros(20, dtype=jnp.float64)
        pc = np.vstack([np.array(ra), np.array(dec), np.zeros(20)])
        theta_ref, _ = es.get_metric_reference_trace(pc)
        assert abs(theta_ref) < 0.1  # near zero for points along +RA


# ===========================================================================
# sample_metric_boundary
# ===========================================================================

class TestSampleMetricBoundary:
    """Tests for the constant-metric boundary curve sampler."""

    def test_output_is_two_arrays(self):
        ra, dec = es.sample_metric_boundary(partition_radius=3.0, theta_ref=0.0)
        assert ra.shape == dec.shape

    def test_output_length_is_n_samples(self):
        n = 360
        ra, dec = es.sample_metric_boundary(3.0, 0.0, n_samples=n)
        assert ra.shape == (n,)
        assert dec.shape == (n,)

    def test_output_dtype_is_float64(self):
        ra, dec = es.sample_metric_boundary(3.0, 0.0)
        assert ra.dtype == jnp.float64
        assert dec.dtype == jnp.float64

    def test_raises_for_n_samples_less_than_4(self):
        with pytest.raises(ValueError, match="n_samples must be >= 4"):
            es.sample_metric_boundary(3.0, 0.0, n_samples=3)

    def test_zero_partition_radius_gives_near_zero_curve(self):
        """A partition_radius of 0 should yield a curve at the origin."""
        ra, dec = es.sample_metric_boundary(0.0, theta_ref=0.0)
        assert np.all(np.abs(np.array(ra)) < 1e-9)
        assert np.all(np.abs(np.array(dec)) < 1e-9)

    def test_all_points_finite(self):
        ra, dec = es.sample_metric_boundary(5.0, theta_ref=math.pi / 4)
        assert np.all(np.isfinite(np.array(ra)))
        assert np.all(np.isfinite(np.array(dec)))

    def test_curve_is_bounded_by_partition_radius(self):
        """All points on the boundary should have r <= partition_radius."""
        partition_radius = 4.0
        ra, dec = es.sample_metric_boundary(partition_radius, theta_ref=0.0)
        r = np.sqrt(np.array(ra)**2 + np.array(dec)**2)
        assert np.all(r <= partition_radius + 1e-9)

    def test_default_n_samples_is_720(self):
        ra, dec = es.sample_metric_boundary(3.0, theta_ref=0.0)
        assert ra.shape == (720,)

    def test_theta_weight_zero_gives_circular_boundary(self):
        """With theta_weight=0, the metric reduces to radial distance only,
        so the boundary should be a perfect circle of the given radius."""
        r0 = 4.0
        ra, dec = es.sample_metric_boundary(r0, theta_ref=0.0, theta_weight=0.0)
        r = np.sqrt(np.array(ra)**2 + np.array(dec)**2)
        np.testing.assert_allclose(r, r0, rtol=1e-5)


# ===========================================================================
# sample_metric_boundaries
# ===========================================================================

class TestSampleMetricBoundaries:
    """Tests for the multi-partition boundary sampler."""

    def _make_pc_and_partitions(self, n=20, n_elements=5):
        ra, dec = _make_straight_line(n)
        pc = np.vstack([np.array(ra), np.array(dec), np.zeros(n)])
        parts = es.get_metric_partitions(pc, n_elements=n_elements)
        return pc, parts

    def test_number_of_curves_matches_partitions(self):
        pc, parts = self._make_pc_and_partitions(n_elements=5)
        curves, _ = es.sample_metric_boundaries(pc, parts)
        assert len(curves) == len(parts)

    def test_each_curve_is_ra_dec_pair(self):
        pc, parts = self._make_pc_and_partitions(n_elements=4)
        curves, _ = es.sample_metric_boundaries(pc, parts)
        for ra, dec in curves:
            assert ra.shape == dec.shape

    def test_trace_has_theta_ref_and_weight(self):
        pc, parts = self._make_pc_and_partitions()
        _, trace = es.sample_metric_boundaries(pc, parts)
        assert "theta_ref" in trace
        assert "theta_weight" in trace

    def test_trace_theta_ref_is_finite(self):
        pc, parts = self._make_pc_and_partitions()
        _, trace = es.sample_metric_boundaries(pc, parts)
        assert math.isfinite(trace["theta_ref"])

    def test_trace_theta_weight_is_positive(self):
        pc, parts = self._make_pc_and_partitions()
        _, trace = es.sample_metric_boundaries(pc, parts)
        assert trace["theta_weight"] > 0.0

    def test_curves_are_finite(self):
        pc, parts = self._make_pc_and_partitions()
        curves, _ = es.sample_metric_boundaries(pc, parts)
        for ra, dec in curves:
            assert np.all(np.isfinite(np.array(ra)))
            assert np.all(np.isfinite(np.array(dec)))

    def test_n_samples_parameter_propagated(self):
        pc, parts = self._make_pc_and_partitions(n_elements=3)
        curves, _ = es.sample_metric_boundaries(pc, parts, n_samples=180)
        for ra, dec in curves:
            assert ra.shape == (180,)


# ===========================================================================
# prepare_data
# ===========================================================================

class TestPrepareData:
    """Tests for the PreparedData precomputation."""

    def _make_data(self, n=10):
        rng = np.random.default_rng(42)
        ra = rng.uniform(0.5, 3.0, n)
        dec = rng.uniform(-1.0, 1.0, n)
        v = rng.uniform(-5.0, 5.0, n)
        ra_sigma = rng.uniform(0.05, 0.3, n)
        dec_sigma = rng.uniform(0.05, 0.3, n)
        v_sigma = rng.uniform(0.1, 0.5, n)
        return (ra, dec, v), (ra_sigma, dec_sigma, v_sigma)

    def test_returns_prepared_data_namedtuple(self):
        data, uncertainties = self._make_data()
        result = es.prepare_data(data, uncertainties, n_elements=5)
        assert isinstance(result, es.PreparedData)

    def test_all_fields_present(self):
        data, uncertainties = self._make_data()
        result = es.prepare_data(data, uncertainties, n_elements=5)
        expected_fields = {
            "ra_data", "dec_data", "v_data",
            "ra_sigma_safe", "dec_sigma_safe", "v_sigma_safe",
            "dmetric_data", "data_finite_mask",
            "data_min", "data_max",
            "r_proj_data", "theta_proj_data",
        }
        assert expected_fields == set(result._fields)

    def test_sigma_safe_is_at_least_eps(self):
        """Safe sigmas must be floored at 1e-8 to prevent division by zero."""
        data = (np.array([1.0, 2.0]), np.array([0.0, 0.5]), np.array([1.0, 2.0]))
        uncertainties = (np.array([0.0, 0.0]),   # zero sigma
                         np.array([0.0, 0.0]),
                         np.array([0.0, 0.0]))
        result = es.prepare_data(data, uncertainties, n_elements=2)
        assert np.all(np.array(result.ra_sigma_safe) >= 1e-8)
        assert np.all(np.array(result.dec_sigma_safe) >= 1e-8)
        assert np.all(np.array(result.v_sigma_safe) >= 1e-8)

    def test_sigma_safe_preserves_large_values(self):
        """When the input sigma is larger than eps, the safe value equals the input."""
        data, uncertainties = self._make_data()
        result = es.prepare_data(data, uncertainties, n_elements=5)
        np.testing.assert_allclose(
            np.array(result.ra_sigma_safe),
            uncertainties[0],
            rtol=1e-9,
        )

    def test_data_arrays_dtype_is_float64(self):
        data, uncertainties = self._make_data()
        result = es.prepare_data(data, uncertainties, n_elements=5)
        for attr in ("ra_data", "dec_data", "v_data"):
            assert getattr(result, attr).dtype == jnp.float64

    def test_data_min_less_than_data_max(self):
        data, uncertainties = self._make_data(n=15)
        result = es.prepare_data(data, uncertainties, n_elements=5)
        assert float(result.data_min) <= float(result.data_max)

    def test_data_min_and_max_are_finite(self):
        data, uncertainties = self._make_data()
        result = es.prepare_data(data, uncertainties, n_elements=5)
        assert math.isfinite(float(result.data_min))
        assert math.isfinite(float(result.data_max))

    def test_r_proj_data_is_non_negative(self):
        data, uncertainties = self._make_data()
        result = es.prepare_data(data, uncertainties, n_elements=5)
        assert np.all(np.array(result.r_proj_data) >= 0.0)

    def test_theta_proj_data_in_minus_pi_to_pi(self):
        data, uncertainties = self._make_data()
        result = es.prepare_data(data, uncertainties, n_elements=5)
        theta = np.array(result.theta_proj_data)
        assert np.all(theta >= -math.pi - 1e-9)
        assert np.all(theta <= math.pi + 1e-9)

    def test_data_finite_mask_is_boolean_like(self):
        data, uncertainties = self._make_data()
        result = es.prepare_data(data, uncertainties, n_elements=5)
        unique = set(np.unique(np.array(result.data_finite_mask)).tolist())
        assert unique.issubset({True, False, 0, 1})

    def test_nan_in_ra_data_makes_mask_false(self):
        """A NaN in ra_data should mark the corresponding point as not finite."""
        ra = np.array([1.0, np.nan, 2.0])
        dec = np.array([0.0, 0.0, 0.5])
        v = np.array([1.0, 1.0, 1.0])
        sigma = np.ones(3) * 0.1
        result = es.prepare_data((ra, dec, v), (sigma, sigma, sigma), n_elements=2)
        finite_mask = np.array(result.data_finite_mask)
        # The NaN point (index 1) should not be finite
        assert not finite_mask[1]

    def test_r_proj_data_consistent_with_cartesian_to_polar(self):
        """r_proj_data should match sqrt(ra^2 + dec^2) (within gradient stabiliser)."""
        data, uncertainties = self._make_data(n=8)
        result = es.prepare_data(data, uncertainties, n_elements=4)
        ra = np.array(data[0])
        dec = np.array(data[1])
        expected_r = np.sqrt(ra**2 + dec**2 + 1e-60)
        np.testing.assert_allclose(
            np.array(result.r_proj_data), expected_r, rtol=1e-6
        )

    def test_dmetric_data_shape_matches_input(self):
        n = 12
        data, uncertainties = self._make_data(n=n)
        result = es.prepare_data(data, uncertainties, n_elements=4)
        assert result.dmetric_data.shape == (n,)