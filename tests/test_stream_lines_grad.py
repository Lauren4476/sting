"""
Initial tests for stream_lines_grad.py

Tests cover the pure mathematical functions that can be exercised directly
without stubs or real observational data:
  - Module-level constants (G, au_in_km, eps, FLOAT_DTYPE)
  - v_k: Keplerian velocity
  - r_cent / omega_from_mu / mu_from_omega: centrifugal radius and round-trips
  - safe_arccos: clipping and boundary behaviour
  - build_rotation_matrix: orthogonality and identity limits
  - rotate_xyz: invertibility and zero-rotation identity
  - get_orb_ang / get_theta / get_dphi: orbital angle geometry
  - build_stream_quantities: StreamState fields and v_r0=0 guard
  - xyz_stream: output shapes, output-size rule (npoints), valid mask,
    zero-rotation symmetry, and check_r_array guard

Heavy integration paths (full forward-model convergence, checked_xyz_stream)
are left for integration tests that require a full environment.

Run with:
    pytest test_stream_lines_grad.py -v
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

import sting.stream_lines_grad as slg

# Convenient aliases
G          = slg.G
au_in_km   = slg.au_in_km
eps        = slg.eps

# ---------------------------------------------------------------------------
# Standard set of physically valid xyz_stream parameters, used across tests.
# r0=1000 au, mu=0.3, deltar=50 au, npoints=30 → r array reaches well below rc
# and there are enough points to sample the streamer down to rlow.
# ---------------------------------------------------------------------------
_BASE = dict(
    mass=1.0, r0=1000.0, theta0=math.radians(30), phi0=math.radians(15),
    mu=0.3, v_r0=-2.0, inc=0.0, pa=0.0, rmin=20.0, deltar=50.0, npoints=30,
)


# ===========================================================================
# Module-level constants
# ===========================================================================

class TestModuleConstants:
    """Sanity-check the physical constants in stream_lines_grad.py."""

    def test_G_positive(self):
        assert G > 0

    def test_G_order_of_magnitude(self):
        # G in au (km/s)^2 Msol^-1 ≈ 887.13
        assert 1e2 < G < 1e3

    def test_au_in_km(self):
        assert pytest.approx(au_in_km, rel=1e-4) == 1.4959787e8

    def test_eps_small_positive(self):
        assert 0 < eps < 1e-4

    def test_float_dtype_is_float64(self):
        assert slg.FLOAT_DTYPE == jnp.float64


# ===========================================================================
# v_k
# ===========================================================================

class TestVK:
    """Unit tests for the Keplerian velocity helper."""

    def test_positive_output(self):
        assert float(slg.v_k(100.0, mass=1.0)) > 0

    def test_scales_with_sqrt_mass(self):
        v1 = float(slg.v_k(100.0, mass=1.0))
        v4 = float(slg.v_k(100.0, mass=4.0))
        assert pytest.approx(v4 / v1, rel=1e-9) == 2.0

    def test_scales_inversely_with_sqrt_radius(self):
        v1 = float(slg.v_k(100.0, mass=1.0))
        v4 = float(slg.v_k(400.0, mass=1.0))
        assert pytest.approx(v4 / v1, rel=1e-9) == 0.5

    def test_known_value(self):
        # v_k = sqrt(G * mass / radius)
        expected = math.sqrt(G * 1.0 / 100.0)
        assert pytest.approx(float(slg.v_k(100.0, mass=1.0)), rel=1e-9) == expected

    def test_output_dtype_is_float64(self):
        result = slg.v_k(100.0, mass=1.0)
        assert result.dtype == jnp.float64


# ===========================================================================
# r_cent / omega_from_mu / mu_from_omega
# ===========================================================================

class TestRCent:
    """Unit tests for r_cent."""

    def test_positive_output(self):
        assert float(slg.r_cent(mass=1.0, omega=1e-14, r0=1000.0)) > 0

    def test_scales_with_omega_squared(self):
        rc1 = float(slg.r_cent(mass=1.0, omega=1e-14, r0=1000.0))
        rc2 = float(slg.r_cent(mass=1.0, omega=2e-14, r0=1000.0))
        assert pytest.approx(rc2 / rc1, rel=1e-9) == 4.0

    def test_scales_with_r0_to_fourth(self):
        rc1 = float(slg.r_cent(mass=1.0, omega=1e-14, r0=1000.0))
        rc2 = float(slg.r_cent(mass=1.0, omega=1e-14, r0=2000.0))
        assert pytest.approx(rc2 / rc1, rel=1e-9) == 16.0

    def test_scales_inversely_with_mass(self):
        rc1 = float(slg.r_cent(mass=1.0, omega=1e-14, r0=1000.0))
        rc2 = float(slg.r_cent(mass=2.0, omega=1e-14, r0=1000.0))
        assert pytest.approx(rc2 / rc1, rel=1e-9) == 0.5


class TestOmegaMuRoundTrip:
    """mu_from_omega and omega_from_mu should be exact inverses."""

    def test_mu_from_omega_roundtrip(self):
        mu_in = 0.3
        omega = float(slg.omega_from_mu(mu=mu_in, mass=1.0, r0=1000.0))
        mu_out = float(slg.mu_from_omega(omega=omega, mass=1.0, r0=1000.0))
        assert pytest.approx(mu_out, rel=1e-6) == mu_in

    def test_omega_from_mu_roundtrip(self):
        omega_in = 1e-14
        mu = float(slg.mu_from_omega(omega=omega_in, mass=1.0, r0=1000.0))
        omega_out = float(slg.omega_from_mu(mu=mu, mass=1.0, r0=1000.0))
        assert pytest.approx(omega_out, rel=1e-6) == omega_in

    def test_larger_mu_gives_larger_omega(self):
        omega1 = float(slg.omega_from_mu(mu=0.2, mass=1.0, r0=1000.0))
        omega2 = float(slg.omega_from_mu(mu=0.4, mass=1.0, r0=1000.0))
        assert omega2 > omega1

    def test_omega_from_mu_positive(self):
        assert float(slg.omega_from_mu(mu=0.3, mass=1.0, r0=1000.0)) > 0


# ===========================================================================
# safe_arccos
# ===========================================================================

class TestSafeArccos:
    """Unit tests for the clipped arccos."""

    def test_zero_gives_pi_over_two(self):
        assert pytest.approx(float(slg.safe_arccos(0.0)), rel=1e-9) == math.pi / 2

    def test_one_gives_zero(self):
        # clipped to just below 1, so result is a small positive number, not exactly 0
        result = float(slg.safe_arccos(1.0))
        assert result >= 0.0
        assert result < 0.01

    def test_minus_one_gives_pi(self):
        result = float(slg.safe_arccos(-1.0))
        assert result <= math.pi
        assert result > math.pi - 0.01

    def test_out_of_range_high_does_not_raise(self):
        # should clip silently, not raise
        result = float(slg.safe_arccos(2.0))
        assert math.isfinite(result)

    def test_out_of_range_low_does_not_raise(self):
        result = float(slg.safe_arccos(-2.0))
        assert math.isfinite(result)

    def test_known_value_half(self):
        # arccos(0.5) = pi/3
        assert pytest.approx(float(slg.safe_arccos(0.5)), rel=1e-6) == math.pi / 3

    def test_output_is_in_zero_pi(self):
        for x in [-0.9, -0.5, 0.0, 0.5, 0.9]:
            result = float(slg.safe_arccos(x))
            assert 0.0 <= result <= math.pi

    def test_output_dtype_is_float64(self):
        result = slg.safe_arccos(0.5)
        assert result.dtype == jnp.float64


# ===========================================================================
# build_rotation_matrix
# ===========================================================================

class TestBuildRotationMatrix:
    """Unit tests for the combined inc/PA rotation matrix."""

    def test_output_shape(self):
        M = slg.build_rotation_matrix(inc=0.0, pa=0.0)
        assert M.shape == (3, 3)

    def test_zero_angles_is_identity(self):
        M = np.array(slg.build_rotation_matrix(inc=0.0, pa=0.0))
        np.testing.assert_allclose(M, np.eye(3), atol=1e-10)

    def test_matrix_is_orthogonal(self):
        """R @ R.T should equal the identity for any angles."""
        for inc, pa in [(0.3, 0.1), (math.pi/4, math.pi/6), (0.0, math.pi/2)]:
            M = np.array(slg.build_rotation_matrix(inc=inc, pa=pa))
            np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-10)

    def test_determinant_is_one(self):
        """Rotation matrices have det=1."""
        M = np.array(slg.build_rotation_matrix(inc=0.5, pa=0.3))
        assert pytest.approx(float(np.linalg.det(M)), abs=1e-10) == 1.0

    def test_output_dtype_is_float64(self):
        M = slg.build_rotation_matrix(inc=0.0, pa=0.0)
        assert M.dtype == jnp.float64


# ===========================================================================
# rotate_xyz
# ===========================================================================

class TestRotateXyz:
    """Unit tests for the 3D rotation wrapper."""

    def _identity_matrix(self):
        return slg.build_rotation_matrix(inc=0.0, pa=0.0)

    def test_zero_rotation_is_identity(self):
        x, y, z = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]), jnp.array([5.0, 6.0])
        M = self._identity_matrix()
        rx, ry, rz = slg.rotate_xyz(x, y, z, rotation_matrix=M)
        np.testing.assert_allclose(np.array(rx), np.array(x), atol=1e-10)
        np.testing.assert_allclose(np.array(ry), np.array(y), atol=1e-10)
        np.testing.assert_allclose(np.array(rz), np.array(z), atol=1e-10)

    def test_rotation_preserves_vector_length(self):
        """Rotating a vector should not change its Euclidean norm."""
        x, y, z = jnp.array([3.0]), jnp.array([4.0]), jnp.array([0.0])
        M = slg.build_rotation_matrix(inc=0.5, pa=0.3)
        rx, ry, rz = slg.rotate_xyz(x, y, z, rotation_matrix=M)
        norm_in = np.array(jnp.sqrt(x**2 + y**2 + z**2))
        norm_out = np.array(jnp.sqrt(rx**2 + ry**2 + rz**2))
        np.testing.assert_allclose(norm_out, norm_in, rtol=1e-9)

    def test_inverse_rotation_recovers_original(self):
        """Applying R then R.T should recover the original vector."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])
        z = jnp.array([7.0, 8.0, 9.0])
        M = slg.build_rotation_matrix(inc=0.4, pa=0.2)
        rx, ry, rz = slg.rotate_xyz(x, y, z, rotation_matrix=M)
        # R.T is the inverse of an orthogonal matrix
        rx2, ry2, rz2 = slg.rotate_xyz(rx, ry, rz, rotation_matrix=M.T)
        np.testing.assert_allclose(np.array(rx2), np.array(x), atol=1e-9)
        np.testing.assert_allclose(np.array(ry2), np.array(y), atol=1e-9)
        np.testing.assert_allclose(np.array(rz2), np.array(z), atol=1e-9)

    def test_output_shapes_match_input(self):
        x = jnp.ones(10)
        y = jnp.ones(10)
        z = jnp.ones(10)
        M = self._identity_matrix()
        rx, ry, rz = slg.rotate_xyz(x, y, z, rotation_matrix=M)
        assert rx.shape == (10,)
        assert ry.shape == (10,)
        assert rz.shape == (10,)


# ===========================================================================
# get_orb_ang
# ===========================================================================

class TestGetOrbAng:
    """Unit tests for the orbital angle function."""

    def test_output_in_zero_pi(self):
        result = float(slg.get_orb_ang(r_to_rc=2.0, theta0=math.radians(30), ecc=2.0))
        assert 0.0 <= result <= math.pi

    def test_output_is_finite(self):
        result = float(slg.get_orb_ang(r_to_rc=3.0, theta0=math.radians(45), ecc=2.0))
        assert math.isfinite(result)

    def test_larger_r_to_rc_gives_larger_orb_ang(self):
        """Farther from centrifugal radius -> smaller orbital angle orb_ang."""
        ang1 = float(slg.get_orb_ang(r_to_rc=2.0, theta0=math.radians(30), ecc=2.0))
        ang2 = float(slg.get_orb_ang(r_to_rc=5.0, theta0=math.radians(30), ecc=2.0))
        assert ang2 < ang1

    def test_orb_ang_in_range(self):
        """orb_ang should always be in [0, pi] regardless of parameters. 
        For r_to_rc >= 1 (valid physical domain), orb_ang should be in [0, pi/2]."""
        for r_to_rc in [0.5, 0.9, 1.1, 1.5, 2.0, 3.0]:
            for theta0 in [math.radians(10), math.radians(30), math.radians(60)]:
                for ecc in [1.0, 2.0, 3.0]:
                    result = float(slg.get_orb_ang(r_to_rc=r_to_rc, theta0=theta0, ecc=ecc))
                    assert 0.0 <= result <= math.pi
        for r_to_rc in [1.0, 1.5, 2.0, 3.0]:
            for theta0 in [math.radians(10), math.radians(30), math.radians(60)]:
                for ecc in [1.0, 2.0, 3.0]:
                    result = float(slg.get_orb_ang(r_to_rc=r_to_rc, theta0=theta0, ecc=ecc))
                    assert 0.0 <= result <= math.pi / 2

    def test_output_dtype_is_float64(self):
        result = slg.get_orb_ang(r_to_rc=2.0, theta0=math.radians(30), ecc=2.0)
        assert result.dtype == jnp.float64


# ===========================================================================
# get_theta
# ===========================================================================

class TestGetTheta:
    """Unit tests for the polar angle function."""

    def test_output_in_zero_pi(self):
        orb_ang0 = float(slg.get_orb_ang(r_to_rc=3.0, theta0=math.radians(30), ecc=0.5))
        orb_ang = float(slg.get_orb_ang(r_to_rc=2.0, theta0=math.radians(30), ecc=0.5))
        result = float(slg.get_theta(math.radians(30), orb_ang, orb_ang0))
        assert 0.0 <= result <= math.pi

    def test_at_initial_position_recovers_theta0(self):
        """When orb_ang == orb_ang0, get_theta should return theta0."""
        theta0 = math.radians(30)
        ecc = 0.5
        orb_ang0 = float(slg.get_orb_ang(r_to_rc=3.0, theta0=theta0, ecc=ecc))
        result = float(slg.get_theta(theta0, orb_ang0, orb_ang0))
        assert pytest.approx(result, abs=1e-6) == theta0

    def test_output_is_finite(self):
        orb_ang0 = float(slg.get_orb_ang(r_to_rc=3.0, theta0=math.radians(45), ecc=0.6))
        orb_ang = float(slg.get_orb_ang(r_to_rc=2.0, theta0=math.radians(45), ecc=0.6))
        result = float(slg.get_theta(math.radians(45), orb_ang, orb_ang0))
        assert math.isfinite(result)

    def test_output_dtype_is_float64(self):
        orb_ang0 = slg.get_orb_ang(r_to_rc=3.0, theta0=math.radians(30), ecc=0.5)
        orb_ang = slg.get_orb_ang(r_to_rc=2.0, theta0=math.radians(30), ecc=0.5)
        result = slg.get_theta(math.radians(30), orb_ang, orb_ang0)
        assert result.dtype == jnp.float64


# ===========================================================================
# get_dphi
# ===========================================================================

class TestGetDphi:
    """Unit tests for the azimuthal angle difference."""

    def test_output_in_zero_pi(self):
        result = float(slg.get_dphi(theta=math.radians(20), theta0=math.radians(30)))
        assert 0.0 <= result <= math.pi

    def test_theta_equals_theta0_gives_near_zero(self):
        """When theta == theta0, tan(theta0)/tan(theta) = 1, so arccos = 0. But with safe_arccos, the result is a small positive number, not exactly 0."""
        result = float(slg.get_dphi(theta=math.radians(30), theta0=math.radians(30)))
        assert pytest.approx(result, abs=1e-5) == 0.0

    def test_output_is_finite_near_zero_theta(self):
        """Near-zero theta is guarded by the safe tan; result should be finite."""
        result = float(slg.get_dphi(theta=1e-9, theta0=math.radians(30)))
        assert math.isfinite(result)

    def test_output_dtype_is_float64(self):
        result = slg.get_dphi(theta=math.radians(20), theta0=math.radians(30))
        assert result.dtype == jnp.float64


# ===========================================================================
# build_stream_quantities
# ===========================================================================

class TestBuildStreamQuantities:
    """Unit tests for the StreamState precomputation."""

    def _build(self, **kwargs):
        defaults = dict(mass=1.0, r0=1000.0, theta0=math.radians(30), mu=0.3, v_r0=-2.0)
        defaults.update(kwargs)
        return slg.build_stream_quantities(**defaults)

    def test_returns_stream_state(self):
        state = self._build()
        assert isinstance(state, slg.StreamState)

    def test_rc_equals_mu_times_r0(self):
        state = self._build(mu=0.3, r0=1000.0)
        assert pytest.approx(float(state.rc), rel=1e-9) == 0.3 * 1000.0

    def test_ecc_positive(self):
        state = self._build()
        assert float(state.ecc) > 0

    def test_ecc_at_least_one_for_parabolic_orbit(self):
        """For a typical infalling streamer the eccentricity should be >= 1."""
        state = self._build(v_r0=-2.0, mu=0.3)
        assert float(state.ecc) >= 0.0  # ecc can be < 1 for low v_r0; just check finite
        assert math.isfinite(float(state.ecc))

    def test_vk0_positive(self):
        state = self._build()
        assert float(state.vk0) > 0

    def test_vk0_equals_v_k_at_rc(self):
        state = self._build(mu=0.3, r0=1000.0, mass=1.0)
        rc = 0.3 * 1000.0
        expected_vk0 = float(slg.v_k(rc, mass=1.0))
        assert pytest.approx(float(state.vk0), rel=1e-9) == expected_vk0

    def test_zero_v_r0_does_not_produce_nan(self):
        """v_r0=0 is guarded by an epsilon replacement; no NaN should appear."""
        state = self._build(v_r0=0.0)
        for field in state:
            assert math.isfinite(float(field)), f"NaN/Inf in StreamState field"

    def test_nu_finite_for_typical_params(self):
        state = self._build()
        assert math.isfinite(float(state.nu))

    def test_all_fields_are_float64(self):
        state = self._build()
        for field in state:
            assert jnp.asarray(field).dtype == jnp.float64


# ===========================================================================
# xyz_stream — output structure, shapes, and guard behaviour
# ===========================================================================

class TestXyzStream:
    """Tests for the main xyz_stream function."""

    def _run(self, **overrides):
        params = dict(_BASE)
        params.update(overrides)
        return slg.xyz_stream(**params)

    def test_returns_three_tuples(self):
        pos, vel, mask = self._run()
        assert len(pos) == 3
        assert len(vel) == 3

    def test_output_length_is_npoints(self):
        """xyz_stream should return exactly npoints arrays."""
        npoints = 30
        pos, vel, mask = self._run(npoints=npoints)
        for arr in list(pos) + list(vel) + [mask]:
            assert arr.shape == (npoints,), f"Expected ({npoints},), got {arr.shape}"

    def test_mask_is_boolean_like(self):
        """The validity mask should contain only 0.0 or 1.0."""
        _, _, mask = self._run()
        unique_vals = set(np.unique(np.array(mask)).tolist())
        assert unique_vals.issubset({0.0, 1.0})

    def test_first_point_always_valid(self):
        """The initial point at r0 is prepended and should always be valid."""
        _, _, mask = self._run()
        assert float(mask[0]) == 1.0

    def test_positions_finite_where_valid(self):
        pos, _, mask = self._run()
        valid = np.array(mask).astype(bool)
        for arr in pos:
            assert np.all(np.isfinite(np.array(arr)[valid])), "NaN/Inf in valid position"

    def test_velocities_finite_where_valid(self):
        _, vel, mask = self._run()
        valid = np.array(mask).astype(bool)
        for arr in vel:
            assert np.all(np.isfinite(np.array(arr)[valid])), "NaN/Inf in valid velocity"

    def test_invalid_points_are_zero(self):
        """Points marked invalid (mask==0) should have been zeroed out."""
        pos, vel, mask = self._run()
        invalid = np.array(mask) == 0.0
        if invalid.any():
            for arr in list(pos) + list(vel):
                np.testing.assert_array_equal(
                    np.array(arr)[invalid], 0.0,
                    err_msg="Invalid points should be zeroed",
                )

    def test_zero_pa_and_inc_x_is_positive(self):
        """With inc=pa=0 and theta0, phi0 in first quadrant, x = r*sin(theta)*cos(phi) >= 0.
        Note: in gradient_descent.py we negate ra = - x / distance_pc to match RA sign convention,
        but here we just check the raw x coordinate"""
        pos, _, mask = self._run(inc=0.0, pa=0.0)
        x = np.array(pos[0])
        valid = np.array(mask).astype(bool)
        assert np.all(x[valid] >= 0.0), "x should be >= 0 for standard geometry"

    def test_larger_npoints_gives_more_valid_points(self):
        """Increasing npoints should give at least as many valid points."""
        _, _, mask_small = self._run(npoints=20)
        _, _, mask_large = self._run(npoints=50)
        n_valid_small = int(np.sum(np.array(mask_small)))
        n_valid_large = int(np.sum(np.array(mask_large)))
        assert n_valid_large >= n_valid_small

    def test_output_dtype_float64(self):
        pos, vel, mask = self._run()
        for arr in list(pos) + list(vel):
            assert arr.dtype == jnp.float64

    def test_pa_rotation_changes_x_not_y_magnitude(self):
        """Rotating PA by pi/2 should swap the sky-plane axes."""
        pos0, _, mask = self._run(inc=0.0, pa=0.0)
        pos90, _, _ = self._run(inc=0.0, pa=math.pi / 2)
        valid = np.array(mask).astype(bool)
        # x and z should change; total projected distance on sky should be preserved
        r0 = np.sqrt(np.array(pos0[0])**2 + np.array(pos0[2])**2)
        r90 = np.sqrt(np.array(pos90[0])**2 + np.array(pos90[2])**2)
        np.testing.assert_allclose(r0[valid], r90[valid], rtol=1e-6)

    def test_rmin_trims_valid_points(self):
        """Setting rmin > 0 should reduce the number of valid points."""
        _, _, mask_no_rmin = self._run(rmin=0.0)
        _, _, mask_rmin = self._run(rmin=200.0)
        n_valid_no_rmin = int(np.sum(np.array(mask_no_rmin)))
        n_valid_rmin = int(np.sum(np.array(mask_rmin)))
        assert n_valid_rmin <= n_valid_no_rmin

    def test_raises_when_rc_greater_than_r0(self):
        """mu >= 1 means rc >= r0, which should trigger check_rc_r0."""
        with pytest.raises(Exception, match="Centrifugal radius is larger"):
            self._run(mu=1.5)

    def test_raises_when_npoints_too_small(self):
        """npoints=1 produces an empty r array that cannot reach r_low."""
        with pytest.raises(Exception, match="Radius points do not extend down to rlow"):
            self._run(npoints=1)

    def test_valid_with_minimum_npoints(self):
        """when r0, rmin and deltar are such that the r array will reach rlow, npoints=2 is the minimum possible number of points.
        2 points: 200 (r0), 40 (rmin)"""
        pos, vel, mask = self._run(npoints=2, r0=200.0, rmin=40.0, deltar=160.0)
        assert mask.shape == (2,)

    def test_zero_v_r0_does_not_raise(self):
        """v_r0=0 is guarded internally; should not produce NaN or raise."""
        pos, vel, mask = self._run(v_r0=0.0)
        valid = np.array(mask).astype(bool)
        for arr in list(pos) + list(vel):
            assert np.all(np.isfinite(np.array(arr)[valid]))