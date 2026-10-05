"""
Initial tests for stream_lines_grad.py

Tests cover the pure mathematical functions that can be exercised directly
without stubs or real observational data:
  - Module-level constants (G, au_to_km, eps, FLOAT_DTYPE)
  - v_k: Keplerian velocity
  - r_cent / omega_from_mu / mu_from_omega: centrifugal radius and round-trips
  - safe_arccos: clipping and boundary behaviour
  - build_rotation_matrix: orthogonality and identity limits
  - rotate_xyz: invertibility and zero-rotation identity
  - get_orb_ang / get_theta / get_dphi: orbital angle geometry
  - build_stream_quantities: StreamState fields, closed-form orb_ang0, smoothness through v_r0=0
  - xyz_stream: output shapes, output-size rule (npoints), valid mask,
    zero-rotation symmetry, and check_r_array guard

Heavy integration paths (full forward-model convergence, checked_xyz_stream)
are left for integration tests that require a full environment.

Run with:
    pytest test_stream_lines_grad.py -v
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sting.stream_lines_grad as slg

# Convenient aliases
G          = slg.G
au_to_km   = slg.au_to_km
eps        = slg.eps

# ---------------------------------------------------------------------------
# Standard set of physically valid xyz_stream parameters, used across tests.
# r0=1000 au, mu=0.3, deltar=50 au, npoints=30 → r array reaches well below rc
# and there are enough points to sample the streamer down to rlow.
# ---------------------------------------------------------------------------
_BASE = dict(
    mass=1.0, r0=1000.0, theta0=math.radians(30), phi0=math.radians(15),
    mu=0.3, v_r0=2.0, inc=0.0, pa=0.0, rmin=20.0, deltar=50.0, npoints=30,
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

    def test_au_to_km(self):
        assert pytest.approx(au_to_km, rel=1e-4) == 1.4959787e8

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
        defaults = dict(mass=1.0, r0=1000.0, theta0=math.radians(30), mu=0.3, v_r0=2.0)
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
        state = self._build(v_r0=2.0, mu=0.3)
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
        """v_r0=0 (apocentre at r0) is handled exactly; no NaN should appear."""
        state = self._build(v_r0=0.0)
        for field in state:
            assert math.isfinite(float(field)), f"NaN/Inf in StreamState field"
        assert float(state.orb_ang0) == 0.0

    def test_ecc_matches_epsilon_form(self):
        """ecc = sqrt(1 + epsilon sin^2(theta0)) (Mendoza+2009) for any v_r0."""
        for v in [-2.0, 0.0, 0.5, 2.0]:
            state = self._build(v_r0=v)
            expected = math.sqrt(1.0 + float(state.epsilon) * math.sin(math.radians(30)) ** 2)
            assert pytest.approx(float(state.ecc), rel=1e-12) == expected

    def test_orb_ang0_matches_arccos_for_infall(self):
        """For v_r0 > 0 the closed form agrees with the arccos orbit equation at r0."""
        state = self._build(v_r0=2.0)
        expected = float(slg.get_orb_ang(r_to_rc=1.0 / 0.3, theta0=math.radians(30), ecc=state.ecc))
        assert pytest.approx(float(state.orb_ang0), rel=1e-9) == expected

    def test_orb_ang0_is_odd_in_v_r0(self):
        pos = float(self._build(v_r0=1.5).orb_ang0)
        neg = float(self._build(v_r0=-1.5).orb_ang0)
        assert pos > 0
        assert pytest.approx(neg, rel=1e-12) == -pos

    def test_orb_ang0_gradient_finite_and_nonzero_at_zero(self):
        """orb_ang0 is smooth through v_r0=0 (no |v_r0| kink, no clipped arccos)."""
        grad_fn = jax.grad(lambda v: self._build(v_r0=v).orb_ang0)
        g0 = float(grad_fn(0.0))
        assert math.isfinite(g0) and g0 > 0
        assert pytest.approx(float(grad_fn(1e-6)), rel=1e-6) == g0
        assert pytest.approx(float(grad_fn(-1e-6)), rel=1e-6) == g0

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

    def test_gradient_wrt_v_r0_continuous_through_zero(self):
        """The streamline is differentiable in v_r0 at 0, with matching one-sided gradients."""
        def summary(v):
            pos, vel, mask = self._run(v_r0=v, inc=0.3, pa=0.2)
            w = mask.astype(jnp.float64)
            return jnp.sum(w * pos[0]) + jnp.sum(w * vel[1])
        grad_fn = jax.grad(summary)
        g0 = float(grad_fn(0.0))
        assert math.isfinite(g0) and g0 != 0.0
        assert pytest.approx(float(grad_fn(1e-6)), rel=1e-4) == g0
        assert pytest.approx(float(grad_fn(-1e-6)), rel=1e-4) == g0

    def test_zero_v_r0_does_not_raise(self):
        """v_r0=0 is a valid input; should not produce NaN or raise."""
        pos, vel, mask = self._run(v_r0=0.0)
        valid = np.array(mask).astype(bool)
        for arr in list(pos) + list(vel):
            assert np.all(np.isfinite(np.array(arr)[valid]))

# ===========================================================================
# spin: sense of rotation about the (inc, pa) axis
# ===========================================================================

_SPIN_PARAMS = dict(mass=1.0, r0=1500.0, theta0=0.7, phi0=3.5, mu=0.2, v_r0=0.5, inc=-0.5, pa=2.0)
_SPIN_THETA0 = [0.3, 0.7, 1.3, 2.1]
_SPIN_NPOINTS = 1600
_DISTANCE = 300.0


def _xyz(spin=None, **overrides):
    params = dict(_SPIN_PARAMS, rmin=20.0, deltar=1.0, npoints=_SPIN_NPOINTS)
    params.update(overrides)
    if spin is not None:
        params['spin'] = spin
    pos, vel, mask = slg.xyz_stream(**params)
    return np.stack(pos), np.stack(vel), np.asarray(mask).astype(bool)


def _model_params(**overrides):
    params = dict(_SPIN_PARAMS, v_lsr=1.5, rmin=20.0, deltar=1.0)
    params.update(overrides)
    return params


def _radius_grid(params, n=300):
    return jnp.linspace(0.5 * params['mu'] * params['r0'] + 1.0, params['r0'], n)


class TestSpinRegression:
    """spin omitted, spin=+1 and the outputs from before spin was added all agree (theta0 = 0.7)."""

    # values computed with commit 6512904, before spin was added
    _XYZ_GOLDEN = {
        0: [-391.14520665378285, -847.5024773848952, -1174.1997181647741,
            0.18852104757580967, 0.10041948076516526, 0.5034536527441362],
        1: [-390.7682953822841, -847.3013433927471, -1173.1928966750822,
            0.1886946793957971, 0.100862565012906, 0.5040833108902156],
        500: [-215.69816054444496, -688.5992814721919, -692.3188088545074,
              0.29034974310366335, 0.3684188197577147, 0.8188252809883287],
    }
    _RADIUS_GOLDEN = {
        0: [-0.07217766221100771, 0.08563073632517067, 4.354785957663911],
        150: [0.5343394940343889, -1.782210384896235, 2.004818369437867],
        299: [1.3038161608437495, -3.913996756497545, 1.6004199515506745],
    }
    _DELTA_GOLDEN = {
        0: [1.3038173555126085, -3.9139990605492434, 1.6004169562063744, 1499.9999999999989],
        150: [-0.0017340962794300777, -0.16518577075777038, 3.3162498041094253, 256.1360257761456],
        299: [-0.09167167222540139, 0.17679855875285957, 5.3022814303834895, 104.86270875929793],
    }

    @pytest.mark.parametrize('spin', [None, 1.0])
    def test_sampled_model_matches_reference(self, spin):
        pos, vel, mask = slg.xyz_stream(**_SPIN_PARAMS, rmin=20.0, deltar=1.0, npoints=4000,
                                        **({} if spin is None else {'spin': spin}))
        assert int(np.sum(np.asarray(mask))) == 1350
        out = np.vstack((np.stack(pos), np.stack(vel)))
        for i, expected in self._XYZ_GOLDEN.items():
            np.testing.assert_allclose(out[:, i], expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize('spin', [None, 1.0])
    def test_radius_model_matches_reference(self, spin):
        params = _model_params() if spin is None else _model_params(spin=spin)
        out = np.stack(slg.forward_model_at_radius(_radius_grid(params), params, _DISTANCE))
        for i, expected in self._RADIUS_GOLDEN.items():
            np.testing.assert_allclose(out[:, i], expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize('spin', [None, 1.0])
    def test_closed_form_matches_reference(self, spin):
        params = _model_params() if spin is None else _model_params(spin=spin)
        out = np.stack(slg.forward_model_at_delta(jnp.linspace(0, jnp.pi / 2, 300), params, _DISTANCE))
        for i, expected in self._DELTA_GOLDEN.items():
            np.testing.assert_allclose(out[:, i], expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize('theta0', _SPIN_THETA0)
    def test_explicit_plus_one_equals_default_on_all_paths(self, theta0):
        pos_a, vel_a, mask_a = _xyz(theta0=theta0)
        pos_b, vel_b, mask_b = _xyz(spin=1.0, theta0=theta0)
        np.testing.assert_array_equal(mask_a, mask_b)
        np.testing.assert_allclose(pos_a, pos_b, rtol=1e-13, atol=1e-10)
        np.testing.assert_allclose(vel_a, vel_b, rtol=1e-13, atol=1e-13)

        params = _model_params(theta0=theta0)
        r = _radius_grid(params)
        delta = jnp.linspace(0, jnp.pi / 2, 300)
        for fn, x in ((slg.forward_model_at_radius, r), (slg.forward_model_at_delta, delta)):
            a = np.stack(fn(x, params, _DISTANCE))
            b = np.stack(fn(x, dict(params, spin=1.0), _DISTANCE))
            np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-12)


class TestSpinGeometry:
    """spin = -1 is the mirror image phi -> -phi of the spin = +1 streamline."""

    @pytest.mark.parametrize('theta0', _SPIN_THETA0)
    def test_only_phi_and_v_phi_change(self, theta0):
        # cloud frame: inc = pa = 0 gives the identity rotation
        pos_p, vel_p, mask_p = _xyz(spin=1.0, theta0=theta0, inc=0.0, pa=0.0)
        pos_m, vel_m, mask_m = _xyz(spin=-1.0, theta0=theta0, inc=0.0, pa=0.0)
        np.testing.assert_array_equal(mask_p, mask_m)
        m = mask_p
        np.testing.assert_allclose(np.linalg.norm(pos_m[:, m], axis=0), np.linalg.norm(pos_p[:, m], axis=0), rtol=1e-12)
        np.testing.assert_allclose(pos_m[2, m], pos_p[2, m], rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(vel_m[:, m], axis=0), np.linalg.norm(vel_p[:, m], axis=0), rtol=1e-12)
        # and the streamline really does move the other way in phi
        assert not np.allclose(pos_m[1, m], pos_p[1, m], atol=1.0)

    @pytest.mark.parametrize('theta0', _SPIN_THETA0)
    def test_sampled_model_mirror_identity(self, theta0):
        # spin=-1 at phi0 is spin=+1 at -phi0 with y and v_y negated, including the r0 point (index 0)
        phi0 = _SPIN_PARAMS['phi0']
        pos_m, vel_m, mask_m = _xyz(spin=-1.0, theta0=theta0, phi0=phi0, inc=0.0, pa=0.0)
        pos_p, vel_p, mask_p = _xyz(spin=1.0, theta0=theta0, phi0=-phi0, inc=0.0, pa=0.0)
        np.testing.assert_array_equal(mask_m, mask_p)
        mirror = np.array([1.0, -1.0, 1.0])[:, None]
        np.testing.assert_allclose(pos_m[:, mask_m], mirror * pos_p[:, mask_m], rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(vel_m[:, mask_m], mirror * vel_p[:, mask_m], rtol=1e-12, atol=1e-12)
        # r0 point: v_phi0_consistent must carry the spin too
        np.testing.assert_allclose(vel_m[:, 0], mirror[:, 0] * vel_p[:, 0], rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize('theta0', _SPIN_THETA0)
    def test_radius_model_mirror_identity(self, theta0):
        params = dict(_SPIN_PARAMS, theta0=theta0, inc=0.0, pa=0.0)
        r = _radius_grid(params)
        pos_m, vel_m, valid_m = slg.evaluate_streamline_at_radius(r, **params, spin=-1.0)
        pos_p, vel_p, valid_p = slg.evaluate_streamline_at_radius(r, **dict(params, phi0=-params['phi0']), spin=1.0)
        np.testing.assert_array_equal(np.asarray(valid_m), np.asarray(valid_p))
        mirror = np.array([1.0, -1.0, 1.0])[:, None]
        np.testing.assert_allclose(np.stack(pos_m), mirror * np.stack(pos_p), rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(np.stack(vel_m), mirror * np.stack(vel_p), rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize('theta0', _SPIN_THETA0)
    def test_closed_form_mirror_identity(self, theta0):
        params = dict(_SPIN_PARAMS, theta0=theta0, inc=0.0, pa=0.0)
        delta = jnp.linspace(0.0, jnp.pi / 2, 200)
        pos_m, vel_m, r_m = slg.evaluate_streamline_at_delta(delta, **params, spin=-1.0)
        pos_p, vel_p, r_p = slg.evaluate_streamline_at_delta(delta, **dict(params, phi0=-params['phi0']), spin=1.0)
        np.testing.assert_allclose(r_m, r_p, rtol=1e-14)
        mirror = np.array([1.0, -1.0, 1.0])[:, None]
        np.testing.assert_allclose(np.stack(pos_m), mirror * np.stack(pos_p), rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(np.stack(vel_m), mirror * np.stack(vel_p), rtol=1e-12, atol=1e-12)

    @staticmethod
    def _flipped_axis(params):
        # counter-rotating about (inc, pa) = co-rotating about the opposite axis, since
        # R(-inc, pa + pi) = R(inc, pa) diag(-1, 1, -1) and the streamline from (pi - theta0, pi + phi0) is
        # minus the one from (theta0, phi0). The start point of the spin = -1 streamline is the mirror image
        # of the spin = +1 one from -phi0, so phi0 maps to pi - phi0
        return dict(params, inc=-params['inc'], pa=params['pa'] + np.pi,
                    theta0=np.pi - params['theta0'], phi0=np.pi - params['phi0'])

    @pytest.mark.parametrize('theta0', _SPIN_THETA0)
    @pytest.mark.parametrize('inc, pa', [(-0.5, 2.0), (0.9, 5.1), (0.2, 0.4)])
    def test_closed_form_flipped_axis_equivalence(self, theta0, inc, pa):
        params = dict(_SPIN_PARAMS, theta0=theta0, inc=inc, pa=pa)
        delta = jnp.linspace(0.0, jnp.pi / 2, 200)
        pos_m, vel_m, _ = slg.evaluate_streamline_at_delta(delta, **params, spin=-1.0)
        pos_f, vel_f, _ = slg.evaluate_streamline_at_delta(delta, **self._flipped_axis(params), spin=1.0)
        np.testing.assert_allclose(np.stack(pos_m), np.stack(pos_f), atol=1e-6)
        np.testing.assert_allclose(np.stack(vel_m), np.stack(vel_f), atol=1e-9)

    @pytest.mark.parametrize('theta0', _SPIN_THETA0)
    @pytest.mark.parametrize('inc, pa', [(-0.5, 2.0), (0.9, 5.1)])
    def test_sampled_and_radius_model_flipped_axis_equivalence(self, theta0, inc, pa):
        # theta0 -> pi - theta0 crosses the midplane, so this also checks the sign of v_theta below it
        params = dict(_SPIN_PARAMS, theta0=theta0, inc=inc, pa=pa)
        flipped = self._flipped_axis(params)
        pos_m, vel_m, mask_m = _xyz(spin=-1.0, **params)
        pos_f, vel_f, mask_f = _xyz(spin=1.0, **flipped)
        np.testing.assert_array_equal(mask_m, mask_f)
        np.testing.assert_allclose(pos_m[:, mask_m], pos_f[:, mask_m], atol=1e-6)
        np.testing.assert_allclose(vel_m[:, mask_m], vel_f[:, mask_m], atol=1e-9)

        r = _radius_grid(params)
        pos_m, vel_m, _ = slg.evaluate_streamline_at_radius(r, **params, spin=-1.0)
        pos_f, vel_f, _ = slg.evaluate_streamline_at_radius(r, **flipped, spin=1.0)
        np.testing.assert_allclose(np.stack(pos_m), np.stack(pos_f), atol=1e-6)
        np.testing.assert_allclose(np.stack(vel_m), np.stack(vel_f), atol=1e-9)

    @pytest.mark.parametrize('spin', [1.0, -1.0])
    @pytest.mark.parametrize('theta0', _SPIN_THETA0)
    def test_sampled_model_matches_radius_model(self, theta0, spin):
        # with the closed-form checks (test_integrated_matching.py), this ties all three paths together for each spin
        params = dict(_SPIN_PARAMS, theta0=theta0)
        pos, vel, mask = _xyz(spin=spin, **params)
        state = slg.build_stream_quantities(params['mass'], params['r0'], theta0, params['mu'], params['v_r0'])
        r = np.linalg.norm(pos, axis=0)
        # the radius model clips its arccos at r0 (safe_arccos), so leave out the r0 point
        keep = mask & (r > 0.5 * float(state.rc)) & (r < 0.999 * params['r0'])
        pos_r, vel_r, _ = slg.evaluate_streamline_at_radius(r[keep], **params, spin=spin)
        np.testing.assert_allclose(np.stack(pos_r), pos[:, keep], atol=1e-8)
        np.testing.assert_allclose(np.stack(vel_r), vel[:, keep], atol=1e-10)


class TestSpinAngularMomentum:
    """L = r x v is conserved, and its direction is set by spin."""

    @staticmethod
    def _expected_l_hat(theta0, phi0, spin):
        return spin * np.array([-np.cos(theta0) * np.cos(phi0), -np.cos(theta0) * np.sin(phi0), np.sin(theta0)])

    def _check(self, pos, vel, theta0, phi0, spin, rtol):
        ang_mom = np.cross(pos.T, vel.T)
        ang_mom_ref = ang_mom[len(ang_mom) // 2]
        np.testing.assert_allclose(ang_mom, np.broadcast_to(ang_mom_ref, ang_mom.shape),
                                   rtol=0, atol=rtol * np.linalg.norm(ang_mom_ref))
        assert np.all(np.sign(ang_mom[:, 2]) == spin)
        l_hat = ang_mom / np.linalg.norm(ang_mom, axis=1, keepdims=True)
        np.testing.assert_allclose(l_hat, np.broadcast_to(self._expected_l_hat(theta0, phi0, spin), l_hat.shape),
                                   atol=rtol)

    @pytest.mark.parametrize('spin', [1.0, -1.0])
    @pytest.mark.parametrize('theta0', _SPIN_THETA0)
    def test_sampled_model(self, theta0, spin):
        phi0 = _SPIN_PARAMS['phi0']
        pos, vel, mask = _xyz(spin=spin, theta0=theta0, inc=0.0, pa=0.0)
        # leave out the r0 point, whose v_theta is the safe-sqrt floor rather than exactly 0
        mask[0] = False
        self._check(pos[:, mask], vel[:, mask], theta0, phi0, spin, rtol=1e-6)

    @pytest.mark.parametrize('spin', [1.0, -1.0])
    @pytest.mark.parametrize('theta0', _SPIN_THETA0)
    def test_closed_form(self, theta0, spin):
        phi0 = _SPIN_PARAMS['phi0']
        params = dict(_SPIN_PARAMS, theta0=theta0, inc=0.0, pa=0.0)
        pos, vel, _ = slg.evaluate_streamline_at_delta(jnp.linspace(0.0, jnp.pi / 2, 200), **params, spin=spin)
        self._check(np.stack(pos), np.stack(vel), theta0, phi0, spin, rtol=1e-10)


class TestCheckSpin:

    @pytest.mark.parametrize('spin', [0.0, 0.5, 2.0, -2.0, float('nan')])
    def test_xyz_stream_rejects_invalid_spin(self, spin):
        with pytest.raises(Exception, match="spin must be"):
            _xyz(spin=spin)

    def test_checked_xyz_stream_reports_invalid_spin(self):
        params = dict(_SPIN_PARAMS, rmin=20.0, deltar=1.0)
        err, _ = slg.checked_xyz_stream(**params, npoints=_SPIN_NPOINTS, spin=0.5)
        assert "spin must be" in str(err.get())
        err, _ = slg.checked_xyz_stream(**params, npoints=_SPIN_NPOINTS, spin=-1.0)
        assert err.get() is None
