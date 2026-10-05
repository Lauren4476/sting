'''
Contains all functions needed for a forward model of a streamline.
All are fully differentiable using JAX

Streamline implementation is based on Mendoza et al. (2009) doi:10.1111/j.1365-2966.2008.14210.x

The assumed input units are:
- distance on sky: au
- velocity: km/s
- mass: solar masses
- angles (PA, i, theta, phi...): radians
- rc (centrifugal radius): au (alternative to omega - either can be used to calculate mu=rc/r0)
- Omega (angular velocity): 1/ (alternative to rc - either can be used to calculate mu=rc/r0)
- distance to source: pc

omega, rc and mu are magnitudes and must be positive.
spin (+1 or -1) is the sense of rotation about the axis set by (inc, pa). +1 is the Mendoza
convention (motion in the positive phi direction) and -1 is counter-rotating. Only phi and v_phi
depend on it: the counter-rotating streamline is the mirror image phi -> -phi of the standard one.

Last updated: 02-10-2026
'''


import astropy.units as u
import jax
import jax.numpy as jnp
from jax.experimental import checkify
jax.config.update("jax_enable_x64", True)
from typing import NamedTuple


## constants 
eps = 1e-8 # small value to avoid division by zero
FLOAT_DTYPE = jnp.float64
G = 6.67430e-11 * (1e-3)**2 * (1.988416e30) / (1.4959787e11) # in au (km/s)^2 * Msol^-1
au_to_km = 1.4959787e8 #km


## important streamline quantities (for easy reuse)
class StreamState(NamedTuple):
    rc: jnp.ndarray
    mu: jnp.ndarray
    nu: jnp.ndarray
    epsilon: jnp.ndarray
    ecc: jnp.ndarray
    vk0: jnp.ndarray
    orb_ang0: jnp.ndarray


@jax.jit
def to_float64(value):
    '''input must be a number or array-like'''
    return jnp.asarray(value, dtype=FLOAT_DTYPE)

@jax.jit
def v_k(radius, mass=0.5):
    '''
    Velocity term that is repeated in all velocity components.
    It corresponds to v_k in Mendoza+(2009)
    :param radius: au
    :param mass: Msun
    :return: v_k, km/s
    '''
    arg = G * mass / radius
    return jnp.sqrt(arg)

@jax.jit
def r_cent(mass, omega=1e-14, r0=1e4):
    '''
    Centrifugal radius or disk radius in the Ulrich (1976) model.
    r_u in Mendoza's nomenclature.

    :param mass: Central mass for the protostar, Msun
    :param omega: Angular speed at the r0 radius, 1/s
    :param r0: Initial radius of the streamline, au
    :return: r_cent, au
    '''
    r_cent = (jnp.power(r0, 4) * jnp.power(omega, 2) / (G * mass)) # in au^3 km^-2
    r_cent_au = r_cent * (jnp.power(au_to_km, 2)) # in au
    return r_cent_au

@jax.jit
def omega_from_mu(mu, mass, r0):
    omega_squared = mu * G * mass / jnp.power(r0, 3) # in km^2 au^-3
    omega = jnp.sqrt(omega_squared) / au_to_km # in 1/s
    return omega

@jax.jit
def mu_from_omega(omega, mass, r0):
    rc = r_cent(mass=mass, omega=omega, r0=r0)
    mu = rc / r0
    return mu


@jax.jit
def build_stream_quantities(mass, r0, theta0, mu, v_r0):
    '''
    precompute streamer quantities reused throughout file, and
    store in class StreamState (near top)

    v_r0 sign convention: v_r0 > 0 is infall at r0, v_r0 = 0 means r0 is the
    apocentre, and v_r0 < 0 means the gas is still moving outwards at r0 (it
    turns around at an apocentre beyond r0 before falling in). The fit restricts
    v_r0 >= 0, but the smooth continuation to v_r0 < 0 is what makes the gradient
    well-defined at v_r0 = 0.

    The initial orbital angle is computed in closed form rather than from
    arccos(cos(orb_ang0)), since
        e cos(orb_ang0) = 1 - mu sin^2(theta0)
        e sin(orb_ang0) = nu sin(theta0)
    (from the orbit equation and v_r at r0). This makes orb_ang0 an odd, smooth
    function of v_r0, so the model is differentiable through v_r0 = 0.
    The arccos form gives orb_ang0 ~ |v_r0|, which has a kink at v_r0 = 0.
    '''
    mu = to_float64(mu)
    rc = mu * r0
    nu = v_r0 * jnp.sqrt(rc / (G * mass))
    sin_theta0 = jnp.sin(theta0)
    sin_theta0_sq = jnp.power(sin_theta0, 2)
    epsilon = jnp.power(nu, 2) + jnp.power(mu, 2) * sin_theta0_sq - 2 * mu

    # e^2 = 1 + epsilon sin^2(theta0) = (nu sin(theta0))^2 + (1 - mu sin^2(theta0))^2
    # the sum-of-squares form is strictly positive for mu < 1, so gradients are finite at v_r0 = 0
    ecc_sin_orb_ang0 = nu * sin_theta0
    ecc_cos_orb_ang0 = 1.0 - mu * sin_theta0_sq
    ecc = jnp.sqrt(jnp.power(ecc_sin_orb_ang0, 2) + jnp.power(ecc_cos_orb_ang0, 2))
    orb_ang0 = jnp.arctan2(ecc_sin_orb_ang0, ecc_cos_orb_ang0)
    vk0 = v_k(rc, mass=mass)

    return StreamState(rc=rc, mu=mu, nu=nu, epsilon=epsilon, ecc=ecc, vk0=vk0, orb_ang0=orb_ang0)

@jax.jit
def safe_arccos(x, eps=1e-10):
    '''
    Safe arccos function with clipping to valid range [-1, 1],
    with a small margin to avoid numerical issues in gradients near the boundaries

    :param x: input value
    :param eps: small offset
    :return: arccos of clipped input
    '''
    x = jnp.asarray(x)
    x = x.astype(FLOAT_DTYPE)

    # Keep away from +/-1 by at least a few ULPs of the active dtype
    eps_user = jnp.asarray(eps, dtype=x.dtype)
    eps_floor = jnp.asarray(32.0 * jnp.finfo(x.dtype).eps, dtype=x.dtype)
    eps_eff = jnp.maximum(eps_user, eps_floor)

    x_safe = jnp.clip(x, -1.0 + eps_eff, 1.0 - eps_eff)
    return jnp.arccos(x_safe)

@jax.jit
def get_theta(theta0, orb_ang, orb_ang0):
    '''
    Gets theta from theta0, orb_ang, and orb_ang0, in radians.
    Eqn (8) in Mendoza+2009
    
    :param theta0: radians
    :param orb_ang: radians
    :param orb_ang0: radians
    :return theta: radians
    '''
    cos_theta = jnp.cos(theta0) * jnp.cos(orb_ang - orb_ang0)
    theta = safe_arccos(cos_theta)
    return theta


@jax.jit
def get_orb_ang(r_to_rc, theta0, ecc):
    '''
    Gets orb_ang (varphi in Mendoza+2009), in radians.
    To get initial orb_ang, set r_to_rc = r0/rc = 1/mu
    
    :param r_to_rc: radius divided by centrifugal radius
    :param theta0: radius
    :param ecc: eccentricity
    :return orb_ang: radians
    '''
    cos_orb_ang = (1/ecc) * (1 - (jnp.power(jnp.sin(theta0), 2) / r_to_rc))
    orb_ang = safe_arccos(cos_orb_ang)
    return orb_ang

@jax.jit
def get_dphi(theta, theta0=jnp.radians(30)):
    '''
    Gets the difference in Phi between initial and current, in radians.

    :param theta: radians
    :param theta0: radians
    :return: difference in Phi angle, radians
    '''
    small_eps = to_float64(1e-12) # need a very small eps since tan(theta) can be very small for theta near 0 or pi
    tan_theta_safe = jnp.where(
        jnp.abs(jnp.tan(theta)) > small_eps,
        jnp.tan(theta),
        jnp.sign(jnp.tan(theta)) * small_eps,
    )
    # handle exact zero case
    tan_theta_safe = jnp.where(tan_theta_safe == 0.0, small_eps, tan_theta_safe)
    arg = jnp.tan(theta0) / tan_theta_safe
    return safe_arccos(arg, eps=small_eps)



@jax.jit
def stream_line(r, r_mask, stream_state, theta0=jnp.radians(30), phi0=jnp.radians(15), spin=1.0):
    '''
    It calculates the stream line following Mendoza et al. (2009),
    only for r < r0. Point r = r0 is handled outside the function.
    It takes the radial velocity and rotation at the streamline
    initial radius and it describes the entire trajectory.

    :param r: au
    :param r_mask: boolean mask for valid r values (r < r0 and r > 0.5*rc)
    :param stream_state: StreamState named tuple containing precomputed quantities for the streamline
    :param theta0: radians
    :param phi0: radians
    :param spin: +1 (co-rotating, Mendoza convention) or -1 (counter-rotating) about the rotation axis
    :return: theta, radians
    '''
    r = jnp.asarray(r, dtype=FLOAT_DTYPE)
    rc = stream_state.rc
    ecc = stream_state.ecc

    # orb_ang is varphi in Mendoza+2009
    orb_ang0 = stream_state.orb_ang0

    r_to_rc_raw = r / rc
    r_to_rc = jnp.where(r_mask, r_to_rc_raw, to_float64(0.6))

    orb_ang = get_orb_ang(r_to_rc=r_to_rc, theta0=theta0, ecc=ecc)
    theta = get_theta(theta0, orb_ang, orb_ang0)
    # get_dphi is always >= 0, so spin sets the direction of motion in phi
    phi = phi0 + spin * get_dphi(theta, theta0=theta0)

    # remove values where r_to_rc < 0.5 (inside centrifugal radius)
    # this will include all the mask points, and also any points that are inside 0.5*rc
    valid_mask = r_mask & (r_to_rc >= 0.5)

    # safe sentinel values for invalid points, to make sure gradients are finite
    # will be masked out later in final output
    orb_ang = jnp.where(valid_mask, orb_ang, jnp.pi/4)
    theta_sentinel = jnp.minimum(theta0 + to_float64(0.1), to_float64(jnp.pi) - to_float64(eps))
    theta = jnp.where(valid_mask, theta, theta_sentinel)
    phi = jnp.where(valid_mask, phi, phi0)

    return orb_ang, theta, phi, valid_mask #in radians


@jax.jit
def stream_line_vel(
    r,
    theta,
    orb_ang,
    stream_state,
    theta0=jnp.radians(30),
    r_mask=None,
    spin=1.0,
):
    '''
    It calculates the velocity along the stream line following Mendoza+(2009)
    It takes the radial velocity and rotation at the streamline
    initial radius and it describes the entire trajectory.

    :param theta: radians
    :param r: au
    :param stream_state: StreamState named tuple containing precomputed quantities for the streamline
    :param theta0: radians
    :param r_mask: boolean mask
    :param spin: +1 (co-rotating, Mendoza convention) or -1 (counter-rotating). Sets the sign of v_phi
    :return: v_r, v_theta, v_phi in units of km/s
    '''
    rc = stream_state.rc
    ecc = stream_state.ecc
    vk0 = stream_state.vk0

    r_to_rc_raw = r / rc
    if r_mask is not None:
        # see sentnel value used for r_to_rc in stream_line. this is the same thing.
        r_to_rc = jnp.where(r_mask, r_to_rc_raw, to_float64(0.6))
    else:
        r_to_rc = r_to_rc_raw
    #
    v_r_all = -ecc * jnp.sin(theta0) * jnp.sin(orb_ang) / r_to_rc /(1 - ecc*jnp.cos(orb_ang))
    sqrt_arg = jnp.power(jnp.cos(theta0), 2) - jnp.power(jnp.cos(theta), 2)
    sqrt_arg_safe = jnp.maximum(sqrt_arg, eps)  # eps = 1e-8 or similar

    # the gas always moves towards the midplane: theta increases for theta0 < pi/2 and decreases for
    # theta0 > pi/2, so v_theta takes the sign of cos(theta0) (the sqrt only gives its magnitude)
    v_theta_all = jnp.sign(jnp.cos(theta0)) * jnp.sin(theta0) / jnp.sin(theta) / r_to_rc \
                  * jnp.sqrt(sqrt_arg_safe)
    v_phi_all = spin * jnp.power(jnp.sin(theta0), 2) / (jnp.sin(theta) * r_to_rc)

    return v_r_all * vk0, v_theta_all * vk0, v_phi_all * vk0

@jax.jit
def build_rotation_matrix(inc, pa):
    '''constructs combined inclination/position-angle rotation matrix'''

    inc = jnp.asarray(inc, dtype=FLOAT_DTYPE)
    pa = jnp.asarray(pa, dtype=FLOAT_DTYPE)

    ci = jnp.cos(inc)
    si = jnp.sin(inc)
    cp = jnp.cos(pa)
    sp = jnp.sin(pa)

    return jnp.array([
        [cp, sp * si, -sp * ci],
        [0.0, ci, si],
        [sp, -cp * si, cp * ci],
    ], dtype=FLOAT_DTYPE)

@jax.jit
def rotate_xyz(x, y, z, rotation_matrix):
    '''
    Rotate on inclination and PA
    x-axis and y-axis are on the plane on the sky,
    z-axis is the

    Rotation around x is inclination angle
    Rotation around y is PA angle

    Using example matrices as described in:
    https://en.wikipedia.org/wiki/3D_projection

    :param x: cartesian x-coordinate, in the direction of decreasing RA
    :param y: cartesian y-coordinate, in the direction away of the observer
    :param z: cartesian z-coordinate, in the direction of increasing Dec.
    :param rotation_matrix: 3x3 rotation matrix combining inclination and PA rotations.
    :return: new x, y, and z-coordinates as observed on the sky, with the
    same units as the input ones.

    '''
    x = jnp.asarray(x, dtype=FLOAT_DTYPE)
    y = jnp.asarray(y, dtype=FLOAT_DTYPE)
    z = jnp.asarray(z, dtype=FLOAT_DTYPE)

    xyz = jnp.stack((x, y, z), axis=0)

    xyz_rot = rotation_matrix @ xyz

    return xyz_rot[0], xyz_rot[1], xyz_rot[2]

def check_rc_r0(rc, r0):
    '''check that centrifugal radius is smaller than initial radius of streamline, otherwise the model is not valid'''
    checkify.check(
        rc < r0,
        "Centrifugal radius is larger than start of streamline. Model is not valid."
    )

def check_spin(spin):
    '''check that spin is exactly +1 or -1'''
    checkify.check(jnp.abs(spin) == 1.0,
                   "spin must be +1 (co-rotating) or -1 (counter-rotating).")

def check_r_array(r, r_low):
    '''check that radius array extends down to r_low, otherwise the model doesn't extend far enough for the given npoints and deltar'''
    r_small = r <= r_low
    # jax.debug.print("r_small: {}", r_small)
    # # print how many of r_small are True
    # jax.debug.print("Number of points below r_low: {}", jnp.sum(r_small))
    # jax.debug.print("r_low: {}", r_low)
    checkify.check(
        jnp.any(r_small),
        f"Radius points do not extend down to rlow. Increase npoints and/or deltar"
    )

@jax.jit
def evaluate_streamline_at_radius(r, mass, r0, theta0, phi0, mu, v_r0, inc, pa, spin=1.0):
    '''Evaluate the analytic Mendoza streamline at arbitrary radius or radius array.

    This is the continuous analogue of the sampled xyz_stream() model: it computes
    spherical geometry, cartesian coordinates, and rotated sky-plane projections
    for a scalar or array of radii without constructing a sampling grid.
    spin is +1 (co-rotating, Mendoza convention) or -1 (counter-rotating) about the rotation axis.
    '''
    r = jnp.asarray(r, dtype=FLOAT_DTYPE)
    mass = jnp.asarray(mass, dtype=FLOAT_DTYPE)
    r0 = jnp.asarray(r0, dtype=FLOAT_DTYPE)
    theta0 = jnp.asarray(theta0, dtype=FLOAT_DTYPE)
    phi0 = jnp.asarray(phi0, dtype=FLOAT_DTYPE)
    mu = jnp.asarray(mu, dtype=FLOAT_DTYPE)
    v_r0 = jnp.asarray(v_r0, dtype=FLOAT_DTYPE)
    inc = jnp.asarray(inc, dtype=FLOAT_DTYPE)
    pa = jnp.asarray(pa, dtype=FLOAT_DTYPE)
    spin = jnp.asarray(spin, dtype=FLOAT_DTYPE)

    stream_state = build_stream_quantities(mass=mass, r0=r0, theta0=theta0, mu=mu, v_r0=v_r0)
    rc = stream_state.rc
    ecc = stream_state.ecc
    rotation_matrix = build_rotation_matrix(inc, pa)

    r_low = jnp.maximum(to_float64(0.0), rc * to_float64(0.5))
    r_valid = (r >= r_low) & (r <= r0) & (r > 0.0)
    r_for_eval = jnp.where(r_valid, r, r_low)

    orb_ang0 = stream_state.orb_ang0
    orb_ang = get_orb_ang(r_to_rc=r_for_eval / rc, theta0=theta0, ecc=ecc)
    theta = get_theta(theta0, orb_ang, orb_ang0)
    # get_dphi is always >= 0, so spin sets the direction of motion in phi
    phi = phi0 + spin * get_dphi(theta, theta0=theta0)

    theta = jnp.where(r_valid, theta, theta0 + to_float64(0.1))
    phi = jnp.where(r_valid, phi, phi0)
    orb_ang = jnp.where(r_valid, orb_ang, orb_ang0)

    v_r, v_theta, v_phi = stream_line_vel(
        r_for_eval,
        theta,
        orb_ang,
        stream_state=stream_state,
        theta0=theta0,
        r_mask=None,
        spin=spin,
    )

    v_x = v_r * jnp.sin(theta) * jnp.cos(phi) + v_theta * jnp.cos(theta) * jnp.cos(phi) - v_phi * jnp.sin(phi)
    v_y = v_r * jnp.sin(theta) * jnp.sin(phi) + v_theta * jnp.cos(theta) * jnp.sin(phi) + v_phi * jnp.cos(phi)
    v_z = v_r * jnp.cos(theta) - v_theta * jnp.sin(theta)
    x = r_for_eval * jnp.sin(theta) * jnp.cos(phi)
    y = r_for_eval * jnp.sin(theta) * jnp.sin(phi)
    z = r_for_eval * jnp.cos(theta)

    rotated_x, rotated_y, rotated_z = rotate_xyz(x, y, z, rotation_matrix=rotation_matrix)
    rotated_v_x, rotated_v_y, rotated_v_z = rotate_xyz(v_x, v_y, v_z, rotation_matrix=rotation_matrix)

    return (rotated_x, rotated_y, rotated_z), (rotated_v_x, rotated_v_y, rotated_v_z), r_valid

@jax.jit
def evaluate_streamline_at_delta(delta, mass, r0, theta0, phi0, mu, v_r0, inc, pa, spin=1.0):
    '''Evaluate the Mendoza streamline in closed form at in-plane angle delta.

    delta is the angle travelled within the orbital plane since r0, i.e. delta = orb_ang - orb_ang0
    (varphi - varphi_0 in Mendoza+2009). delta = 0 is the start of the streamline at r0, and
    delta = pi/2 is where the streamline reaches the disk midplane (z = 0 before rotation).

    This is algebraically identical to evaluate_streamline_at_radius, but uses no arccos, tan or
    safe-epsilon clipping, so it is smooth in every parameter at every delta. See docs/integrated_matching.tex
    for the derivation. With s = sin(theta0):
        1 - e cos(orb_ang) = 1 - (1 - mu s^2) cos(delta) + nu s sin(delta)
        e sin(orb_ang)     = nu s cos(delta) + (1 - mu s^2) sin(delta)
        r = rc s^2 / (1 - e cos(orb_ang))
        position = r n,  velocity = (vk0 / s) (-e sin(orb_ang) n + (1 - e cos(orb_ang)) t)
    where n and t are the radial and tangential unit vectors in the orbital plane.

    spin = -1 (counter-rotating) mirrors the in-plane unit vectors in y before the rotation by phi0.
    Since reflect_y . Rz(-phi0) = Rz(phi0) . reflect_y, this is the mirror image of the spin = +1
    streamline started at -phi0, which is what phi = phi0 + spin*dphi and spin*v_phi give in
    evaluate_streamline_at_radius and xyz_stream.

    :return: (x, y, z) in au, (v_x, v_y, v_z) in km/s, both rotated onto the sky, and r in au
    '''
    delta = jnp.asarray(delta, dtype=FLOAT_DTYPE)
    spin = jnp.asarray(spin, dtype=FLOAT_DTYPE)
    stream_state = build_stream_quantities(mass=mass, r0=r0, theta0=theta0, mu=mu, v_r0=v_r0)
    rc = stream_state.rc
    nu = stream_state.nu
    vk0 = stream_state.vk0

    sin_theta0 = jnp.sin(theta0)
    cos_theta0 = jnp.cos(theta0)
    cos_delta = jnp.cos(delta)
    sin_delta = jnp.sin(delta)

    # e cos(orb_ang0) and e sin(orb_ang0), as in build_stream_quantities
    ecc_cos_orb_ang0 = 1.0 - stream_state.mu * jnp.power(sin_theta0, 2)
    ecc_sin_orb_ang0 = nu * sin_theta0
    # angle-addition expansions of 1 - e cos(orb_ang) and e sin(orb_ang)
    one_minus_ecos = 1.0 - ecc_cos_orb_ang0 * cos_delta + ecc_sin_orb_ang0 * sin_delta
    ecc_sin = ecc_sin_orb_ang0 * cos_delta + ecc_cos_orb_ang0 * sin_delta

    r = rc * jnp.power(sin_theta0, 2) / one_minus_ecos

    # radial (n) and tangential (t) unit vectors in the orbital plane, before rotating by phi0.
    # The orbital plane contains the start point (theta0, phi0) and the initial azimuthal direction.
    # spin mirrors them in y, so the streamline moves in the -phi direction for spin = -1
    n_x = sin_theta0 * cos_delta
    n_y = spin * sin_delta
    n_z = cos_theta0 * cos_delta
    t_x = -sin_theta0 * sin_delta
    t_y = spin * cos_delta
    t_z = -cos_theta0 * sin_delta

    # rotate about the z-axis by phi0
    cos_phi0 = jnp.cos(phi0)
    sin_phi0 = jnp.sin(phi0)
    n_x, n_y = cos_phi0 * n_x - sin_phi0 * n_y, sin_phi0 * n_x + cos_phi0 * n_y
    t_x, t_y = cos_phi0 * t_x - sin_phi0 * t_y, sin_phi0 * t_x + cos_phi0 * t_y

    x = r * n_x
    y = r * n_y
    z = r * n_z

    v_n = -vk0 * ecc_sin / sin_theta0  # radial velocity, v_r
    v_t = vk0 * one_minus_ecos / sin_theta0  # tangential speed, h / r
    v_x = v_n * n_x + v_t * t_x
    v_y = v_n * n_y + v_t * t_y
    v_z = v_n * n_z + v_t * t_z

    rotation_matrix = build_rotation_matrix(inc, pa)
    rotated_x, rotated_y, rotated_z = rotate_xyz(x, y, z, rotation_matrix=rotation_matrix)
    rotated_v_x, rotated_v_y, rotated_v_z = rotate_xyz(v_x, v_y, v_z, rotation_matrix=rotation_matrix)

    return (rotated_x, rotated_y, rotated_z), (rotated_v_x, rotated_v_y, rotated_v_z), r


def mu_from_model_params(model_params):
    '''Get mu = rc/r0 from whichever of 'mu', 'rc' or 'omega' is in model_params'''
    if 'mu' in model_params:
        return model_params['mu']
    if 'rc' in model_params:
        return model_params['rc'] / model_params['r0']
    if 'omega' in model_params:
        return mu_from_omega(omega=model_params['omega'], mass=model_params['mass'], r0=model_params['r0'])
    raise ValueError("model_params must contain either 'rc', 'omega', or 'mu'")


@jax.jit
def forward_model_at_delta(delta, model_params, distance_pc):
    '''Evaluate the closed-form streamline at in-plane angles delta (see evaluate_streamline_at_delta),
    returning RA offset (arcsec), Dec offset (arcsec), velocity (km/s) and radius (au), in the
    same conventions as forward_model_at_radius().'''
    distance_pc = jnp.asarray(distance_pc, dtype=FLOAT_DTYPE)
    (x, y, z), (vx, vy, vz), r = evaluate_streamline_at_delta(
        delta=delta,
        mass=model_params['mass'],
        r0=model_params['r0'],
        theta0=model_params['theta0'],
        phi0=model_params['phi0'],
        mu=mu_from_model_params(model_params),
        v_r0=model_params['v_r0'],
        inc=model_params['inc'],
        pa=model_params['pa'],
        spin=model_params.get('spin', 1.0),
    )
    ra_model = -x / distance_pc
    dec_model = z / distance_pc
    v_model = vy + model_params['v_lsr']
    return ra_model, dec_model, v_model, r


@jax.jit
def forward_model_at_radius(r, model_params, distance_pc):
    '''Evaluate the analytic streamline model at arbitrary radii, returning sky-plane
    offsets and velocity in the same conventions as forward_model().'''
    distance_pc = jnp.asarray(distance_pc, dtype=FLOAT_DTYPE)

    if 'mu' in model_params:
        mu = model_params['mu']
    elif 'rc' in model_params:
        mu = model_params['rc'] / model_params['r0']
    elif 'omega' in model_params:
        mu = mu_from_omega(omega=model_params['omega'], mass=model_params['mass'], r0=model_params['r0'])
    else:
        raise ValueError("model_params must contain either 'rc', 'omega', or 'mu'")

    model_params = dict(model_params)
    model_params['mu'] = mu
    (x, y, z), (vx, vy, vz), valid_mask = evaluate_streamline_at_radius(
        r=r,
        mass=model_params['mass'],
        r0=model_params['r0'],
        theta0=model_params['theta0'],
        phi0=model_params['phi0'],
        mu=model_params['mu'],
        v_r0=model_params['v_r0'],
        inc=model_params['inc'],
        pa=model_params['pa'],
        spin=model_params.get('spin', 1.0),
    )

    ra_model = -x / distance_pc
    dec_model = z / distance_pc
    v_model = vy + model_params['v_lsr']
    return jnp.where(valid_mask, ra_model, 0.0), jnp.where(valid_mask, dec_model, 0.0), jnp.where(valid_mask, v_model, 0.0)

def xyz_stream(mass=0.5, r0=1e4, theta0=jnp.radians(30),
               phi0=jnp.radians(15), mu=0.1, v_r0=0,
               inc=0, pa=0, rmin=None, deltar=1, npoints=1e6, spin=1.0):
    '''
    it gets xyz coordinates and velocities for a stream line.
    They are also rotated in PA and inclination along the line of sight.
    This is a wrapper around stream_line() and rotate_xyz()

    Spherical into cartesian transformation is done for position and velocity
    using:
    https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates

    :param mass: Central mass (unitless, Msun)
    :param r0: Initial radius of streamline (unitless, au)
    :param theta0: Initial polar angle of streamline (unitless, radians)
    :param phi0: Initial azimuthal angle of streamline (unitless, radians)
    :param mu: dimensionless, rc/r0 in (0, 1)
    :param v_r0: Initial radial velocity of the streamline, (km/s). v_r0 > 0 is infall at r0,
        v_r0 = 0 means r0 is the apocentre, v_r0 < 0 means gas is still moving outwards at r0
    :param inc: inclination with respect of line-of-sight, inc=0 is an edge-on-disk (unitless, radians)
    :param pa: Position angle of the rotation axis, measured due East from North. This is usually estimated from the outflow PA, or the disk PA-90deg., (unitless, radians)
    :param rmin: smallest radius for calculation, (unitless, au)
    :param deltar: spacing between two consecutive radii in the sampling of the streamer, in (unitless, au)
    :param npoints: number of points to sample along the streamer
        This is just so that arrays are fixed length for jax/jit compatibility,
        but the actual number of valid points is determined by r0, rmin, rc, deltar,
        so some of the returned points may be NaN if npoints is larger than the number of valid points
    :param spin: sense of rotation about the axis set by (inc, pa): +1 is co-rotating (Mendoza convention,
        motion in the positive phi direction), -1 is counter-rotating. Must be exactly +1 or -1
    :return: x, y, z in (au), v_x, v_y, v_z in (km/s)
    '''

    mass = jnp.asarray(mass, dtype=FLOAT_DTYPE)
    r0 = jnp.asarray(r0, dtype=FLOAT_DTYPE)
    theta0 = jnp.asarray(theta0, dtype=FLOAT_DTYPE)
    phi0 = jnp.asarray(phi0, dtype=FLOAT_DTYPE)
    mu = jnp.asarray(mu, dtype=FLOAT_DTYPE)
    v_r0 = jnp.asarray(v_r0, dtype=FLOAT_DTYPE)
    inc = jnp.asarray(inc, dtype=FLOAT_DTYPE)
    pa = jnp.asarray(pa, dtype=FLOAT_DTYPE)
    deltar = jnp.asarray(deltar, dtype=FLOAT_DTYPE)
    spin = jnp.asarray(spin, dtype=FLOAT_DTYPE)
    stream_state = build_stream_quantities(mass=mass, r0=r0, theta0=theta0, mu=mu, v_r0=v_r0)
    rc = stream_state.rc
    mu = stream_state.mu
    ecc = stream_state.ecc

    rotation_matrix = build_rotation_matrix(inc, pa)

    check_rc_r0(rc, r0)
    check_spin(spin)

    # find the smallest radius for calculation
    # this is the maximum between rmin and 0.5*rc
    r_low = jnp.maximum(rmin, rc*0.5)

    # r is values internal to the initial radius r0 for computation
    # r_mask is used to mask out points that are outside the valid range, but we still need to compute them for jax/jit compatibility
    r = (r0 - deltar) - jnp.arange(npoints-1, dtype=FLOAT_DTYPE) * deltar
    check_r_array(r, r_low)
    r_mask = r > r_low
    
    # calculate positions and velocities inside r0
    # the valid_mask will later be used to mask out invalid points. currently these values are zero
    orb_ang, theta, phi, valid_mask = stream_line(r, r_mask, stream_state=stream_state, theta0=theta0, phi0=phi0, spin=spin)
    v_r, v_theta, v_phi = stream_line_vel(r, theta, orb_ang, stream_state=stream_state, theta0=theta0, r_mask=r_mask, spin=spin)
    # prepend initial positions and velocities at r0
    valid_mask_full = jnp.concatenate((jnp.asarray([True], dtype=bool), valid_mask))
    r_full = jnp.concatenate((jnp.asarray([r0], dtype=FLOAT_DTYPE), r))
    theta_full = jnp.concatenate((jnp.asarray([theta0], dtype=FLOAT_DTYPE), theta))
    phi_full = jnp.concatenate((jnp.asarray([phi0], dtype=FLOAT_DTYPE), phi))
    orb_ang0 = stream_state.orb_ang0
    orb_ang_full = jnp.concatenate((jnp.asarray([orb_ang0], dtype=FLOAT_DTYPE), orb_ang))
    v_r0_consistent, v_theta0_consistent, v_phi0_consistent = stream_line_vel(
        r0, theta0, orb_ang0, stream_state=stream_state, theta0=theta0, r_mask=None, spin=spin
    )
    v_r_full = jnp.concatenate((jnp.asarray([v_r0_consistent], dtype=FLOAT_DTYPE), v_r))
    v_theta_full = jnp.concatenate((jnp.asarray([v_theta0_consistent], dtype=FLOAT_DTYPE), v_theta))
    v_phi_full = jnp.concatenate((jnp.asarray([v_phi0_consistent], dtype=FLOAT_DTYPE), v_phi))

    # convert from spherical into cartesian coordinates
    v_x = v_r_full * jnp.sin(theta_full) * jnp.cos(phi_full) \
          + v_theta_full * jnp.cos(theta_full) * jnp.cos(phi_full) \
          - v_phi_full * jnp.sin(phi_full)
    v_y = v_r_full * jnp.sin(theta_full) * jnp.sin(phi_full) \
          + v_theta_full * jnp.cos(theta_full) * jnp.sin(phi_full) \
          + v_phi_full * jnp.cos(phi_full)
    v_z = v_r_full * jnp.cos(theta_full) \
          - v_theta_full * jnp.sin(theta_full)
    x = r_full * jnp.sin(theta_full) * jnp.cos(phi_full)
    y = r_full * jnp.sin(theta_full) * jnp.sin(phi_full)
    z = r_full * jnp.cos(theta_full)
    rotated_x, rotated_y, rotated_z = rotate_xyz(x, y, z, rotation_matrix=rotation_matrix)
    rotated_v_x, rotated_v_y, rotated_v_z = rotate_xyz(v_x, v_y, v_z, rotation_matrix=rotation_matrix)
    # get mask from smallest radius for calculation
    gd_rlow = (r_full > r_low)
    gd_rlow = jnp.logical_or(gd_rlow, valid_mask_full)
    gd_rlow = gd_rlow.astype(x.dtype)
    # apply mask to set invalid points to zero
    rotated_x = jnp.where(gd_rlow, rotated_x, 0.0)
    rotated_y = jnp.where(gd_rlow, rotated_y, 0.0)
    rotated_z = jnp.where(gd_rlow, rotated_z, 0.0)
    rotated_v_x = jnp.where(gd_rlow, rotated_v_x, 0.0)
    rotated_v_y = jnp.where(gd_rlow, rotated_v_y, 0.0)
    rotated_v_z = jnp.where(gd_rlow, rotated_v_z, 0.0)
    return (rotated_x, rotated_y, rotated_z), \
           (rotated_v_x, rotated_v_y, rotated_v_z), \
           gd_rlow


checked_xyz_stream = jax.jit(checkify.checkify(xyz_stream), static_argnames=['npoints'])