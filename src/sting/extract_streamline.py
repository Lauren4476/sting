'''
This file contains functions to extract streamline emission from a data cube,
and extract a 1D streamline from that.
'''

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
from collections import namedtuple
import jax.numpy as jnp
import jax

BIG = 1e30

def to_float64(value):
    return jnp.asarray(value, dtype=jnp.float64)

PreparedData = namedtuple('PreparedData', [
    'ra_data', 'dec_data', 'v_data',
    'ra_sigma_safe', 'dec_sigma_safe', 'v_sigma_safe',
    'dmetric_data', 'data_finite_mask',
    'data_min', 'data_max',
    'r_proj_data', 'theta_proj_data',
])


#@jax.jit
def wrap_to_pi(angle):
    '''Wrap angles to [-pi, pi)'''
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

#@jax.jit
def circular_median(theta_vals, weights):
    '''Branch-cut-safe median angle. (unrwap, linear median, rewrap)
    theta values with weight = 0 are ignored in the median calculation'''
    weights = weights / (jnp.sum(weights) + 1e-12) # normalize weights to sum to 1, add small value to avoid division by zero
    theta_anchor = jnp.arctan2(
        jnp.sum(weights * jnp.sin(theta_vals)),
        jnp.sum(weights * jnp.cos(theta_vals))
    )
    theta_delta = wrap_to_pi(theta_vals - theta_anchor)
    theta_unwrapped = theta_anchor + theta_delta
    sort_idx = jnp.argsort(theta_unwrapped)
    sorted_vals = theta_unwrapped[sort_idx]
    sorted_weights = weights[sort_idx]
    cumulative_weights = jnp.cumsum(sorted_weights)
    cutoff = 0.5 * jnp.sum(sorted_weights)
    median_idx = jnp.argmax(cumulative_weights >= cutoff)
    theta_ref = sorted_vals[median_idx]
    return wrap_to_pi(theta_ref)


#@jax.jit
def wrap_to_pi_numpy(angle):
    '''Wrap angles to [-pi, pi)'''
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def extract_streamer_subcube(cube, vmin=None, vmax=None, xmin=None, xmax=None, ymin=None, ymax=None, rms_thresh=None):
    """Extract a subcube containing the streamer emission, by applying velocity and spatial limits, and masking out low SNR emission."""
    streamer_cube = cube
    if (vmin is not None) and (vmax is not None):
        streamer_cube = streamer_cube.spectral_slab(vmin, vmax)
    if (xmin is not None) and (xmax is not None) and (ymin is not None) and (ymax is not None):
        celestial_wcs = streamer_cube.wcs.celestial
        ny, nx = streamer_cube.shape[1], streamer_cube.shape[2]
        # reference sky coord corresponding to the reference pixel in the WCS
        ref_ra, ref_dec = celestial_wcs.wcs.crval
        ref_coord = SkyCoord(ref_ra*u.deg, ref_dec*u.deg, frame=FK5)
        # convert the limits from offsets to sky coords
        corner1 = SkyCoord(ref_coord.ra + xmin, ref_coord.dec + ymin, frame=FK5) #'bottom left'
        corner2 = SkyCoord(ref_coord.ra + xmax, ref_coord.dec + ymax, frame=FK5) #'top right'
        x1, y1 = celestial_wcs.world_to_pixel(corner1)
        x2, y2 = celestial_wcs.world_to_pixel(corner2)
        xmin_pix = max(0, int(np.floor(min(x1, x2))))
        xmax_pix = min(nx, int(np.ceil(max(x1, x2))))
        ymin_pix = max(0, int(np.floor(min(y1, y2))))
        ymax_pix = min(ny, int(np.ceil(max(y1, y2))))
        streamer_cube = streamer_cube[:, ymin_pix:ymax_pix, xmin_pix:xmax_pix]
    if rms_thresh is not None:
        rms_estimate = streamer_cube.mad_std()
        streamer_cube = streamer_cube.with_mask(streamer_cube > rms_thresh*rms_estimate) 

    return streamer_cube

def reduce_to_1D(streamer_cube, yso_centre, n_elements=10):
    '''
    Reduce a cube of emission to a 1D 'streamline' by weighted means

    Parameters
    ----------
    streamer_cube : SpectralCube object, should contain only streamer emission
    yso_centre : SkyCoord, the coordinates of the star, used to compute RA and Dec offsets in arcsec
    n_elements : int, number of elements to reduce the cube to

    Returns
    -------
    pc_means : array of shape (3, n_elements), the weighted mean coordinates of each bin
    index 0 = RA offsets (arcsec)
    index 1 = Dec offsets (arcsec)
    index 2 = velocity (km/s)
    '''
    print('Starting reduction')
    nz, ny, nx = streamer_cube.shape
    yso_centre_icrs = yso_centre.icrs

    # create RA and Dec offset arrays in arcsec relative to the yso centre
    y_indices, x_indices = np.mgrid[0:ny, 0:nx]
    world_coords = streamer_cube.wcs.celestial.pixel_to_world_values(x_indices.ravel(), y_indices.ravel())
    ra_unit = u.Unit(streamer_cube.header.get('CUNIT1', streamer_cube.wcs.celestial.world_axis_units[0]))
    dec_unit = u.Unit(streamer_cube.header.get('CUNIT2', streamer_cube.wcs.celestial.world_axis_units[1]))
    world_sky = SkyCoord(
        ra=world_coords[0] * ra_unit,
        dec=world_coords[1] * dec_unit,
        frame='icrs'
    )
    dra, ddec = yso_centre_icrs.spherical_offsets_to(world_sky)
    ra_coords = dra.to(u.arcsec).value.reshape(ny, nx)
    dec_coords = ddec.to(u.arcsec).value.reshape(ny, nx)
    # create velocity array relative to the reference channel, then express it in km/s
    spectral_unit = u.Unit(streamer_cube.header.get('CUNIT3', streamer_cube.spectral_axis.unit))
    spectral_axis = streamer_cube.spectral_axis.to(spectral_unit)
    v_coords = spectral_axis.to(u.km / u.s).value
    print('Created coordinate arrays')

    # get data and mask
    pcloud = np.array(streamer_cube)
    rms_mask = ~np.isnan(pcloud)
    flux = pcloud[rms_mask]
    # get indices of valid points in pc
    pc_indices = np.indices(pcloud.shape)
    pc_z = pc_indices[0][rms_mask]
    pc_y = pc_indices[1][rms_mask]
    pc_x = pc_indices[2][rms_mask]
    print('Got point cloud with', len(flux), 'points')

    # extract coordinates of valid points using the arrays above
    pc_ra = ra_coords[pc_y, pc_x]
    pc_dec = dec_coords[pc_y, pc_x]
    pc_v = v_coords[pc_z]
    pc_coords = np.array([pc_ra, pc_dec, pc_v]) # shape (3, n_points)   

    # compute partitions for binning the point cloud
    distance_metric, _ = get_distance_metric(pc_coords[0], pc_coords[1], n_elements=n_elements)
    b_per = np.linspace(0, 100, n_elements+1) # percentiles to bin the pc into
    partitions = np.array([np.percentile(distance_metric, per) for per in b_per])
    print("Partition boundaries for projected distance metric:", np.round(partitions, 3))

    # flux-weighted means and stds in each bin
    pc_means = np.zeros((3, n_elements))
    pc_stds = np.zeros((3, n_elements))
    for i in range(n_elements):
        distance_indices = (distance_metric > partitions[i]) & (distance_metric <= partitions[i+1])
        pc_means[:, i] = np.average(pc_coords.T[distance_indices],
                                 axis=0,
                                 weights=flux[distance_indices])
        pc_stds[:, i] = np.sqrt(np.average((pc_coords.T[distance_indices] - pc_means[:, i])**2,
                                         axis=0,
                                         weights=flux[distance_indices]))
        
    # flip arrays so that they go from large to small distance (towards star)
    pc_means = pc_means[:, ::-1]
    pc_stds = pc_stds[:, ::-1]
    
    return pc_coords, pc_means, pc_stds


#@jax.jit
def safe_percentile(values, percentile):
    """
    jax and jit-safe percentile ignoring invalid values, which does not change array shape
    """
    mask = jnp.isfinite(values)
    
    # Sort valid values to the front by pushing invalid ones to BIG
    cleaned = jnp.where(mask, values, to_float64(BIG))
    sorted_vals = jnp.sort(cleaned)  # valid values are at the front

    # make a new mask which is all the not BIG values 
    percentile_mask = sorted_vals < to_float64(BIG)
    n_valid = jnp.sum(percentile_mask)
    total_n = values.size
    # Compute the index into only the valid portion
    idx = jnp.clip(
        jnp.floor(percentile / 100.0 * n_valid).astype(jnp.int32),
        0,
        jnp.maximum(n_valid - 1, 0)
    )
    return sorted_vals[idx]

        
#@jax.jit
def get_distance_metric(ra_coords, dec_coords, n_elements=10):
    '''
    Compute radial + angular distance metric for point cloud binning
    Uses a circular angular deviation to avoid branch-cut artifacts.
    '''
    pc_r, pc_theta = cartesian_to_polar(ra_coords, dec_coords)
    pc_r = jnp.where(jnp.abs(pc_r) < 1e-12, to_float64(BIG), pc_r)

    finite_mask = jnp.isfinite(pc_r) & jnp.isfinite(pc_theta)

    # deal with if there are no valid points
    def empty_case(_):

        distance_metric = jnp.full_like(pc_r, to_float64(BIG))
        trace = {
            "n_points":            jnp.array(pc_r.size,   dtype=jnp.int32),
            "n_finite_points":     jnp.array(0,            dtype=jnp.int32),
            "n_reference_points":  jnp.array(0,            dtype=jnp.int32),
            "r_percentile_thresh": to_float64(0.0),
            "r_thresh":            to_float64(BIG),
            "theta_ref":           to_float64(0.0),
            "theta_weight":        to_float64(1.0),
            "close_point_count":   jnp.array(0,            dtype=jnp.int32),
        }
        return distance_metric, trace

    def notempty_case(_):

        theta_weight = 1.0 # maybe make this a tunable parameter
        finite_count = jnp.sum(finite_mask)

        percentile = 100.0 / n_elements
        r_thresh = safe_percentile(pc_r, percentile)
        # close_mask gives 0s if point is not finite or outside the threshold, and 
        # 1s if point is finite and within the threshold
        small_enough_r = pc_r <= r_thresh
        close_mask = (finite_mask & (pc_r <= r_thresh)).astype(jnp.float64)
        # circular median can not have nans passed in,
        # so change nans to 0 (this is fine because they already have weight=0 in the median calculation)
        pc_theta_no_nan = jnp.where(finite_mask, pc_theta, 0.0)
        theta_ref = circular_median(pc_theta_no_nan, weights=close_mask)

        # cyclic angular deviation
        theta_dev = jnp.pi - jnp.abs(
            jnp.pi - jnp.abs(wrap_to_pi(pc_theta - theta_ref))
        )

        distance_metric = pc_r * jnp.sqrt(1.0 + (theta_weight * theta_dev) ** 2)
        distance_metric = jnp.where(finite_mask, distance_metric, to_float64(BIG))
        trace = {
            "n_points":            jnp.array(pc_r.size,       dtype=jnp.int32),
            "n_finite_points":     jnp.array(finite_count,    dtype=jnp.int32),
            "n_reference_points":  jnp.array(jnp.sum(small_enough_r), dtype=jnp.int32),
            "r_percentile_thresh": to_float64(percentile),
            "r_thresh":            r_thresh,
            "theta_ref":           theta_ref,
            "theta_weight":        theta_weight,
            "close_point_count":   jnp.array(close_mask.size, dtype=jnp.int32),
        }

        return distance_metric, trace
    
    return jax.lax.cond(finite_mask.any(), notempty_case, empty_case, operand=None)

#@jax.jit
def cartesian_to_polar(x, y):
    '''
    Convert cartesian coordinates (x,y) to polar coordinates
    e.g. inputs could be RA and Dec offsets
    Note theta is returned in radians
    '''
    r = jnp.sqrt(x**2 + y**2 + to_float64(1e-60)) # add small value for gradient stability
    theta = jnp.arctan2(y, x) # angle wrt x-axis, in radians

    return (r, theta)


def get_metric_partitions(pc_coords, n_elements):
    '''
    Compute percentile partitions for the streamline distance metric

    Parameters
    ----------
    pc_coords : array
        Point cloud coordinates. Index 0 = RA, Index 1 = Dec, Index 2 = velocity
    n_elements : int
        Number of partitions required

    Returns
    -------
    partitions : ndarray
        Percentile boundaries of the distance metric
    '''
    if n_elements < 1:
        raise ValueError('n_elements must be >= 1')

    ra_coords = pc_coords[0]
    dec_coords = pc_coords[1]
    distance_metric, _ = get_distance_metric(ra_coords, dec_coords, n_elements=n_elements)
    distance_metric = np.asarray(distance_metric)
    finite_mask = np.isfinite(distance_metric)
    finite_metric = distance_metric[finite_mask]

    if finite_metric.size == 0:
        return np.full(n_elements + 1, np.nan, dtype=np.float64)

    b_per = np.linspace(0.0, 100.0, n_elements + 1)
    return np.asarray([np.percentile(finite_metric, per) for per in b_per], dtype=np.float64)


def get_metric_reference_trace(pc_coords, n_elements=10):
    '''get the metric reference angle and weight used for boundary sampling'''
    ra_coords = pc_coords[0]
    dec_coords = pc_coords[1]
    _, trace = get_distance_metric(ra_coords, dec_coords, n_elements=n_elements)
    theta_ref = float(trace.get('theta_ref', 0.0))
    theta_weight = float(trace.get('theta_weight', 1.0))
    return theta_ref, theta_weight


def sample_metric_boundary(partition_radius, theta_ref, theta_weight=1.0, n_samples=720):
    '''create a constant-metric boundary as a closed RA/Dec curve (for plotting)'''
    if n_samples < 4:
        raise ValueError('n_samples must be >= 4')

    theta = jnp.linspace(-jnp.pi, jnp.pi, n_samples, endpoint=False)
    theta_dev = jnp.pi - jnp.abs(jnp.pi - jnp.abs(wrap_to_pi(theta - theta_ref)))
    radius = partition_radius / jnp.sqrt(1.0 + (theta_weight * theta_dev) ** 2)
    ra = radius * jnp.cos(theta)
    dec = radius * jnp.sin(theta)
    return ra, dec


def sample_metric_boundaries(pc_coords, partitions, n_samples=720):
    '''create all metric boundary curves for a point cloud and partition set'''
    theta_ref, theta_weight = get_metric_reference_trace(pc_coords, n_elements=len(partitions)-1)
    curves = [
        sample_metric_boundary(partition_radius, theta_ref, theta_weight=theta_weight, n_samples=n_samples)
        for partition_radius in np.asarray(partitions)
    ]
    trace = {
        'theta_ref': theta_ref,
        'theta_weight': theta_weight,
    }
    return curves, trace


def plot_metric_boundaries(ax, pc_coords, curves, color='lightgrey', linewidth=1, alpha=0.5, n_samples=720, zorder=1):
    '''plot metric boundary curves on a RA/Dec axis'''
    for ra, dec in curves:
        ax.plot(ra, dec, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    return curves


def prepare_data(data, uncertainties, n_elements):
    '''
    Precompute all the constant data-only quantities used by the gradient descent,
    to speed up later iterations

    Parameterss
    ----------
    data : tuple of arrays (ra_data, dec_data, v_data)
        Observed RA offset (arcsec), Dec offset (arcsec), velocity (km/s)
    uncertainties : tuple of arrays (ra_sigma, dec_sigma, v_sigma)
        Uncertainties on the data
    n_elements : int
        Number of elements to reduce the cube to, used for computing the distance metric and its partitions

    Returns
    -------
    PreparedData
        Container containing the precomputed quantities
    '''
    ra_data = jnp.asarray(data[0], dtype=jnp.float64)
    dec_data = jnp.asarray(data[1], dtype=jnp.float64)
    v_data = jnp.asarray(data[2], dtype=jnp.float64)

    ra_sigma = jnp.asarray(uncertainties[0], dtype=jnp.float64)
    dec_sigma = jnp.asarray(uncertainties[1], dtype=jnp.float64)
    v_sigma = jnp.asarray(uncertainties[2], dtype=jnp.float64)

    eps = jnp.asarray(1e-8, dtype=jnp.float64)
    ra_sigma_safe = jnp.maximum(ra_sigma, eps)
    dec_sigma_safe = jnp.maximum(dec_sigma, eps)
    v_sigma_safe = jnp.maximum(v_sigma, eps)

    dmetric_data, _ = get_distance_metric(ra_data, dec_data, n_elements=n_elements)
    data_finite_mask = jnp.isfinite(ra_data) & jnp.isfinite(dec_data) & jnp.isfinite(dmetric_data)

    data_metric_for_min = jnp.where(data_finite_mask, dmetric_data, jnp.inf)
    data_metric_for_max = jnp.where(data_finite_mask, dmetric_data, -jnp.inf)
    data_min = jnp.min(data_metric_for_min)
    data_max = jnp.max(data_metric_for_max)

    r_proj_data, theta_proj_data = cartesian_to_polar(ra_data, dec_data)

    return PreparedData(
        ra_data=ra_data,
        dec_data=dec_data,
        v_data=v_data,
        ra_sigma_safe=ra_sigma_safe,
        dec_sigma_safe=dec_sigma_safe,
        v_sigma_safe=v_sigma_safe,
        dmetric_data=dmetric_data,
        data_finite_mask=data_finite_mask,
        data_min=data_min,
        data_max=data_max,
        r_proj_data=r_proj_data,
        theta_proj_data=theta_proj_data,
    )


