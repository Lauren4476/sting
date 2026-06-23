'''
This file contains the loss function and optimisation routines for streamfit.

The optimisation uses adam (adaptive moment estimation) optimiser to fit
streamline model parameters to observed data by minimizing chi-squared loss.

Last updated: 19-06-2026
'''

import os
from collections import namedtuple

import jax.numpy as jnp
from jax import value_and_grad, lax
import jax
from jax.experimental import checkify
from jax.random import key
import optax
from . import stream_lines_grad
from . import extract_streamline
import csv
import astropy.units as u
import math
import traceback
jax.config.update("jax_enable_x64", True)

# settings and constants
VR0_MIN = 1e-6
BIG = 1e30
BIG_NEG = -1e30

LOSS_METHOD_CHOICES = [0, 1]

LOSS_METHOD_COMPONENT_KEYS = {
    0: ('chi2_ra', 'chi2_dec', 'chi2_v'), #radecvel
    1: ('chi2_r', 'chi2_theta', 'chi2_v'), #rthetavel
}

TRACE_COMMON_FIELDNAMES = [
    'epoch',
    'loss',
    'chi2_total',
    'grad_norm',
    'model_points_total',
    'model_nan_count',
    'model_valid_points',
    'model_metric_span',
    'model_inner_count',
    'data_inner_count',
    'data_points_total',
    'data_valid_points',
    'data_retained_count',
    'model_retained_count',
    'overlap_metric_min',
    'overlap_metric_max',
]

CANONICAL_UNITS = {
    "r0": u.au,
    "theta0": u.rad,
    "phi0": u.rad,
    "inc": u.rad,
    "pa": u.rad,
    "v_r0": u.km / u.s,
    "mass": u.Msun,
    "rmin": u.au,
    "deltar": u.au,
    "v_lsr": u.km / u.s,
    "rc": u.au,
    "omega": 1 / u.s,
    # mu = rc/r0 is dimensionless, so no units
}

ANGLE_KEYS = {'theta0', 'phi0', 'inc', 'pa'}
DISPLAY_UNITS = {
        'r0':      'au',
        'v_r0':    'km/s',
        'mass':    'M_sun',
        'rmin':    'au',
        'deltar':  'au',
        'v_lsr':   'km/s',
        'rc':      'au',
        'omega':   '1/s',
        # mu is dimensionless
    }

STREAMLINE_MODEL_PARAM_KEYS = (
    'r0',
    'theta0',
    'phi0',
    'rc',
    'omega',
    'mu',
    'v_r0',
    'mass',
    'inc',
    'pa',
    'rmin',
    'deltar',
    'v_lsr',
)

def convert_and_strip_bound_units(bounds):
    """
    Convert bounds that are astropy quantitiesinto canonical units,
    then strip the units.

    If input is already plain numeric, assume it's already in canonical units.

    Required as JAX optimiser works with unitless arrays.
    """

    output = {}

    for key, val in bounds.items():
        if isinstance(val, u.Quantity):
            if key not in CANONICAL_UNITS:
                raise ValueError(f"The parameter {key} doesn't have defined canonical units...")
            bounds = val.to(CANONICAL_UNITS[key])
            output[key] = (
                float(bounds[0].value),
                float(bounds[1].value),
            )
        else:
            # already unitless
            output[key] = tuple(float(v) for v in val)
    return output

def check_loss_method(loss_method):
    """Check that the selected loss method is valid and return it"""
    if loss_method not in LOSS_METHOD_CHOICES:
        raise ValueError(
            f"Unknown loss_method '{loss_method}'. "
            f"Choose from: 0: radecvel 1: rthetavel"
        )
    return loss_method


def trace_fieldnames_for_loss_method(loss_method):
    """Return the trace csv headers for the chosen loss method"""
    loss_method = check_loss_method(loss_method)
    return ['epoch', 'loss', *LOSS_METHOD_COMPONENT_KEYS[loss_method], *TRACE_COMMON_FIELDNAMES[2:]]


def is_numeric_value(value):
    """Return True for scalar/array-like numeric values"""
    try:
        arr = jnp.asarray(value)
    except Exception:
        return False
    if arr.dtype == jnp.bool_:
        return False
    return bool(jnp.issubdtype(arr.dtype, jnp.number))

@jax.jit
def to_float64(value):
    """Convert a numeric value or array-like input to float64"""
    return jnp.asarray(value, dtype=jnp.float64)

@jax.jit
def softplus(x):
    """Softplus, used for v_r0"""
    return jnp.logaddexp(x, 0.0)         

@jax.jit
def inv_softplus(y):
    """used for v_r0, stable for y>0"""
    y = to_float64(y)
    return y + jnp.log1p(-jnp.exp(-y))


def get_checkify_error_message(err):
    """Extract human-readable error message from a checkify.Error if possible,
    or None if it doesn't contain anything"""
    if hasattr(err, 'get'):
        return err.get()
    try:
        err.throw()
    except Exception as e:
        return str(e)
    return None

def make_data_tuple_float64(values):
    """Convert tuple/list of arrays to float64 arrays"""
    return tuple(to_float64(value) for value in values)


def clean_model_param_dict(params, dict_name):
    """Convert parameter dictionary to float64 and standardise it"""
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise TypeError(f"{dict_name} must be a dictionary, got {type(params).__name__}.")

    sanitized = {}

    for key, val in params.items():
        if val is None:
            sanitized[key] = None
            continue
        if isinstance(val, u.Quantity):
            if key not in CANONICAL_UNITS:
                raise ValueError(f"The parameter {key} doesn't have defined canonical units...")
            val = val.to(CANONICAL_UNITS[key]).value
        # if it's already a raw number, assume it's already correct
        sanitized[key] = jnp.asarray(val, dtype=jnp.float64)

    tiny = to_float64(1e-8)
    # Protect against exact polar-angle edge values which can cause
    # downstream numerical issues (theta=0 or theta=pi). 
    # If the uservsupplied exactly 0 or pi, 
    # nudge by a tiny amount into the open interval (0, pi).
    if 'theta0' in sanitized:
        try:
            theta_val = to_float64(sanitized['theta0'])
            if bool(jnp.all(jnp.isclose(theta_val, to_float64(0.0)))):
                sanitized['theta0'] = theta_val + tiny
            elif bool(jnp.all(jnp.isclose(theta_val, to_float64(jnp.pi)))):
                sanitized['theta0'] = theta_val - tiny
        except Exception:
            pass

    unknown = sorted(key for key in sanitized if key not in STREAMLINE_MODEL_PARAM_KEYS)
    if unknown:
        raise KeyError(
            f"Unknown parameter keys in {dict_name}: {unknown} "
            f"Supported keys are: {list(STREAMLINE_MODEL_PARAM_KEYS)}"
        )

    return sanitized


def check_param_types(opt_params, fixed_params):
    """Check that model parameters are of the correct type (numeric or None for rmin)"""
    for key, value in opt_params.items():
        if key == 'rmin' and value is None:
            raise ValueError("'rmin' cannot be None")
        if isinstance(value, bool) or not is_numeric_value(value):
            raise TypeError(
                f"Optimisable parameter '{key}' must be numeric, "
                f"got value of type {type(value).__name__}."
            )

    for key, value in fixed_params.items():
        if key == 'rmin' and value is None:
            continue
        if isinstance(value, bool) or not is_numeric_value(value):
            raise TypeError(
                f"Fixed parameter '{key}' must be numeric"
                " (or None only for 'rmin'), "
                f"got value of type {type(value).__name__}."
            )


def sanitize_param_partition(initial_opt_params, fixed_params, require_nonempty_opt=False):
    """Sanitize and validate opt/fixed parameter partition for streamline modeling.
    Note: exactly one of 'rc' or 'omega' must be present across initial_opt_params and fixed_params, to determine mu=rc/r0."""
    opt_params = clean_model_param_dict(initial_opt_params, 'initial_opt_params')
    fixed_params = clean_model_param_dict(fixed_params, 'fixed_params')

    overlap = sorted(set(opt_params) & set(fixed_params))
    if overlap:
        raise KeyError(
            f"Parameters cannot be present in both initial_opt_params and fixed_params! Overlap: {overlap}"
        )
    
    all_params = set(opt_params) | set(fixed_params)

    # check that exactly one of rc, omega, or mu (the rotation keys) is supplied
    rotation_keys_present = [ key for key in ('rc', 'omega', 'mu') if key in all_params]
    if len(rotation_keys_present) != 1:
        raise KeyError(
            f"Exactly one of 'rc', 'omega', or 'mu' must be provided. You have provided: {rotation_keys_present}"
        )
    
    # check all other required parameters are present (except mu, rc, omega which we already dealt with)
    already_dealt_with = {'rc', 'omega', 'mu'}
    missing = []
    for key in STREAMLINE_MODEL_PARAM_KEYS:
        if key not in all_params and key not in already_dealt_with:
            missing.append(key)
    if missing:
        raise KeyError(
            "Missing required streamline parameters across initial_opt_params and fixed_params: "
            f"{missing}."
        )

    if require_nonempty_opt and len(opt_params) == 0:
        raise ValueError(
            "initial_opt_params must contain at least one optimisable parameter. "
        )

    check_param_types(opt_params, fixed_params)

    return opt_params, fixed_params


def prepare_model_params(opt_params, fixed_params):
    """Construct merged model parameters and clean opt/fixed dictionaries"""
    opt_params, fixed_params = sanitize_param_partition(opt_params, fixed_params)
    model_params = fixed_params.copy()
    model_params.update(opt_params)
    return model_params, opt_params, fixed_params


def standardise_param_bounds(param_bounds):
    """Check/standardise parameter-bound keys"""
    if param_bounds is None:
        return None

    standardised = dict(param_bounds)

    unknown = sorted(key for key in standardised if key not in STREAMLINE_MODEL_PARAM_KEYS)
    if unknown:
        raise KeyError(
            f"Unknown params in param_bounds: {unknown}. "
            f"Supported params are: {list(STREAMLINE_MODEL_PARAM_KEYS)}"
        )

    return standardised


def build_normalisation_spec(opt_params, param_bounds):
    """Build shift and scale for normalisation ofoptimised parameters, from bounds.
    Doesn't include v_r0 since we use softplus transform instead for that."""
    if param_bounds is None:
        raise ValueError(
            "param_bounds is required because optimisation is performed in normalised space. "
            "Provide bounds for every parameter you want to optimise (except v_r0, which is handled separately)."
        )
 
    missing = []
    for key in opt_params:        
        if key == 'v_r0':
            continue
        if key not in param_bounds:
                missing.append(key)
    if missing:
        raise ValueError(
            "Missing bounds for optimised parameters: "
            f"{missing}. Please add (min, max) entries for all parameters you want to optimise."
        )

    normalisation_spec = {}
    for key, value in opt_params.items():
        if key == 'v_r0':
            if key in param_bounds:
                print("Notice: Ignoring user-supplied bounds for 'v_r0' since we use a softplus transform for this parameter instead of normalisation.")
            continue
        bounds = param_bounds[key]
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(
                f"Bounds for '{key}' must be a 2-element (min, max) tuple."
                f"Got: {bounds!r}"
            )

        lower_bound = to_float64(bounds[0])
        upper_bound = to_float64(bounds[1])
        if not bool(jnp.isfinite(lower_bound)) or not bool(jnp.isfinite(upper_bound)):
            raise ValueError(f"Bounds for '{key}' must be finite. Got ({lower_bound}, {upper_bound})")
        if not bool(upper_bound > lower_bound):
            raise ValueError(
                f"Bounds for '{key}' must satisfy min < max. Got ({float(lower_bound)}, {float(upper_bound)})"
            )

        value = to_float64(value)
        if not bool((value >= lower_bound) & (value <= upper_bound)):
            raise ValueError(
                f"Initial value for '{key}' ({float(value)}) is outside bounds"
                f"({float(lower_bound)}, {float(upper_bound)})."
            )

        scale = upper_bound - lower_bound
        normalisation_spec[key] = {
            'offset': lower_bound,
            'scale': scale,
        }

    return normalisation_spec


def normalise_opt_params(opt_params, normalisation_spec):
    """normalise optimised parameters to [0, 1], unless v_r0 which is transformed by softplus instead"""
    normalised = {}
    for key, value in opt_params.items():
        if key == 'v_r0':
            # save the value 'raw' such that v_r0 = softplus(raw)
            value = to_float64(value)
            if value < 0:
                raise ValueError(f"v_r0 must be non-negative, got {float(value)}")
            normalised[key] = inv_softplus(value)
            continue
        offset = normalisation_spec[key]['offset']
        scale = normalisation_spec[key]['scale']
        normalised[key] = (to_float64(value) - offset) / scale
    return normalised


def denormalise_opt_params(norm_opt_params, normalisation_spec):
    """Convert normalised optimised parameters back to physical/log parameter values"""
    denormalised = {}
    for key, value in norm_opt_params.items():
        if key == 'v_r0':
            denormalised[key] = softplus(to_float64(value))
            continue
        offset = normalisation_spec[key]['offset']
        scale = normalisation_spec[key]['scale']
        if key == 'phi0':
            # special handling for phi0 because circular
            denormalised[key] = jnp.mod(to_float64(value) * scale + offset, 2*jnp.pi)
        else:
            denormalised[key] = to_float64(value) * scale + offset
    return denormalised


def get_rotation_param_key(opt_params, fixed_params):
    """Return whichever of 'rc', 'omega', 'mu' is present in the parameters"""
    all_params = set(opt_params) | set(fixed_params)
    for key in ('rc', 'omega', 'mu'):
        if key in all_params:
            return key
    raise KeyError("None of the parameters 'rc', 'omega', or 'mu' are present in the parameters!")


def mu_from_rotation_param(rotation_key, value, mass, r0):
    """Convert whatever rotation parameter is preesnt into mu"""
    if rotation_key == 'mu':
        return value
    elif rotation_key == 'rc':
        return value / r0
    elif rotation_key == 'omega':
        return stream_lines_grad.mu_from_omega(omega=value, mass=mass, r0=r0)

def rotation_param_from_mu(rotation_key, mu, mass, r0):
    """Convert mu into the rotation parameter that is being used (the one that was input by the user)"""
    if rotation_key == 'mu':
        return mu
    elif rotation_key == 'rc':
        return mu * r0
    elif rotation_key == 'omega':
        return stream_lines_grad.omega_from_mu(mu=mu, mass=mass, r0=r0)

def with_mu_substituted(opt_params, fixed_params, param_bounds=None):
    """ Replace the user's input rotation parameter (either rc or omega) with mu, which is the parameter used internally for the physics calculations and optimisation,
    because it has obvious bounds (0,1) that will mean that optimisation won't explore regions where rc > r0.
    
    User should not have supplied bounds for 'rc' or 'omega', but if they did, prints a notice and ignores them.
    (Because the optimisation is performed in mu space where the bounds should be (0,1))"""
    rotation_key = get_rotation_param_key(opt_params, fixed_params)
    opt_params = dict(opt_params)
    fixed_params = dict(fixed_params)
    if param_bounds is not None:
        param_bounds = dict(param_bounds)
    else:
        param_bounds = {}

    all_params = {**fixed_params, **opt_params}
    mass = all_params['mass']
    r0 = all_params['r0']

    if rotation_key in opt_params:
        rotation_value = opt_params[rotation_key]
        mu_value = mu_from_rotation_param(rotation_key, rotation_value, mass, r0)
        del opt_params[rotation_key]
        opt_params['mu'] = mu_value
        # drop any rc/omega/mu bounds the user supplied, we use (0,1) bounds for mu
        if rotation_key in param_bounds:
            print(
                f"Notice: Ignoring user-supplied bounds for '{rotation_key}', since the optimisation is performed in 'mu' space. "
                f"Using (0, 1) bounds for 'mu' instead."
            )
            del param_bounds[rotation_key]
        param_bounds['mu'] = (0.0+1e-6, 1.0-1e-6) # tiny epsilon to avoid rc=r0 or rc=0
    else:
        # rotation parameter is in fixed params. so just rename it to 'mu' in fixed params for consistency
        rotation_value = fixed_params[rotation_key]
        mu_value = mu_from_rotation_param(rotation_key, rotation_value, mass, r0)
        del fixed_params[rotation_key]
        fixed_params['mu'] = mu_value
    
    return opt_params, fixed_params, param_bounds, rotation_key

def format_param(key, value):
    """
    Format parameter for display in output, with units. Notably:
    - converts angles (theta0, phi0, inc, pa) from radians to degrees
    """
    val = float(value)
    if key in ANGLE_KEYS:
        deg = math.degrees(val)
        return f"{deg:.6g} deg"
    if key == 'mu':
        return f"{val:.6g}"
    if key == 'omega':
        return f"{val:.6g} 1/s"
    unit = DISPLAY_UNITS.get(key, '')
    if unit:
        suffix = f" {unit}"
    else:
        suffix = ""
    return f"{val:.6g}{suffix}"

def add_rc_omega_to_log(row, opt_params, fixed_params, all_param_keys):
    """Add rc and omega to the log row for user-friendly output, converting from mu if necessary"""
    if 'mu' not in all_param_keys:
        return row
    mass_val = opt_params['mass'] if 'mass' in opt_params else fixed_params.get('mass', None)
    r0_val = opt_params['r0'] if 'r0' in opt_params else fixed_params.get('r0', None)
    mu_val = opt_params['mu'] if 'mu' in opt_params else fixed_params.get('mu', None)
    if mass_val is not None and r0_val is not None and mu_val is not None:
        rc_val = mu_val * r0_val
        omega_val = stream_lines_grad.omega_from_mu(mu=mu_val, mass=mass_val, r0=r0_val)
        row['rc'] = float(rc_val)
        row['omega'] = float(omega_val)
    return row


def build_trace_row(epoch, loss_value, loss_trace, grad_norm, loss_method):
    """Flatten trace dictionary into a csv row for output"""
    loss_method = check_loss_method(loss_method)
    chi2_components = loss_trace.get('chi2_components', {})
    matching = loss_trace.get('matching', {})
    model_metric_trace = matching.get('distance_metric_model', {})
    data_metric_trace = matching.get('distance_metric_data', {})

    row = {
        'epoch': epoch,
        'loss': loss_value,
    }
    for component_key in LOSS_METHOD_COMPONENT_KEYS[loss_method]:
        row[component_key] = chi2_components.get(component_key, float('nan'))

    row.update({
        'chi2_total': chi2_components.get('chi2_total', float('nan')),
        'grad_norm': grad_norm,
        'model_points_total': matching.get('model_points_total', 0),
        'model_nan_count': matching.get('model_nan_count', 0),
        'model_valid_points': matching.get('model_valid_points', 0),
        'model_metric_span': matching.get('model_metric_span', float('nan')),
        'model_inner_count': model_metric_trace.get('inner_count', 0),
        'data_inner_count': data_metric_trace.get('inner_count', 0),
        'data_points_total': matching.get('data_points_total', 0),
        'data_valid_points': matching.get('data_valid_points', 0),
        'data_retained_count': matching.get('data_retained_count', 0),
        'model_retained_count': matching.get('model_retained_count', 0),
        'overlap_metric_min': matching.get('overlap_metric_min', float('nan')),
        'overlap_metric_max': matching.get('overlap_metric_max', float('nan')),
    })

    return row

def trace_tree_to_python(value):
    """Go through the trace tree and convert JAX arrays to Python scalars where possible"""
    # go through containers, converting JAX arrays to Python scalars where possible, and leaving non-numeric values as-is
    if isinstance(value, dict):
        return {key: trace_tree_to_python(v) for key, v in value.items()}
    if isinstance(value, list):
        return [trace_tree_to_python(v) for v in value]
    if isinstance(value, tuple):
        return tuple(trace_tree_to_python(v) for v in value)
    # preserve Nones as-is
    if value is None:
        return None
    # convert jax arrays to python scalars where posible
    try:
        array_value = jnp.asarray(value)
    except Exception:
        return value
    # if it's a scalar array, convert to scalar
    if array_value.ndim == 0:
        return array_value.item()
    return value


@jax.jit
def gradient_l2_norm(grad_tree):
    """Compute L2 norm of gradients across all leaves in a pytree
    (a pytree is a nested structure of lists/dicts/tuples containing arrays, 
    used by jax for gradients)."""
    grad_leaves = jax.tree_util.tree_leaves(grad_tree)
    grad_sum_sq = jnp.asarray(0.0, dtype=jnp.float64)
    for grad_leaf in grad_leaves:
        grad_sum_sq = grad_sum_sq + jnp.sum(jnp.square(grad_leaf))
    return jnp.sqrt(grad_sum_sq)

@jax.jit(static_argnames=("npoints",))
def forward_model(model_params, distance_pc, npoints=10000):
    """
    Run the forward model using stream_lines_grad.checked_xyz_stream
    
    Parameters:
    -----------
    model_params: dict
        Dictionary of model parameters, including both optimised and fixed parameters
    distance_pc : float
        Distance to source in parsecs
    npoints : int
        Number of points to sample along the streamer
        This is just for jax/jit compatibility to have fixed-length arrays, but
        the actual number of valid points is determined by r0, rmin, rc, deltar,
        so some of the returned points may be NaN if npoints is larger than the number of valid points.
        
    Returns:
    --------
    tuple: (ra_offsets, dec_offsets, velocities)
        - RA offsets in arcsec (negative for standard convention)
        - Dec offsets in arcsec
        - Line-of-sight velocities in km/s, relative to v_lsr
    """


    distance_pc = to_float64(distance_pc)

    # Protect near-zero v_r0 from creating singularities in physics calculations
    # Allow negative v_r0, but replace exact-zero or tiny values with signed epsilon
    # v_r0_protected = model_params['v_r0']
    # threshold = to_float64(1e-6)
    # v_r0_protected = jnp.where(
    #     jnp.isclose(v_r0_protected, to_float64(0.0)),
    #     - jnp.sign(v_r0_protected) * threshold,
    #     v_r0_protected
    #     )

    # Run the forward model - returns positions in au, velocities in km/s
    # valid_mask is a boolean array marking which points are valid in the returned arrays, 
    # which can be used for masking in the loss function
    rmin = model_params['rmin']
    if rmin is None:
        rmin = to_float64(0.0)  # rc*0.5 will always dominate in jnp.maximum
    # derive mu from rc or omega (whicever is provided)
    if 'mu' in model_params:
        mu = model_params['mu']
    elif 'rc' in model_params:
        mu = model_params['rc'] / model_params['r0']
    elif 'omega' in model_params:
        mu = stream_lines_grad.mu_from_omega(omega=model_params['omega'], mass=model_params['mass'], r0=model_params['r0'])
    else:
        raise ValueError("model_params must contain either 'rc', 'omega', or 'mu'")
    model_params = dict(model_params)
    model_params['mu'] = mu

    err, ((x, y, z), (vx, vy, vz), valid_mask) = stream_lines_grad.checked_xyz_stream(
        mass=model_params['mass'],
        r0=model_params['r0'],
        theta0=model_params['theta0'],
        phi0=model_params['phi0'],
        mu=model_params['mu'],
        v_r0=model_params['v_r0'],
        inc=model_params['inc'],
        pa=model_params['pa'],
        rmin=rmin,
        deltar=model_params['deltar'],
        npoints=npoints
    )
    # err.throw()


    # Convert positions from au to arcsec offsets
    # x = RA offset (with negative for standard RA convention)
    # z = Dec offset
    # y = line-of-sight velocity
    ra_model = -x / distance_pc  # arcsec
    dec_model = z / distance_pc  # arcsec
    # make velocity absolute by adding back v_lsr
    v_model = vy + model_params['v_lsr']  # km/s 
    # make sure model points outside valid_mask are set to 0
    ra_model = jnp.where(valid_mask, ra_model, 0.0)
    dec_model = jnp.where(valid_mask, dec_model, 0.0)
    v_model = jnp.where(valid_mask, v_model, 0.0)


    return ra_model, dec_model, v_model, valid_mask, err


@jax.jit
def distance_metric_overlap(dmetric_model, model_finite_mask, dmetric_data, data_finite_mask):
    """Compute the overlapping range in the streamline distance metric between data and model"""
    model_metric_for_min = jnp.where(model_finite_mask, dmetric_model, to_float64(BIG))
    model_metric_for_max = jnp.where(model_finite_mask, dmetric_model, to_float64(BIG_NEG))
    data_metric_for_min = jnp.where(data_finite_mask, dmetric_data, to_float64(BIG))
    data_metric_for_max = jnp.where(data_finite_mask, dmetric_data, to_float64(BIG_NEG))

    model_min = jnp.min(model_metric_for_min)
    model_max = jnp.max(model_metric_for_max)
    data_min = jnp.min(data_metric_for_min)
    data_max = jnp.max(data_metric_for_max)

    overlap_min = jnp.maximum(model_min, data_min)
    overlap_max = jnp.minimum(model_max, data_max)
    return model_min, model_max, data_min, data_max, overlap_min, overlap_max

@jax.jit
def match_model_to_data_curve(ra_model, dec_model, v_model, valid_mask_model, ra_data, dec_data):
    """
    Extract model values corresponding to data positions using the distance metric from
    extract_streamline.get_distance_metric

    Method:
    1. Compute the distance metric for model and data points
    2. Apply finite masks
    3. Normalise both metrics to [0, 1] based on their finite ranges
    4. Map data normalised positions to model normalised positions
    5. Interpolate model RA, Dec, and velocity at the mapped positions

    Returns
    -------
    ra_model_interp, dec_model_interp, v_model_interp, valid, dmetric_model, matching_trace
        where valid is a boolean mask with shape len(original data), marking
        retained data points
    """
    ra_model = to_float64(ra_model)
    dec_model = to_float64(dec_model)
    v_model = to_float64(v_model)
    ra_data = to_float64(ra_data)
    dec_data = to_float64(dec_data)

    # get distance metrics
    dmetric_model, _ = extract_streamline.get_distance_metric(ra_model, dec_model)
    dmetric_data, _ = extract_streamline.get_distance_metric(ra_data, dec_data)

    model_valid = valid_mask_model

    data_valid = (
        jnp.isfinite(ra_data)
        & jnp.isfinite(dec_data)
        & jnp.isfinite(dmetric_data)
    )

    # # we also filter model to keep only model points with dmetric >= minimum of data dmetric
    # this is becuase the model shouldn't go further in than the innermost data point
    # as this is where we no longer observe the streamer
    d_data_valid = jnp.where(data_valid, dmetric_data, to_float64(BIG))
    data_min = jnp.min(d_data_valid)

    # enforce both constraints on model
    model_keep = model_valid.astype(bool) & (dmetric_model >= data_min)

    # weights: 0 = ignore, 1 = use. This is for jax/jit compatibility
    w_model = model_keep.astype(jnp.float64)

    d_model = jnp.where(model_keep, dmetric_model, 0.0)
    d_data  = jnp.where(data_valid, dmetric_data, 0.0)

    # ---- sort model using metric + weight penalty (pushes invalid points to the end) ----
    model_sort_key = d_model + (1.0 - w_model) * to_float64(BIG)
    model_idx = jnp.argsort(model_sort_key)

    d_model_s = d_model[model_idx]
    ra_s = ra_model[model_idx]
    dec_s = dec_model[model_idx]
    v_s = v_model[model_idx]
    w_model_s = w_model[model_idx]
 
    # stats for trace and interpolation domain
    data_min_eff = jnp.min(jnp.where(data_valid, dmetric_data, to_float64(BIG)))
    data_max_eff = jnp.max(jnp.where(data_valid, dmetric_data, to_float64(BIG_NEG)))
    model_min = jnp.min(jnp.where(model_keep, dmetric_model, to_float64(BIG)))
    model_max = jnp.max(jnp.where(model_keep, dmetric_model, to_float64(BIG_NEG)))


    model_span = model_max - model_min
    data_span = data_max_eff - data_min_eff
    model_span_safe = jnp.where(model_span > to_float64(0.0), model_span, to_float64(1.0))
    data_span_safe  = jnp.where(data_span  > to_float64(0.0), data_span,  to_float64(1.0))


    data_has_valid  = data_min_eff < to_float64(BIG)
    model_has_valid = model_min    < to_float64(BIG)
    both_valid      = data_has_valid & model_has_valid
    # normalise data metric
    d_data_norm = (d_data - data_min_eff) / data_span_safe
    d_goal = model_min + d_data_norm * model_span_safe

    # interpolate model at data points, using weights to ignore invalid model points 
    # by giving them huge distance values so they don't affect the interpolation
    xp = jnp.where(w_model_s > 0, d_model_s, to_float64(BIG))

    ra_interp = jnp.interp(d_goal, xp, ra_s)
    dec_interp = jnp.interp(d_goal, xp, dec_s)
    v_interp = jnp.interp(d_goal, xp, v_s)

    # things for trace
    valid = data_valid

    matching_trace = {
    "model_points_total": model_idx.size,
    "model_nan_count": jnp.sum(jnp.isnan(dmetric_model)),
    "model_valid_points": model_valid.sum(),
    "data_points_total": ra_data.size,
    "data_nan_count": jnp.sum(jnp.isnan(dmetric_data)),
    "data_valid_points": data_valid.sum(),
    "model_metric_min": model_min,
    "model_metric_max": model_max,
    "data_metric_min": data_min_eff,
    "data_metric_max": data_max_eff,
    "model_metric_span": model_span_safe,
    "data_metric_span": data_span_safe}

    return ra_interp, dec_interp, v_interp, valid, model_keep, dmetric_model, matching_trace

checked_matching = checkify.checkify(match_model_to_data_curve)


@jax.jit
def checked_match_model_to_data_curve(*args, **kwargs):
    """Wrapper around match_model_to_data_curve with checkify checks for errors (to remain jax compatible)"""
    errors, result = checked_matching(*args, **kwargs)
    errors.throw()
    return result

@jax.jit(static_argnames=("loss_method", "npoints"))
def chi2_loss(
    model_params,
    distance_pc,
    prepared_data,
    loss_method=0,
    npoints=10000,
):
    """Generates model via forward model and calculates loss between data and model.
    Returns (chi2_total, loss_trace, err) where err is a checkify"""
 
    loss_method = check_loss_method(loss_method)
 
    distance_pc = to_float64(distance_pc)
 
    ra_data = prepared_data.ra_data
    dec_data = prepared_data.dec_data
    v_data = prepared_data.v_data
    ra_sigma = prepared_data.ra_sigma_safe
    dec_sigma = prepared_data.dec_sigma_safe
    v_sigma = prepared_data.v_sigma_safe
 
    ra_model, dec_model, v_model, valid_mask_model, err = forward_model(model_params, distance_pc, npoints=npoints)
    valid_mask_model = valid_mask_model.astype(jnp.bool_)
 
    ra_model_interp, dec_model_interp, v_model_interp, valid, model_keep, dmetric_model, _ = (
        checked_match_model_to_data_curve(ra_model, dec_model, v_model, valid_mask_model, ra_data, dec_data)
    )
 
    dmetric_data = prepared_data.dmetric_data
    valid = jnp.asarray(valid, dtype=bool)
    valid_weights = valid.astype(jnp.float64)
 
    model_finite_mask = (
        jnp.isfinite(ra_model)
        & jnp.isfinite(dec_model)
        & jnp.isfinite(v_model)
        & jnp.isfinite(dmetric_model)
    )
 
 
    # Only compute chi2 on valid/retained data points 

    chi2_v = jnp.sum(valid_weights * (((v_data - v_model_interp) / v_sigma) ** 2))
 

    if loss_method == 0:
        chi2_ra = jnp.sum(valid_weights * (((ra_data - ra_model_interp) / ra_sigma) ** 2))
        chi2_dec = jnp.sum(valid_weights * (((dec_data - dec_model_interp) / dec_sigma) ** 2))
        chi2_total = chi2_ra + chi2_dec + chi2_v
    else:
        r_proj_data = prepared_data.r_proj_data
        theta_proj_data = prepared_data.theta_proj_data
        r_proj_model, theta_proj_model = extract_streamline.cartesian_to_polar(
            ra_model_interp,
            dec_model_interp,
        )
 
        dtheta = extract_streamline.wrap_to_pi(theta_proj_data - theta_proj_model)
 
        sigma_r = jnp.sqrt(ra_sigma**2 + dec_sigma**2)
        r_eps = to_float64(1e-8)
        r_safe = jnp.maximum(jnp.abs(r_proj_data), r_eps)
        sigma_theta = jnp.sqrt(((dec_data * ra_sigma)**2 + (ra_data * dec_sigma)**2)) / (r_safe**2)
        sigma_theta = jnp.maximum(sigma_theta, r_eps)
 
        chi2_r = jnp.sum(valid_weights * (((r_proj_data - r_proj_model) / sigma_r) ** 2))
        chi2_theta = jnp.sum(valid_weights * ((dtheta / sigma_theta) ** 2))
        chi2_total = chi2_r + chi2_theta + chi2_v
 

    data_finite_mask = (
        jnp.isfinite(ra_data)
        & jnp.isfinite(dec_data)
        & jnp.isfinite(dmetric_data)
    )
 
    model_min, model_max, data_min, data_max, overlap_min, overlap_max = distance_metric_overlap(
        dmetric_model,
        model_finite_mask,
        dmetric_data,
        data_finite_mask,
    )
 
    model_nan_count = jnp.sum(~model_finite_mask)
    model_points_total = ra_model.size
    model_valid_points = model_points_total - model_nan_count
 
    data_keep = data_finite_mask & (dmetric_data >= overlap_min) & (dmetric_data <= overlap_max)
    model_keep = model_finite_mask & (dmetric_model >= overlap_min) & (dmetric_model <= overlap_max)
 
    sort_idx = jnp.argsort(dmetric_model)
    d_model_sorted = dmetric_model[sort_idx]
    d_diff = jnp.diff(d_model_sorted)
    if d_diff.size > 0:
        model_metric_min_gap = jnp.min(d_diff)
        model_metric_near_tie_count = jnp.sum(jnp.abs(d_diff) <= 1e-8)
        model_metric_duplicate_count = jnp.sum(d_diff == 0.0)
        model_metric_non_monotonic_count = jnp.sum(d_diff < 0.0)
    else:
        model_metric_min_gap = to_float64(float('nan'))
        model_metric_near_tie_count = to_float64(0.0)
        model_metric_duplicate_count = to_float64(0.0)
        model_metric_non_monotonic_count = to_float64(0.0)
 
    model_metric_span = d_model_sorted[-1] - d_model_sorted[0] if d_model_sorted.size > 1 else to_float64(0.0)
 
    if loss_method == 0:
        chi2_components = {
            'chi2_ra': chi2_ra,
            'chi2_dec': chi2_dec,
            'chi2_v': chi2_v,
            'overlap_width': overlap_max - overlap_min,
            'chi2_total': chi2_total,
        }
    else:
        chi2_components = {
            'chi2_r': chi2_r,
            'chi2_theta': chi2_theta,
            'chi2_v': chi2_v,
            'overlap_width': overlap_max - overlap_min,
            'chi2_total': chi2_total,
        }
 
    matching_trace = {
        'model_points_total': model_points_total,
        'model_nan_count': model_nan_count,
        'model_valid_points': model_valid_points,
        'model_retained_count': jnp.sum(model_keep),
        'data_points_total': ra_data.size,
        'data_valid_points': jnp.sum(data_finite_mask),
        'data_retained_count': jnp.sum(data_keep),
        'overlap_metric_min': overlap_min,
        'overlap_metric_max': overlap_max,
        'model_metric_span': model_metric_span,
        'model_metric_min_gap': model_metric_min_gap,
        'model_metric_near_tie_count': model_metric_near_tie_count,
        'model_metric_duplicate_count': model_metric_duplicate_count,
        'model_metric_non_monotonic_count': model_metric_non_monotonic_count,
    }
 
    loss_trace = {
        'chi2_components': chi2_components,
        'matching': matching_trace,
        'loss_method': loss_method,
    }
    return chi2_total, loss_trace, err



InitialGuessResult = namedtuple('InitialGuessResult', [
    'model_params',
    'ra_model',
    'dec_model',
    'v_model',
    'ra_model_interp',
    'dec_model_interp',
    'v_model_interp',
    'valid',
    'chi2_total',
    'chi2_components',
])
 
 
def evaluate_initial_guess(
    initial_opt_params,
    fixed_params,
    data,
    uncertainties,
    distance_pc,
    n_elements=10,
    loss_method=0,
):
    """
    Run the forward model and compute chi2 loss for the initial parameter guess.
 
    Parameters
    ----------
    initial_opt_params : dict
        Initial guesses for the parameters to optimise.
    fixed_params : dict
        Fixed (non-optimised) parameters. Together with initial_opt_params
        this must provide a full, non-overlapping partition of
        STREAMLINE_MODEL_PARAM_KEYS.
    data : tuple of arrays (ra_data, dec_data, v_data)
        Observed RA offset (arcsec), Dec offset (arcsec), velocity (km/s).
    uncertainties : tuple of arrays (ra_sigma, dec_sigma, v_sigma)
        Uncertainties on the data.
    distance_pc : float
        Distance to source in parsecs.
    n_elements : int
        Number of distance-metric partitions, i.e. the number of 1D data
        points. Must match the value used when reducing the cube.
    loss_method : int
        Loss definition to use. Options:
        - 0: radecvel — RA, Dec, and velocity residuals.
        - 1: rthetavel — radial distance, polar angle, and velocity residuals.
 
    Returns
    -------
    InitialGuessResult
        Named tuple with entries:
        - model_params       : merged dict of all model parameters (float64)
        - ra_model           : full model RA offsets (arcsec)
        - dec_model          : full model Dec offsets (arcsec)
        - v_model            : full model velocities (km/s)
        - ra_model_interp    : model RA interpolated at data positions
        - dec_model_interp   : model Dec interpolated at data positions
        - v_model_interp     : model velocity interpolated at data positions
        - valid              : boolean mask of retained data points
        - chi2_total         : total chi2 loss (float)
        - chi2_components    : dict of per-component chi2 values and chi2_total
    """
    loss_method = check_loss_method(loss_method)
 
    model_params, _, _ = prepare_model_params(initial_opt_params, fixed_params)
 
    ra_model, dec_model, v_model, valid_mask_model, err = forward_model(
        model_params, distance_pc
    )
    err.throw()
 
    ra_model_interp, dec_model_interp, v_model_interp, valid, _, _, _ = (
        checked_match_model_to_data_curve(
            ra_model, dec_model, v_model, valid_mask_model,
            jnp.asarray(data[0], dtype=jnp.float64),
            jnp.asarray(data[1], dtype=jnp.float64),
        )
    )
 
    prepared_data = extract_streamline.prepare_data(data, uncertainties, n_elements=n_elements)
    chi2_total, loss_trace, _ = chi2_loss(
        model_params, distance_pc, prepared_data, loss_method=loss_method
    )
    chi2_components = loss_trace['chi2_components']
 
    n_valid = int(jnp.sum(valid))
    n_total = len(valid)
 
    return InitialGuessResult(
        model_params=model_params,
        ra_model=ra_model,
        dec_model=dec_model,
        v_model=v_model,
        ra_model_interp=ra_model_interp,
        dec_model_interp=dec_model_interp,
        v_model_interp=v_model_interp,
        valid=valid,
        chi2_total=float(chi2_total),
        chi2_components={k: float(v) for k, v in chi2_components.items()},
    )


def fit_streamline(initial_opt_params, fixed_params, streamer, distance_pc,
                   learning_rate=0.001, param_bounds=None, n_epochs=1000,
                   beta1=0.9, beta2=0.999,
                   info_every=100, loss_threshold=None, loss_threshold_epochs=1,
                   gradient_tol=None, gradient_tol_epochs=1,
                   early_stopping_patience=50,
                   save_folder='sting_results',
                   loss_method=0, # 0: radecvel, 1: rthetavel
                   v_lsr=None,
                   show_plots=False
                   ):
    """
    Fit streamline model parameters to data using Adam optimiser.
    Any supported streamline parameter can be optimised or fixed.
    Parameters are split by dictionary:
    - keys in initial_opt_params are optimised
    - keys in fixed_params are held fixed
    The union must contain each key in STREAMLINE_MODEL_PARAM_KEYS exactly once.
    
    Parameters:
    -----------
    initial_opt_params : dict
        Initial guesses for the parameters to optimise.
        Allowed keys are STREAMLINE_MODEL_PARAM_KEYS.
    fixed_params : dict
        Fixed (non-optimised) parameters using the same key space.
        Together with initial_opt_params, this must provide a full,
        non-overlapping partition of STREAMLINE_MODEL_PARAM_KEYS.
    streamer: NamedTuple with fields:
        pc_coords, ra_data, dec_data, v_data, ra_sigma, dec_sigma, v_sigma, data, uncertainties 
    data : tuple of arrays (ra_data, dec_data, v_data)
        Observed RA offset (arcsec), Dec offset (arcsec), velocity (km/s)
    uncertainties : tuple of arrays (ra_sigma, dec_sigma, v_sigma)
        Uncertainties on the data
    distance_pc : float
            Distance to source in parsecs
    learning_rate : float
        Adam learning rate applied uniformly to all normalised parameters.
    param_bounds : dict or None
        Parameter bounds in physical/log parameter units.
        optimisation is performed in normalised space using
        x_norm = (x - min) / (max - min), so bounds are required for all
        optimised keys and are used as normalisation anchors.
        You may provide 'omega' bounds as linear bounds; these are converted
        to 'log_omega' bounds internally.
    n_epochs : int
        Maximum number of optimisation iterations
    beta1 : float
        Adam exponential decay rate for first moment
    beta2 : float
        Adam exponential decay rate for second moment
    info_every : int
        Print loss every N epochs
    early_stopping_patience : int
        Stop if loss doesn't improve for N epochs
    save_folder : str
        Folder to save output CSV and trace files, and figures. Created if it doesn't exist.
    loss_method : int
        Loss definition to use. Options:
        - 0: radecvel: optimise RA, Dec, and velocity residuals.
        - 1: rthetavel: optimise radial distance, polar angle, and velocity residuals.
        Both options use the same model-data matching and overlap penalty.
    v_lsr : float or None
        Systemic velocity (km/s). When provided, drawn as a reference line on the best-fit
        velocity-radius plot
    loss_threshold : float or None
        Optional absolute loss threshold for threshold-based stopping.
        If provided, optimisation stops after loss is <= loss_threshold for
        loss_threshold_epochs consecutive epochs.
    loss_threshold_epochs : int
        Number of consecutive epochs with loss <= loss_threshold required to
        trigger threshold-based early stopping. Must be >= 1.
    gradient_tol : float or None
        Optional gradient norm tolerance for stopping in normalised space.
        If provided, optimisation stops when the L2 norm of gradients with
        respect to normalised parameters
        is less than this threshold for gradient_tol_epochs consecutive epochs,
        indicating convergence.
    gradient_tol_epochs : int
        Number of consecutive epochs with ||grad|| < gradient_tol required to
        trigger normalised-space gradient norm-based early stopping. Must be >= 1.
    show_plots : bool
        Whether to show diagnostic plots during optimisation
        
    Epoch 0: initial state before any updates, with initial_opt_params
    Epoch n (n>=1): state after applying parameter update n
    Tracking and checks are all performed at the end of each epoch. So e.g. loss n = loss after applying update n, using the updated parameters

    Returns:
    --------   
    dict: optimised parameters (same keys as initial_opt_params), also including
        derived 'omega' when 'mu', 'mass', and 'r0' are available
    list: Loss history (indexed by epoch: loss_history[i] = loss at epoch i)
    """
    # lazy imports to avoid circular imports
    from . import outputs
    from . import errors
    # Initialize parameters
    loss_method = check_loss_method(loss_method)

    opt_params, fixed_params = sanitize_param_partition(
        initial_opt_params,
        fixed_params,
        require_nonempty_opt=True,
    )


    param_bounds = standardise_param_bounds(param_bounds)
    param_bounds = convert_and_strip_bound_units(param_bounds)
    
    # we perform optimisation in mu-space when either rc or omega is present. conversion is here
    # rotation_key records which of 'rc', 'omega', or 'mu' is input as rotation parameter by user, 
    # so we know which one to convert back to at the end
    opt_params, fixed_params, param_bounds, rotation_key = with_mu_substituted(opt_params, fixed_params, param_bounds)  


    opt_param_keys = list(opt_params.keys())
    data = make_data_tuple_float64(streamer.data)
    uncertainties = make_data_tuple_float64(streamer.uncertainties)
    distance_pc = to_float64(distance_pc)
    learning_rate = to_float64(learning_rate)
    if not bool(jnp.isfinite(learning_rate)):
        raise ValueError(f'learning_rate must be finite. Got {learning_rate}.')
    if not bool(learning_rate > 0):
        raise ValueError(f'learning_rate must be > 0. Got {float(learning_rate)}.')
    normalisation_spec = build_normalisation_spec(opt_params, param_bounds)

    # Keep optimisation variables in normalised coordinates; convert back to
    # physical/log units only when evaluating the forward model and diagnostics.
    opt_params_norm = normalise_opt_params(opt_params, normalisation_spec)

    # Use one global learning rate on normalised parameters
    solver = optax.adam(learning_rate=learning_rate, b1=beta1, b2=beta2)

    opt_state = solver.init(opt_params_norm)

    # Precompute data-only quantities once before optimisation loop
    prepared_data = extract_streamline.prepare_data(data, uncertainties, n_elements=len(data[0]))
    # npoints for forward model evaluation: fixed large number set by max r0 bound and deltar
    # this is necessary to ensure forward model has constant array lengths for jax/jit compatability
    if 'r0' in param_bounds:
        max_r0 = param_bounds['r0'][1]
        deltar = fixed_params['deltar'] if 'deltar' in fixed_params else 1.0
        npoints = int(jnp.ceil(max_r0 / deltar))
    else: 
        npoints = 50000

    fixed_params_for_core = fixed_params

    @jax.jit
    def loss_from_normalised(norm_opt_params):
        physical_opt_params = denormalise_opt_params(norm_opt_params, normalisation_spec)
        model_params = {**fixed_params_for_core, **physical_opt_params}
        chi2_total, loss_trace, err = chi2_loss(
            model_params,
            distance_pc,
            prepared_data,
            loss_method=loss_method,
            npoints=npoints,
        )
        return chi2_total, (loss_trace, err)


    # Create gradient functions in normalised space.
    loss_and_grad_fn = value_and_grad(loss_from_normalised, has_aux=True)

    
    # Track loss history
    loss_history = []
    initial_loss, (_, initial_err) = loss_from_normalised(opt_params_norm)

    # raise any initial errors
    initial_error_message = get_checkify_error_message(initial_err)
    if initial_error_message is not None:
        raise ValueError(
            f"Initial loss computation failed with error: {initial_error_message}. "
        )
    
    initial_loss = float(initial_loss)
    loss_history.append(initial_loss) # 'epoch 0' loss (initial state, before any updates)
    best_loss = initial_loss
    best_opt_params = opt_params.copy()
    best_opt_params_norm = opt_params_norm.copy()
    best_epoch = 0
    patience_counter = 0
    loss_threshold_counter = 0
    gradient_tol_counter = 0
    ordered_best_opt_params = {k: best_opt_params[k] for k in opt_param_keys}

    if loss_threshold is not None:
        loss_threshold = float(loss_threshold)
        if not math.isfinite(loss_threshold):
            raise ValueError('loss_threshold must be finite when provided.')
        if loss_threshold_epochs < 1:
            raise ValueError('loss_threshold_epochs must be >= 1 when loss_threshold is provided.')
    
    if gradient_tol is not None:
        gradient_tol = float(gradient_tol)
        if not math.isfinite(gradient_tol):
            raise ValueError('gradient_tol must be finite when provided.')
        if gradient_tol <= 0:
            raise ValueError('gradient_tol must be positive when provided.')
        if gradient_tol_epochs < 1:
            raise ValueError('gradient_tol_epochs must be >= 1 when gradient_tol is provided.')
    
    # initialise log and trace files if output_folder is provided
    log_file = None
    log_writer = None
    trace_file = None
    trace_writer = None
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

        log_file = os.path.join(save_folder, 'optimisation_log.csv')
        log_file = open(log_file, 'w', newline='')
        # Create header: epoch, loss, then all optimisable params
        fieldnames = ['epoch', 'loss'] + opt_param_keys
        all_param_keys = set(opt_param_keys) | set(fixed_params.keys())
        if 'mu' in all_param_keys:
            # also log derived rc and omega when mu is present for convenience in comparing to old codes
            if 'rc' not in fieldnames:
                fieldnames.append('rc')
            if 'omega' not in fieldnames:
                fieldnames.append('omega')
        log_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        log_writer.writeheader()
        log_file.flush()

        trace_file = os.path.join(save_folder, 'optimisation_trace.csv')
        trace_file = open(trace_file, 'w', newline='')
        trace_writer = csv.DictWriter(
            trace_file,
            fieldnames=trace_fieldnames_for_loss_method(loss_method),
        )
        trace_writer.writeheader()
        trace_file.flush()
    
    print(f"Starting optimisation with {n_epochs} epochs...")
    print(f"Loss method: {loss_method}")
    print(f"optimising parameters: {opt_param_keys}")
    print(f"Fixed parameters: {list(fixed_params.keys())}")
    if loss_threshold is not None:
        print(
            f"Threshold-based stopping enabled: loss <= {loss_threshold:.6g} "
            f"for {loss_threshold_epochs} consecutive epochs."
        )
    if gradient_tol is not None:
        print(
            f"Gradient norm stopping enabled (normalised space): ||grad|| < {gradient_tol:.6g} "
            f"for {gradient_tol_epochs} consecutive epochs."
        )
    print(f"Initial optimisable values:")
    for key in opt_param_keys:
        print(f"  {key}: {format_param(key, opt_params[key])}")
    print(f"Initial loss: {initial_loss:.6g}")
    
    # Log epoch 0: initial state (before any updates)
    initial_loss = float(initial_loss)
    if log_writer is not None:
        row = {'epoch': 0, 'loss': initial_loss}
        for key in opt_param_keys:
            row[key] = float(opt_params[key])
        row = add_rc_omega_to_log(row, opt_params, fixed_params, all_param_keys)
        log_writer.writerow(row)
        log_file.flush()
    
    # Log epoch 0 trace if trace file is requested
    if trace_writer is not None:
        # Compute initial loss and trace
        (loss_value_trace, (loss_trace_raw, _)), norm_grads_trace = loss_and_grad_fn(opt_params_norm)
        loss_trace = trace_tree_to_python(loss_trace_raw)
        grad_norm = float(gradient_l2_norm(norm_grads_trace))
        
        # Build and write trace row for epoch 0
        trace_row = build_trace_row(0, float(loss_value_trace), loss_trace, grad_norm, loss_method)
        trace_writer.writerow(trace_row)
        trace_file.flush()

    
    try:
        for epoch in range(1, n_epochs + 1):
            if epoch % info_every == 0:
                print(f"\n Starting Epoch {epoch} -------------------------")
            # Compute loss and gradients at pre-update normalised parameters.
            loss_trace = None
            (loss_before, _), norm_grads = loss_and_grad_fn(opt_params_norm)


            # Perform Optax Adam step in normalised space (apply update).
            updates, opt_state = solver.update(norm_grads, opt_state, params=opt_params_norm)
            opt_params_norm = optax.apply_updates(opt_params_norm, updates)

            # Enforce normalised bounds and map back to physical/log values.
            for key in opt_param_keys:
                if key == 'phi0':
                    # phi0 is cyclic; wrap to [0, 1) in normalised space
                    opt_params_norm[key] = jnp.mod(opt_params_norm[key], 1.0)
                elif key == 'rc':
                    # must be positive >0
                    opt_params_norm[key] = jnp.clip(opt_params_norm[key], to_float64(1e-6), 1.0)
                elif key == 'omega':
                    # must be positive >0
                    opt_params_norm[key] = jnp.clip(opt_params_norm[key], to_float64(1e-6), 1.0)
                elif key == 'v_r0':
                    #already dealt with
                    continue
                else:
                    opt_params_norm[key] = jnp.clip(opt_params_norm[key], 0.0, 1.0)

            # Gradient-aware epsilon protection for v_r0 near zero:
            # When v_r0 is very close to zero, use the sign of the gradient to determine
            # which direction to protect towards, allowing the optimiser to continue smoothly.
            # if 'v_r0' in opt_param_keys:
            #     threshold_norm = to_float64(1e-12)  # normalised space threshold
            #     v_r0_norm_val = opt_params_norm['v_r0']
            #     if bool(jnp.all(jnp.abs(v_r0_norm_val) < threshold_norm)):
            #         # v_r0 is very close to zero; check gradient direction
            #         grad_v_r0 = norm_grads['v_r0']
            #         # In gradient descent, we move opposite to gradient:
            #         protect_sign = -jnp.sign(grad_v_r0)
            #         # Default to positive if gradient is exactly zero
            #         protect_sign = jnp.where(protect_sign == 0, 1.0, protect_sign)
            #         # Set v_r0 to small epsilon in the gradient-indicated direction
            #         epsilon_norm = threshold_norm
            #         opt_params_norm['v_r0'] = protect_sign * epsilon_norm

            # Now materialize physical parameters from the (possibly clamped)
            # normalised parameters.
            opt_params = denormalise_opt_params(opt_params_norm, normalisation_spec)

            # Compute loss and gradient at the post-update state S(epoch) for logging 
            (loss_value, (loss_trace_raw, err)), norm_grads = loss_and_grad_fn(opt_params_norm)
            # print the gradients by parameter for debugging
            grad_norm = float(gradient_l2_norm(norm_grads))

            # raise any errors
            error_message = get_checkify_error_message(err)
            if error_message is not None:
                print(
                    f"\nStopping at epoch {epoch}: {error_message} "
                )
                break


            loss_trace = trace_tree_to_python(loss_trace_raw)
            loss_value = float(loss_value)


            # Log post-update state for this epoch
            if log_writer is not None:
                row = {'epoch': epoch, 'loss': loss_value}
                for key in opt_param_keys:
                    row[key] = float(opt_params[key])
                row = add_rc_omega_to_log(row, opt_params, fixed_params, all_param_keys)
                log_writer.writerow(row)
                log_file.flush()
        
            # Track loss (store the loss for the loss_history)
            loss_history.append(loss_value)

            if trace_writer is not None and loss_trace is not None:
                # Use the post-update loss value for trace logging
                trace_row = build_trace_row(epoch, loss_value, loss_trace, grad_norm, loss_method)
                trace_writer.writerow(trace_row)
                trace_file.flush()
        
            # Early stopping checks (use post-update loss)
            if loss_value < best_loss:
                best_loss = loss_value
                best_opt_params = opt_params.copy()
                best_opt_params_norm = opt_params_norm.copy()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if loss_threshold is not None:
                if loss_value <= loss_threshold:
                    loss_threshold_counter += 1
                else:
                    loss_threshold_counter = 0
            
            if gradient_tol is not None:
                if grad_norm < gradient_tol:
                    gradient_tol_counter += 1
                else:
                    gradient_tol_counter = 0
        
            # Print progress
            if epoch % info_every == 0:
                if gradient_tol is not None:
                    print(f'Epoch {epoch}/{n_epochs}, Loss: {loss_value:.6f}, Best Loss: {best_loss:.6f}, ||grad||: {grad_norm:.6e}')
                else:
                    print(f'Epoch {epoch}/{n_epochs}, Loss: {loss_value:.6f}, Best Loss: {best_loss:.6f}')

            # Early stopping conditions (any one is sufficient to stop)
            if loss_threshold is not None and loss_threshold_counter >= loss_threshold_epochs:
                print(
                    f"\nEarly stopping at epoch {epoch}: loss <= {loss_threshold:.6g} "
                    f"for {loss_threshold_epochs} consecutive epochs"
                )
                break

            if gradient_tol is not None and gradient_tol_counter >= gradient_tol_epochs:
                print(
                    f"\nEarly stopping at epoch {epoch}: normalised gradient norm {grad_norm:.6e} < {gradient_tol:.6e} "
                    f"for {gradient_tol_epochs} consecutive epochs"
                )
                break
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}: no improvement for {early_stopping_patience} epochs")
                break
    
        
        # restore canonical parameter order before returning
        ordered_best_opt_params = {k: best_opt_params[k] for k in opt_param_keys}

    finally:
        # Always close the CSV file if it was opened
        if log_file is not None:
            log_path = log_file.name
            log_file.close()
            print(f"Optimisation log saved to: {log_path}")
        if trace_file is not None:
            trace_path = trace_file.name
            trace_file.close()
            print(f"Matching trace log saved to: {trace_path}")

    print(f"Optimisation complete!")
    print(f"Best-fit parameters found at epoch: {best_epoch}, with loss: {best_loss:.6f}")

    # compute errors on best-fit parameters
    print("\nEstimating parameter uncertainties from Hessian...")
    param_errors = None
    cov_matrix = None
    cov_transformed_dict = None
    # only input the rotation key if if actually needs transforming back from mu
    key_needs_transform = rotation_key if rotation_key in ('rc', 'omega') else None
    try:
        param_errors, cov_matrix, cov_transformed_dict = errors.estimate_parameter_errors(
            ordered_best_opt_params,
            fixed_params,
            distance_pc,
            prepared_data,
            loss_method=loss_method,
            gradient_tol=gradient_tol,
            normalisation_spec=normalisation_spec,
            best_norm_opt_params=best_opt_params_norm,
            rotation_key=key_needs_transform,
            npoints=npoints,
        )
    except Exception as e:
        print(f"\nWarning: parameter uncertainty estimation failed: ({e}).")
        traceback.print_exc()
        print("Continuing without error estimates")

    display_opt_params = dict(ordered_best_opt_params)
    display_fixed_params = dict(fixed_params)
    display_param_errors = dict(param_errors) if param_errors is not None else None

    if cov_transformed_dict is not None and key_needs_transform is not None and display_param_errors is not None:
        if key_needs_transform in cov_transformed_dict['keys']:
            all_params_for_transform = {**display_fixed_params, **display_opt_params}
            mu_best = float(ordered_best_opt_params['mu'])
            mass_val = float(all_params_for_transform['mass'])
            r0_val   = float(all_params_for_transform['r0'])
            display_opt_params[key_needs_transform] = rotation_param_from_mu(key_needs_transform, mu_best, mass_val, r0_val)
            display_opt_params.pop('mu', None)
            display_param_errors[key_needs_transform] = cov_transformed_dict['errors'][key_needs_transform]
            display_param_errors.pop('mu', None)

    print("\nFinal parameters at best-fit:")
    all_display_params = {**display_fixed_params, **display_opt_params}
    for key in all_display_params.keys():
        value = all_display_params[key]
        if display_param_errors is not None and key in display_param_errors:
            error = display_param_errors[key]
            print(f"  {key}: {format_param(key, value)} ± {format_param(key, error)}")
        else:
            print(f"  {key}: {format_param(key, value)}")

    outputs.save_best_fit_params(display_opt_params, display_fixed_params, display_param_errors, save_folder=save_folder)

    # now we will make some plots of the results
    if save_folder is not None:
        print("\nMaking diagnostic plots...")
        outputs.plot_fitting_results(
            ordered_best_opt_params,
            opt_param_keys,
            fixed_params,
            streamer,
            distance_pc,
            loss_history,
            param_errors=param_errors,
            cov_matrix=cov_matrix,
            v_lsr=v_lsr,
            save_folder=save_folder,
            show_plots=show_plots,
            transformed_cov_result=cov_transformed_dict,
        )





    return display_opt_params, loss_history, display_param_errors