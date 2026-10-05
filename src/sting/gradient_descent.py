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
import types
jax.config.update("jax_enable_x64", True)

# settings and constants
VR0_MIN = 1e-6
BIG = 1e30
BIG_NEG = -1e30

LOSS_METHOD_CHOICES = [0, 1]
MATCHING_METHOD_CHOICES = ('continuous', 'continuous_point_cloud', 'integrated', 'integrated_point_cloud', 'legacy')
INTEGRATED_MATCHING_METHODS = ('integrated', 'integrated_point_cloud')
POINT_CLOUD_MATCHING_METHODS = ('continuous_point_cloud', 'integrated_point_cloud')
MATCHING_ITERATIONS = 48
INTEGRATION_NODES = 128  # number of fixed in-plane angle nodes for the integrated methods
INTEGRATION_BATCH_SIZE = 2048  # data points processed together in the integrated loss, to bound memory
INTEGRATION_RESOLUTION_WARN = 1.0  # warn if matched segments are longer than this many sigma

LOSS_METHOD_COMPONENT_KEYS = {
    0: ('chi2_ra', 'chi2_dec', 'chi2_v', 'chi2_prior'), #radecvel
    1: ('chi2_r', 'chi2_theta', 'chi2_v', 'chi2_prior'), #rthetavel
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

# The only optimisable parameters whose range depends on the source, so the user must supply bounds
USER_BOUNDED_PARAM_KEYS = ('r0', 'mass')

# Normalisation of every other optimisable parameter is fixed here (canonical units), so no user bounds are needed:
# - 'bounds': (min, max) set by physics. Normalised to [0, 1] and clipped during optimisation,
#   or wrapped if 'cyclic' is True (azimuthal angles, where min and max are the same point).
# - 'scale': no natural (min, max) range. Normalised as value/scale. Adam steps are ~learning_rate
#   in normalised units, so this sets the step size. An optional one-sided 'min' is enforced by clipping.
AUTO_NORMALISATION = {
    'theta0': {'bounds': (0.0, jnp.pi)},
    'phi0': {'bounds': (0.0, 2 * jnp.pi), 'cyclic': True},
    'inc': {'bounds': (-jnp.pi/2, jnp.pi/2)},
    'pa': {'bounds': (0.0, 2 * jnp.pi), 'cyclic': True},
    'mu': {'bounds': (0.0+1e-6, 1.0-1e-6)},  # tiny epsilon to avoid rc=r0 or rc=0
    'v_r0': {'scale': 1.0, 'min': 0.0},  # km/s. Infall only. The model is smooth through v_r0 = 0, so the bound is safe to sit on
    'v_lsr': {'scale': 1.0},  # km/s
}

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
    'spin',
)

#### Return types

# Contains all the information about the covariance matrix and errors of the fitted model
CovarianceResult = namedtuple(
    'CovarianceResult',
    [
        'covariance',       # 2-D array: physical-space covariance matrix from the Hessian, ordered consistently with opt_keys / best_params / fixed_params.
        'opt_keys',         # list[str]: parameter names for rows/cols of covariance (mu-substituted, i.e. 'mu' in place of 'rc'/'omega').
        'best_opt_params',  # dict: best-fit values in the mu-substituted parameterisation.
        'fixed_params',     # dict: fixed parameters in the mu-substituted parameterisation.
        'param_errors',     # dict: 1-sigma errors keyed by opt_keys names, or None.
        'transformed_cov',  # dict or None: Jacobian-transformed result when 'mu' was substituted for 'rc'/'omega'; keys are 'keys', 'cov', 'errors'.
    ]
)

ContinuousMatchResult = namedtuple(
    'ContinuousMatchResult',
    ['best_u', 'best_radius', 'ra_model_matched', 'dec_model_matched',
     'v_model_matched', 'valid', 'residual_ra', 'residual_dec', 'residual_v'],
)

# Contains all the information about the model fit result, including best-fit parameters
FitResult = namedtuple(
    'FitResult',
    [
        'best_opt_params',      # dict: best-fit optimised parameters in the original user-supplied parameterisation (rc/omega restored).
        'loss_history',     # list[float]: loss value at every epoch.
        'param_errors',     # dict or None: 1-sigma errors in display parameterisation, or None if uncertainty estimation failed. Parameters in at_bound are left out.
        'covariance_result',       # CovarianceResult or None: full covariance information needed for sampling, or None if estimation failed.
        'at_bound',         # dict: {display param name: 'lower' or 'upper'} for parameters that finished on a bound with the loss pushing past it.
    ]
)

def convert_and_strip_bound_units(bounds):
    """
    Convert bounds that are astropy quantitiesinto canonical units,
    then strip the units.

    If input is already plain numeric, assume it's already in canonical units.

    Required as JAX optimiser works with unitless arrays.
    """
    if bounds is None:
        return {}
    
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


def check_matching_method(matching_method):
    """Validate the selected matching mode and return it."""
    if matching_method not in MATCHING_METHOD_CHOICES:
        raise ValueError(
            f"Unknown matching_method '{matching_method}'. "
            f"Choose from: {', '.join(repr(m) for m in MATCHING_METHOD_CHOICES)}."
        )
    return matching_method


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

def get_checkify_error_message(err):
    """Extract human-readable error message from a checkify.Error if possible,
    or None if it doesn't contain anything"""
    if err is None:
        return None
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
    Note: exactly one of 'rc' or 'omega' must be present across initial_opt_params and fixed_params, to determine mu=rc/r0.
    The rotation parameter ('rc', 'omega' or 'mu') is a magnitude and must be positive.
    'spin' (+1 co-rotating, -1 counter-rotating) is optional and can only be fixed. If it is missing,
    fixed_params['spin'] = +1 is added."""
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
    
    # spin is a discrete choice (no gradient), so it can't be optimised. It defaults to +1 (co-rotating)
    if 'spin' in opt_params:
        raise ValueError(
            "spin is a discrete choice (+1 or -1) and cannot be optimised. Put it in fixed_params, "
            "and run both senses to compare chi2."
        )
    if 'spin' not in fixed_params:
        fixed_params['spin'] = to_float64(1.0)
    spin = fixed_params['spin']
    if spin is None or not is_numeric_value(spin) or jnp.ndim(spin) != 0 or float(spin) not in (1.0, -1.0):
        raise ValueError(f"spin must be +1 (co-rotating) or -1 (counter-rotating). Got {spin!r}.")

    # the rotation parameter is a magnitude. The sense of rotation is set by spin
    rotation_key = rotation_keys_present[0]
    rotation_value = opt_params.get(rotation_key, fixed_params.get(rotation_key))
    if is_numeric_value(rotation_value) and not bool(jnp.all(to_float64(rotation_value) > 0)):
        if rotation_key == 'omega':
            description = "omega is the magnitude of the angular velocity"
        elif rotation_key == 'rc':
            description = "rc is the centrifugal radius"
        else:
            description = "mu = rc/r0 is a ratio of radii"
        raise ValueError(
            f"{description} and must be positive. Got {rotation_key} = {rotation_value!r}. "
            "To model counter-rotation, set spin = -1 in fixed_params."
        )

    # check all other required parameters are present (except mu, rc, omega and spin which we already dealt with)
    already_dealt_with = {'rc', 'omega', 'mu', 'spin'}
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
    """Build offset and scale for normalisation of optimised parameters.
    r0 and mass are normalised from the user-supplied bounds. All other parameters use AUTO_NORMALISATION,
    and any bounds the user supplied for them are ignored (with a notice)."""
    param_bounds = {} if param_bounds is None else param_bounds

    not_optimisable = [key for key in opt_params if key not in USER_BOUNDED_PARAM_KEYS and key not in AUTO_NORMALISATION]
    if not_optimisable:
        raise ValueError(
            f"Parameters {not_optimisable} cannot be optimised. Please move them to fixed_params."
        )

    missing = [key for key in opt_params if key in USER_BOUNDED_PARAM_KEYS and key not in param_bounds]
    if missing:
        raise ValueError(
            "Missing bounds for optimised parameters: "
            f"{missing}. Please add (min, max) entries in param_bounds for {list(USER_BOUNDED_PARAM_KEYS)} if you want to optimise them."
        )

    ignored = sorted(key for key in param_bounds if key not in USER_BOUNDED_PARAM_KEYS)
    if ignored:
        print(
            f"Notice: Ignoring supplied bounds for {ignored}. Bounds are only needed for {list(USER_BOUNDED_PARAM_KEYS)}; "
            "the normalisation of all other parameters is set automatically."
        )

    normalisation_spec = {}
    for key, value in opt_params.items():
        auto = AUTO_NORMALISATION.get(key, {})
        if 'scale' in auto:
            scale = to_float64(auto['scale'])
            lower_bound = to_float64(auto.get('min', -jnp.inf))
            if bool(to_float64(value) < lower_bound):
                raise ValueError(
                    f"Initial value for '{key}' ({float(value)}) is below its minimum ({float(lower_bound)})."
                )
            normalisation_spec[key] = {
                'offset': to_float64(0.0),
                'scale': scale,
                'clip_min': lower_bound / scale,
                'clip_max': to_float64(jnp.inf),
                'cyclic': False,
            }
            continue

        bounds = auto['bounds'] if 'bounds' in auto else param_bounds[key]
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

        cyclic = bool(auto.get('cyclic', False))
        value = to_float64(value)
        if not cyclic and not bool((value >= lower_bound) & (value <= upper_bound)):
            raise ValueError(
                f"Initial value for '{key}' ({float(value)}) is outside bounds"
                f"({float(lower_bound)}, {float(upper_bound)})."
            )

        normalisation_spec[key] = {
            'offset': lower_bound,
            'scale': upper_bound - lower_bound,
            'clip_min': to_float64(0.0),
            'clip_max': to_float64(1.0),
            'cyclic': cyclic,
        }

    return normalisation_spec


def get_physical_bounds(opt_keys, param_bounds=None):
    """(min, max) bounds in canonical units for each optimised parameter that has them:
    user-supplied bounds for r0 and mass, and the AUTO_NORMALISATION bounds for the rest.
    Parameters with only a minimum (v_r0) get (min, inf); parameters with neither (v_lsr) are left out.
    Cyclic parameters (see is_cyclic_param) should be wrapped into these bounds rather than clipped."""
    param_bounds = convert_and_strip_bound_units(param_bounds)
    physical_bounds = {}
    for key in opt_keys:
        if key in USER_BOUNDED_PARAM_KEYS:
            if key in param_bounds:
                physical_bounds[key] = param_bounds[key]
        elif 'bounds' in AUTO_NORMALISATION.get(key, {}):
            physical_bounds[key] = tuple(float(b) for b in AUTO_NORMALISATION[key]['bounds'])
        elif 'min' in AUTO_NORMALISATION.get(key, {}):
            physical_bounds[key] = (float(AUTO_NORMALISATION[key]['min']), math.inf)
    return physical_bounds


def is_cyclic_param(key):
    """True for parameters that wrap round their bounds (phi0, pa)"""
    return bool(AUTO_NORMALISATION.get(key, {}).get('cyclic', False))


def normalise_opt_params(opt_params, normalisation_spec):
    """normalise optimised parameters to [0, 1] (parameters with (min, max) bounds; others are just rescaled).
    Cyclic parameters are wrapped into [0, 1), so e.g. pa = -10 deg is accepted as 350 deg."""
    normalised = {}
    for key, value in opt_params.items():
        offset = normalisation_spec[key]['offset']
        scale = normalisation_spec[key]['scale']
        normalised[key] = (to_float64(value) - offset) / scale
        if normalisation_spec[key]['cyclic']:
            normalised[key] = jnp.mod(normalised[key], 1.0)
    return normalised


def denormalise_opt_params(norm_opt_params, normalisation_spec):
    """Convert normalised optimised parameters back to physical/log parameter values"""
    denormalised = {}
    for key, value in norm_opt_params.items():
        offset = normalisation_spec[key]['offset']
        scale = normalisation_spec[key]['scale']
        if normalisation_spec[key]['cyclic']:
            denormalised[key] = offset + jnp.mod(to_float64(value), 1.0) * scale
        else:
            denormalised[key] = to_float64(value) * scale + offset
    return denormalised

def log_header(param_key):
    if param_key == 'mu':
        return 'mu'
    unit = CANONICAL_UNITS.get(param_key)
    return f'{param_key} [{unit}]' if unit is not None else param_key



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

def with_mu_substituted(opt_params, fixed_params):
    """ Replace the user's input rotation parameter (either rc or omega) with mu, which is the parameter used internally for the physics calculations and optimisation,
    because it has obvious bounds (0,1) that will mean that optimisation won't explore regions where rc > r0."""
    rotation_key = get_rotation_param_key(opt_params, fixed_params)
    opt_params = dict(opt_params)
    fixed_params = dict(fixed_params)

    all_params = {**fixed_params, **opt_params}
    mass = all_params['mass']
    r0 = all_params['r0']

    if rotation_key in opt_params:
        rotation_value = opt_params[rotation_key]
        mu_value = mu_from_rotation_param(rotation_key, rotation_value, mass, r0)
        del opt_params[rotation_key]
        opt_params['mu'] = mu_value
    else:
        # rotation parameter is in fixed params. so just rename it to 'mu' in fixed params for consistency
        rotation_value = fixed_params[rotation_key]
        mu_value = mu_from_rotation_param(rotation_key, rotation_value, mass, r0)
        del fixed_params[rotation_key]
        fixed_params['mu'] = mu_value
    
    return opt_params, fixed_params, rotation_key

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
    if key == 'spin':
        return f"{val:+.0f}"
    unit = DISPLAY_UNITS.get(key, '')
    if unit:
        suffix = f" {unit}"
    else:
        suffix = ""
    return f"{val:.6g}{suffix}"

def validate_priors(priors, opt_params, fixed_params):
    """Check the validity of the priors dictionary supplied by the user.

    priors must be a dict of the form::

        {param_name: (mean, sigma) * u.Unit, ...}

    where mean and sigma are plain floats in the same canonical units as the rest of the code,
    or a length-2 astropy Quantity with the appropriate units.  
    
    Only parameters that are being optimised (i.e. keys present in opt_params) can have priors.

    Parameters
    ----------
    priors : dict or None
    opt_params : dict : optimisable parameters (after sanitisation)
    fixed_params : dict :fixed parameters (after sanitisation)

    Returns
    -------
    dict : validated priors, or {} if None.
    """
    if priors is None:
        return {}
    if not isinstance(priors, dict):
        raise TypeError(f"priors must be a dict, got {type(priors).__name__}.")

    validated = {}
    for key, val in priors.items():
        if key not in STREAMLINE_MODEL_PARAM_KEYS:
            raise KeyError(
                f"Unknown parameter '{key}' in priors. "
                f"Supported keys are: {list(STREAMLINE_MODEL_PARAM_KEYS)}"
            )
        if key in fixed_params:
            raise ValueError(
                f"Parameter '{key}' is fixed and cannot have a prior. "
                "Move it to initial_opt_params if you want to optimise it with a prior."
            )
        if key not in opt_params:
            raise ValueError(
                f"Parameter '{key}' has a prior but is not being optimised. "
                "Add it to initial_opt_params or remove it from priors."
            )
        if isinstance(val, u.Quantity) and val.shape == (2,):
            mean = val[0]
            sigma = val[1]
        elif isinstance(val, (tuple, list)) and len(val) == 2:
            mean, sigma = val
        else:
            raise ValueError(
                f"Prior for '{key}' must be a 2-element (mean, sigma) tuple, "
                f"or a length-2 Quantity e.g. (4.0, 1.0) * u.Msun. Got: {val!r}"
            )
        mean, sigma = val

        if isinstance(mean, u.Quantity):
            if key not in CANONICAL_UNITS:
                raise ValueError(f"No canonical units defined for '{key}'.")
            mean = float(mean.to(CANONICAL_UNITS[key]).value)
        else:
            mean = float(mean) # assume it's already in the right units
        if isinstance(sigma, u.Quantity):
            if key not in CANONICAL_UNITS:
                raise ValueError(f"No canonical units defined for '{key}'.")
            sigma = float(sigma.to(CANONICAL_UNITS[key]).value)
        else:
            sigma = float(sigma) # assume it's already in the right units
        if sigma <= 0:
            raise ValueError(
                f"Prior sigma for '{key}' must be positive, got {sigma}."
            )
        validated[key] = (mean, sigma)
    return validated


@jax.jit(static_argnames=('priors_keys',))
def compute_prior_penalty(model_params, priors_means, priors_sigmas, priors_keys):
    """Compute the Gaussian prior penalty term: 
        sum_k ((theta_k - mu_k) / sigma_k)^2.

    Parameters
    ----------
    model_params : dict: physical model parameters
    priors_means : tuple of float: prior means, one per prior key
    priors_sigmas : tuple of float: prior sigmas, one per prior key
    priors_keys : tuple of str: corresponding prior keys (parameter names)

    Returns
    -------
    scalar JAX array : the prior chi-squared contribution
    """
    penalty = jnp.asarray(0.0, dtype=jnp.float64)
    for key, mu_p, sigma_p in zip(priors_keys, priors_means, priors_sigmas):
        mu_p = jnp.asarray(mu_p, dtype=jnp.float64)
        sigma_p = jnp.asarray(sigma_p, dtype=jnp.float64)
        penalty = penalty + ((model_params[key] - mu_p) / sigma_p) ** 2
    return penalty

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

def find_active_bounds(norm_opt_params, norm_grads, normalisation_spec):
    """Find optimised parameters sitting on a clip bound with the gradient pushing them further past it.
    There the constrained minimum is on the bound rather than at a stationary point, so the gradient
    component for that parameter never reaches zero, and the quadratic (Hessian) approximation does not
    describe it. Cyclic parameters have no bounds.

    Returns dict {param key: 'lower' or 'upper'}"""
    active_bounds = {}
    for key, spec in normalisation_spec.items():
        if spec['cyclic'] or key not in norm_opt_params:
            continue
        value = float(norm_opt_params[key])
        grad = float(norm_grads[key])
        # gradient descent moves along -grad
        if value <= float(spec['clip_min']) and grad > 0:
            active_bounds[key] = 'lower'
        elif value >= float(spec['clip_max']) and grad < 0:
            active_bounds[key] = 'upper'
    return active_bounds


def projected_gradient_l2_norm(norm_grads, active_bounds):
    """L2 norm of the gradient, excluding parameters held at an active bound (see find_active_bounds),
    whose gradient components cannot go to zero."""
    return gradient_l2_norm({key: grad for key, grad in norm_grads.items() if key not in active_bounds})


@jax.jit(static_argnames=("npoints",))
def forward_model(model_params, distance_pc, npoints=1e6):
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
        npoints=npoints,
        spin=model_params.get('spin', 1.0),
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


@jax.jit(static_argnames=('loss_method',))
def _continuous_objective_at_u(
    u,
    ra_data,
    dec_data,
    v_data,
    ra_sigma,
    dec_sigma,
    v_sigma,
    r0,
    rmin,
    model_params,
    distance_pc,
    loss_method,
):
    r_inner = jnp.maximum(rmin, to_float64(0.5) * model_params['mu'] * r0)
    r = r_inner + u * (r0 - r_inner)
    ra_model, dec_model, v_model = stream_lines_grad.forward_model_at_radius(
        r,
        model_params,
        distance_pc,
    )

    if loss_method == 0:
        q = ((ra_data - ra_model) / ra_sigma) ** 2 + ((dec_data - dec_model) / dec_sigma) ** 2 + ((v_data - v_model) / v_sigma) ** 2
    else:
        r_proj_data, theta_proj_data = extract_streamline.cartesian_to_polar(ra_data, dec_data)
        r_proj_model, theta_proj_model = extract_streamline.cartesian_to_polar(ra_model, dec_model)
        dtheta = extract_streamline.wrap_to_pi(theta_proj_data - theta_proj_model)
        sigma_r = jnp.sqrt(ra_sigma ** 2 + dec_sigma ** 2)
        r_eps = to_float64(1e-8)
        r_safe = jnp.maximum(jnp.abs(r_proj_data), r_eps)
        sigma_theta = jnp.sqrt((dec_data * ra_sigma) ** 2 + (ra_data * dec_sigma) ** 2) / (r_safe ** 2)
        sigma_theta = jnp.maximum(sigma_theta, r_eps)
        q = ((r_proj_data - r_proj_model) / sigma_r) ** 2 + (dtheta / sigma_theta) ** 2 + ((v_data - v_model) / v_sigma) ** 2
    return q


@jax.jit(static_argnames=('loss_method', 'n_iterations'))
def golden_section_search_u(
    r0,
    rmin,
    data_ra,
    data_dec,
    data_v,
    sigma_ra,
    sigma_dec,
    sigma_v,
    model_params,
    distance_pc,
    loss_method,
    n_iterations=48,
):
    r_inner = jnp.maximum(rmin, to_float64(0.5) * model_params['mu'] * r0)
    lower = jnp.zeros_like(data_ra, dtype=jnp.float64)
    upper = jnp.ones_like(data_ra, dtype=jnp.float64)
    golden_ratio = (jnp.sqrt(to_float64(5.0)) - to_float64(1.0)) / to_float64(2.0)
    u1 = upper - golden_ratio * (upper - lower)
    u2 = lower + golden_ratio * (upper - lower)

    def body_fn(i, state):
        lower_i, upper_i, u1_i, u2_i = state
        q1 = _continuous_objective_at_u(u1_i, data_ra, data_dec, data_v, sigma_ra, sigma_dec, sigma_v, r0, rmin, model_params, distance_pc, loss_method)
        q2 = _continuous_objective_at_u(u2_i, data_ra, data_dec, data_v, sigma_ra, sigma_dec, sigma_v, r0, rmin, model_params, distance_pc, loss_method)
        lower_next = jnp.where(q1 <= q2, lower_i, u1_i)
        upper_next = jnp.where(q1 <= q2, u2_i, upper_i)
        u1_next = upper_next - golden_ratio * (upper_next - lower_next)
        u2_next = lower_next + golden_ratio * (upper_next - lower_next)
        return lower_next, upper_next, u1_next, u2_next

    lower_f, upper_f, _, _ = jax.lax.fori_loop(0, n_iterations, body_fn, (lower, upper, u1, u2))
    best_u = (lower_f + upper_f) / to_float64(2.0)
    best_u = jax.lax.stop_gradient(best_u)
    best_radius = r_inner + best_u * (r0 - r_inner)
    return best_u, best_radius


def match_continuous_model_to_data(prepared_data, model_params, distance_pc,
                                   loss_method=0, matching_iterations=MATCHING_ITERATIONS):
    """Match any prepared representative array directly to physical model radii."""
    model_params = dict(model_params)
    if 'mu' not in model_params:
        if 'rc' in model_params:
            model_params['mu'] = model_params['rc'] / model_params['r0']
        elif 'omega' in model_params:
            model_params['mu'] = stream_lines_grad.mu_from_omega(
                omega=model_params['omega'],
                mass=model_params['mass'],
                r0=model_params['r0'],
            )
        else:
            raise ValueError("model_params must contain 'mu', 'rc', or 'omega'.")
    valid = jnp.asarray(prepared_data.data_finite_mask, dtype=bool)
    best_u, best_radius = golden_section_search_u(
        r0=model_params['r0'],
        rmin=model_params.get('rmin', 0.0) or 0.0,
        data_ra=jnp.where(valid, prepared_data.ra_data, 0.0),
        data_dec=jnp.where(valid, prepared_data.dec_data, 0.0),
        data_v=jnp.where(valid, prepared_data.v_data, 0.0),
        sigma_ra=jnp.where(valid, prepared_data.ra_sigma, 1.0),
        sigma_dec=jnp.where(valid, prepared_data.dec_sigma, 1.0),
        sigma_v=jnp.where(valid, prepared_data.v_sigma, 1.0),
        model_params=model_params,
        distance_pc=distance_pc,
        loss_method=loss_method,
        n_iterations=matching_iterations,
    )
    ra_model, dec_model, v_model = stream_lines_grad.forward_model_at_radius(
        best_radius, model_params, distance_pc
    )
    return ContinuousMatchResult(
        best_u=best_u,
        best_radius=best_radius,
        ra_model_matched=ra_model,
        dec_model_matched=dec_model,
        v_model_matched=v_model,
        valid=valid,
        residual_ra=jnp.where(valid, (prepared_data.ra_data - ra_model) / prepared_data.ra_sigma, 0.0),
        residual_dec=jnp.where(valid, (prepared_data.dec_data - dec_model) / prepared_data.dec_sigma, 0.0),
        residual_v=jnp.where(valid, (prepared_data.v_data - v_model) / prepared_data.v_sigma, 0.0),
    )


@jax.jit
def match_continuous_model_to_point_cloud(
    ra_model_full,
    dec_model_full,
    v_model_full,
    valid_mask_model,
    ra_data,
    dec_data,
    v_data,
    intensity,
    ra_sigma,
    dec_sigma,
    v_sigma,
    model_params,
    distance_pc,
    loss_method=0,
    point_cloud_loss_scale=None,
    matching_iterations=MATCHING_ITERATIONS,
):
    ra_data = jnp.asarray(ra_data, dtype=jnp.float64)
    dec_data = jnp.asarray(dec_data, dtype=jnp.float64)
    v_data = jnp.asarray(v_data, dtype=jnp.float64)
    ra_sigma = jnp.asarray(ra_sigma, dtype=jnp.float64)
    dec_sigma = jnp.asarray(dec_sigma, dtype=jnp.float64)
    v_sigma = jnp.asarray(v_sigma, dtype=jnp.float64)
    intensity = jnp.asarray(intensity, dtype=jnp.float64)

    r0 = model_params['r0']
    rmin = model_params.get('rmin', to_float64(0.0))
    if rmin is None:
        rmin = to_float64(0.0)
    if point_cloud_loss_scale is None:
        point_cloud_loss_scale = jnp.asarray(ra_data.size, dtype=jnp.float64)
    else:
        point_cloud_loss_scale = jnp.asarray(point_cloud_loss_scale, dtype=jnp.float64)

    best_u, best_radius = golden_section_search_u(
        r0=r0,
        rmin=rmin,
        data_ra=ra_data,
        data_dec=dec_data,
        data_v=v_data,
        sigma_ra=ra_sigma,
        sigma_dec=dec_sigma,
        sigma_v=v_sigma,
        model_params=model_params,
        distance_pc=distance_pc,
        loss_method=loss_method,
        n_iterations=matching_iterations,
    )

    ra_model, dec_model, v_model = stream_lines_grad.forward_model_at_radius(
        best_radius,
        model_params,
        distance_pc,
    )
    if loss_method == 0:
        residual_ra = (ra_data - ra_model) / ra_sigma
        residual_dec = (dec_data - dec_model) / dec_sigma
        residual_v = (v_data - v_model) / v_sigma
        weighted_term = intensity / jnp.sum(intensity)
        weighted_loss = point_cloud_loss_scale * jnp.sum(weighted_term * (residual_ra ** 2 + residual_dec ** 2 + residual_v ** 2))
        return best_u, best_radius, ra_model, dec_model, v_model, weighted_loss, residual_ra, residual_dec, residual_v, {
            'point_cloud_loss_scale': point_cloud_loss_scale,
            'intensity_sum': jnp.sum(intensity),
            'intensity_min': jnp.min(intensity),
            'intensity_max': jnp.max(intensity),
        }

    r_proj_data, theta_proj_data = extract_streamline.cartesian_to_polar(ra_data, dec_data)
    r_proj_model, theta_proj_model = extract_streamline.cartesian_to_polar(ra_model, dec_model)
    dtheta = extract_streamline.wrap_to_pi(theta_proj_data - theta_proj_model)
    sigma_r = jnp.sqrt(ra_sigma ** 2 + dec_sigma ** 2)
    r_eps = to_float64(1e-8)
    r_safe = jnp.maximum(jnp.abs(r_proj_data), r_eps)
    sigma_theta = jnp.sqrt((dec_data * ra_sigma) ** 2 + (ra_data * dec_sigma) ** 2) / (r_safe ** 2)
    sigma_theta = jnp.maximum(sigma_theta, r_eps)
    residual_r = (r_proj_data - r_proj_model) / sigma_r
    residual_theta = dtheta / sigma_theta
    residual_v = (v_data - v_model) / v_sigma
    weighted_term = intensity / jnp.sum(intensity)
    weighted_loss = point_cloud_loss_scale * jnp.sum(weighted_term * (residual_r ** 2 + residual_theta ** 2 + residual_v ** 2))
    return best_u, best_radius, ra_model, dec_model, v_model, weighted_loss, residual_r, residual_theta, residual_v, {
        'point_cloud_loss_scale': point_cloud_loss_scale,
        'intensity_sum': jnp.sum(intensity),
        'intensity_min': jnp.min(intensity),
        'intensity_max': jnp.max(intensity),
    }


def integration_delta_nodes(n_nodes=INTEGRATION_NODES):
    """Fixed in-plane angle nodes for the integrated methods, from r0 (delta=0) to the disk midplane (delta=pi/2).
    The nodes never depend on the model parameters, so the model is smooth in every parameter at every node."""
    if n_nodes < 2:
        raise ValueError('integration_nodes must be >= 2.')
    return jnp.linspace(to_float64(0.0), to_float64(jnp.pi / 2), n_nodes)


def polar_sigmas(ra_data, dec_data, ra_sigma, dec_sigma):
    """Uncertainties on projected radius and polar angle of the data, as used by loss_method 1"""
    sigma_r = jnp.sqrt(ra_sigma ** 2 + dec_sigma ** 2)
    r_eps = to_float64(1e-8)
    r_proj_data, _ = extract_streamline.cartesian_to_polar(ra_data, dec_data)
    r_safe = jnp.maximum(jnp.abs(r_proj_data), r_eps)
    sigma_theta = jnp.sqrt((dec_data * ra_sigma) ** 2 + (ra_data * dec_sigma) ** 2) / (r_safe ** 2)
    sigma_theta = jnp.maximum(sigma_theta, r_eps)
    return sigma_r, sigma_theta


def _log1mexp(x):
    """log(1 - exp(x)) for x < 0, stable for x near 0 and for x very negative"""
    return jnp.where(x > -jnp.log(2.0), jnp.log(-jnp.expm1(x)), jnp.log1p(-jnp.exp(x)))


def log_diff_ndtr(a, b):
    """log(Phi(b) - Phi(a)) for b >= a, where Phi is the standard normal CDF. Stable in both tails.
    Uses Phi(b) - Phi(a) = Phi(-a) - Phi(-b) to keep both arguments on the lower tail."""
    flip = a > 0.0
    lower = jnp.where(flip, -b, a)
    upper = jnp.where(flip, -a, b)
    log_upper = jax.scipy.special.log_ndtr(upper)
    log_lower = jax.scipy.special.log_ndtr(lower)
    # cap the log-ratio just below zero, so zero-length segments give a finite, negligible contribution
    return log_upper + _log1mexp(jnp.minimum(log_lower - log_upper, to_float64(-1e-12)))


def _integrated_point_terms(point, node_coords, node_physical, delta_nodes, loss_method):
    """Integrated loss terms for a single data point. See integrated_match_terms."""
    coords = jnp.stack(point[:3])
    sigmas = jnp.stack(point[3:])

    # segment vectors and point-minus-segment-start vectors, in sigma units
    seg = node_coords[:, 1:] - node_coords[:, :-1]
    to_point = coords[:, None] - node_coords[:, :-1]
    if loss_method == 1:
        seg = seg.at[1].set(extract_streamline.wrap_to_pi(seg[1]))
        to_point = to_point.at[1].set(extract_streamline.wrap_to_pi(to_point[1]))
    seg = seg / sigmas[:, None]
    to_point = to_point / sigmas[:, None]

    seg_len2 = jnp.maximum(jnp.sum(seg ** 2, axis=0), to_float64(1e-30))
    seg_len = jnp.sqrt(seg_len2)
    # position of the closest point on the (infinite) line through each segment, as a fraction of the segment
    t_star = jnp.sum(to_point * seg, axis=0) / seg_len2
    perp2 = jnp.sum((to_point - t_star * seg) ** 2, axis=0)
    # log[(1/sqrt(2 pi)) * integral over the segment of exp(-q/2) ds]
    lower = -seg_len * t_star
    upper = seg_len * (1.0 - t_star)
    log_mass = log_diff_ndtr(lower, upper)
    log_seg = -0.5 * perp2 + log_mass
    loss = -2.0 * jax.scipy.special.logsumexp(log_seg)

    # diagnostics only (not part of the loss): each segment's share of the integral, and the expected
    # matched point. Along a segment the weight is a normal in t truncated to [0, 1], with mean
    # t_star + (pdf(lower) - pdf(upper)) / (seg_len * (cdf(upper) - cdf(lower)))
    resp = jax.nn.softmax(log_seg)
    log_pdf_norm = -0.5 * jnp.log(2.0 * jnp.pi)
    pdf_ratio = (jnp.exp(-0.5 * lower ** 2 + log_pdf_norm - log_mass)
                 - jnp.exp(-0.5 * upper ** 2 + log_pdf_norm - log_mass))
    t_mean = jnp.clip(t_star + pdf_ratio / seg_len, 0.0, 1.0)
    residual2 = (to_point - t_mean * seg) ** 2
    seg_physical = node_physical[:, 1:] - node_physical[:, :-1]
    matched_physical = jnp.sum(resp * (node_physical[:, :-1] + t_mean * seg_physical), axis=1)
    matched_delta = jnp.sum(resp * (delta_nodes[:-1] + t_mean * (delta_nodes[1:] - delta_nodes[:-1])))
    return {
        'loss': loss,
        'residual2': jnp.sum(resp * residual2, axis=1),
        'matched_ra': matched_physical[0],
        'matched_dec': matched_physical[1],
        'matched_v': matched_physical[2],
        'matched_delta': matched_delta,
        'segment_length': jnp.sum(resp * seg_len),
        'outer_end_weight': resp[0] * (t_star[0] < 0.0),
        'inner_end_weight': resp[-1] * (t_star[-1] > 1.0),
    }


@jax.jit(static_argnames=('loss_method', 'batch_size'))
def integrated_match_terms(
    ra_nodes, dec_nodes, v_nodes, delta_nodes,
    ra_data, dec_data, v_data, ra_sigma, dec_sigma, v_sigma,
    loss_method=0, batch_size=INTEGRATION_BATCH_SIZE,
):
    """
    Integrated (marginal) loss of each data point against the whole model streamline.

    Instead of matching each data point to one model point, the Gaussian likelihood of the point is
    integrated along the model curve, with position along the curve measured as arc length in
    sigma-normalised (RA, Dec, v) or (r, theta, v) space:
        loss_i = -2 log[ (1/sqrt(2 pi)) * integral exp(-q_i(s)/2) ds ]
    where q_i is the chi2 of data point i against the model point at arc length s. The model is evaluated
    analytically at fixed nodes, and the Gaussian is integrated exactly along the straight segment between each
    pair of neighbouring nodes. Where the curve is locally straight, loss_i is the perpendicular chi2 of the point.
    Nothing is sorted, searched for or matched, so the loss is smooth in the model parameters.
    See docs/integrated_matching.tex.

    Returns a dict of per-data-point arrays:
        loss             : integrated loss of each point
        residual2        : (3, N) segment-weighted squared residuals per coordinate (diagnostic only)
        matched_ra/dec/v : segment-weighted expected matched model point (diagnostic only)
        matched_delta    : segment-weighted expected in-plane angle of the match (diagnostic only)
        segment_length   : segment-weighted segment length in sigma units. Values > ~1 mean integration_nodes is too small
        outer_end_weight : share of the integral beyond r0
        inner_end_weight : share of the integral beyond the disk midplane end
    """
    node_physical = jnp.stack([ra_nodes, dec_nodes, v_nodes])
    if loss_method == 0:
        node_coords = node_physical
        data_coords = (ra_data, dec_data, v_data)
        data_sigmas = (ra_sigma, dec_sigma, v_sigma)
    else:
        r_nodes, theta_nodes = extract_streamline.cartesian_to_polar(ra_nodes, dec_nodes)
        node_coords = jnp.stack([r_nodes, theta_nodes, v_nodes])
        r_data, theta_data = extract_streamline.cartesian_to_polar(ra_data, dec_data)
        sigma_r, sigma_theta = polar_sigmas(ra_data, dec_data, ra_sigma, dec_sigma)
        data_coords = (r_data, theta_data, v_data)
        data_sigmas = (sigma_r, sigma_theta, v_sigma)

    # checkpoint so reverse-mode recomputes per-batch intermediates rather than storing all N x K of them
    point_fn = jax.checkpoint(
        lambda point: _integrated_point_terms(point, node_coords, node_physical, delta_nodes, loss_method)
    )
    terms = jax.lax.map(point_fn, (*data_coords, *data_sigmas), batch_size=batch_size)
    terms['residual2'] = terms['residual2'].T
    return terms


def outer_anchor_chi2(ra_start, dec_start, v_start, outer_point, outer_sigma, loss_method=0):
    """chi2 between the start of the model (r0) and the outermost data point, in the coordinates of loss_method.

    Neither the continuous nor the integrated loss penalises model length beyond the data, so on their own they
    leave r0 weakly constrained from above. This term adds the assumption that the observed outer end of the streamer is where
    the streamline starts. It is evaluated at a fixed node (delta = 0), so it is smooth in every parameter."""
    ra_out, dec_out, v_out = outer_point[0], outer_point[1], outer_point[2]
    ra_sigma, dec_sigma, v_sigma = outer_sigma[0], outer_sigma[1], outer_sigma[2]
    if loss_method == 0:
        residuals = jnp.stack([
            (ra_out - ra_start) / ra_sigma,
            (dec_out - dec_start) / dec_sigma,
            (v_out - v_start) / v_sigma,
        ])
    else:
        r_out, theta_out = extract_streamline.cartesian_to_polar(ra_out, dec_out)
        r_start, theta_start = extract_streamline.cartesian_to_polar(ra_start, dec_start)
        sigma_r, sigma_theta = polar_sigmas(ra_out, dec_out, ra_sigma, dec_sigma)
        residuals = jnp.stack([
            (r_out - r_start) / sigma_r,
            extract_streamline.wrap_to_pi(theta_out - theta_start) / sigma_theta,
            (v_out - v_start) / v_sigma,
        ])
    return jnp.sum(residuals ** 2)


def integrated_model_nodes(model_params, distance_pc, integration_nodes=INTEGRATION_NODES):
    """Evaluate the closed-form model at the fixed integration nodes"""
    delta_nodes = integration_delta_nodes(integration_nodes)
    ra_nodes, dec_nodes, v_nodes, r_nodes = stream_lines_grad.forward_model_at_delta(
        delta_nodes, model_params, distance_pc
    )
    return delta_nodes, ra_nodes, dec_nodes, v_nodes, r_nodes


def match_integrated_model_to_data(prepared_data, model_params, distance_pc, loss_method=0,
                                   integration_nodes=INTEGRATION_NODES):
    """Expected matched model point for each data point under the integrated methods. For plotting and
    diagnostics only: the integrated loss itself never picks a single matched point."""
    valid = jnp.asarray(prepared_data.data_finite_mask, dtype=bool)
    delta_nodes, ra_nodes, dec_nodes, v_nodes, _ = integrated_model_nodes(
        model_params, distance_pc, integration_nodes
    )
    terms = integrated_match_terms(
        ra_nodes, dec_nodes, v_nodes, delta_nodes,
        jnp.where(valid, prepared_data.ra_data, 0.0),
        jnp.where(valid, prepared_data.dec_data, 0.0),
        jnp.where(valid, prepared_data.v_data, 0.0),
        jnp.where(valid, prepared_data.ra_sigma, 1.0),
        jnp.where(valid, prepared_data.dec_sigma, 1.0),
        jnp.where(valid, prepared_data.v_sigma, 1.0),
        loss_method=loss_method,
    )
    _, _, _, matched_radius = stream_lines_grad.forward_model_at_delta(
        terms['matched_delta'], model_params, distance_pc
    )
    ra_model = terms['matched_ra']
    dec_model = terms['matched_dec']
    v_model = terms['matched_v']
    return ContinuousMatchResult(
        best_u=terms['matched_delta'] / (jnp.pi / 2),
        best_radius=matched_radius,
        ra_model_matched=ra_model,
        dec_model_matched=dec_model,
        v_model_matched=v_model,
        valid=valid,
        residual_ra=jnp.where(valid, (prepared_data.ra_data - ra_model) / prepared_data.ra_sigma, 0.0),
        residual_dec=jnp.where(valid, (prepared_data.dec_data - dec_model) / prepared_data.dec_sigma, 0.0),
        residual_v=jnp.where(valid, (prepared_data.v_data - v_model) / prepared_data.v_sigma, 0.0),
    )


def prepare_matching_data(streamer, matching_method, n_elements=None, point_sigma_ra=None, point_sigma_dec=None,
                          point_sigma_v=None, point_cloud_loss_scale=None, anchor_outer_point=True):
    """Precompute the data container used by a continuous or integrated matching method (None for legacy).

    With anchor_outer_point=True, the outermost metric bin of the point cloud and its uncertainties are stored as
    outer_point/outer_sigma, and chi2_loss then ties the start of the model (r0) to it."""
    matching_method = check_matching_method(matching_method)
    if matching_method == 'legacy':
        return None
    if n_elements is None:
        n_elements = len(streamer.ra_data)
    binned = extract_streamline.prepare_binned_continuous_data(streamer, n_elements=n_elements)
    if matching_method in POINT_CLOUD_MATCHING_METHODS:
        prepared = extract_streamline.prepare_point_cloud_data(
            streamer,
            point_sigma_ra=point_sigma_ra,
            point_sigma_dec=point_sigma_dec,
            point_sigma_v=point_sigma_v,
            point_cloud_loss_scale=point_cloud_loss_scale,
        )
    else:
        prepared = binned
    if anchor_outer_point:
        # bins are in outer-first order of the distance metric, so bin 0 is the outermost
        prepared = prepared._replace(
            outer_point=jnp.stack([binned.ra_data[0], binned.dec_data[0], binned.v_data[0]]),
            outer_sigma=jnp.stack([binned.ra_sigma[0], binned.dec_sigma[0], binned.v_sigma[0]]),
        )
    return prepared


def match_model_to_data(prepared_data, model_params, distance_pc, matching_method, loss_method=0,
                        matching_iterations=MATCHING_ITERATIONS, integration_nodes=INTEGRATION_NODES):
    """Matched model point for each data point, using the given continuous or integrated matching method"""
    if matching_method in INTEGRATED_MATCHING_METHODS:
        return match_integrated_model_to_data(
            prepared_data, model_params, distance_pc,
            loss_method=loss_method, integration_nodes=integration_nodes,
        )
    return match_continuous_model_to_data(
        prepared_data, model_params, distance_pc,
        loss_method=loss_method, matching_iterations=matching_iterations,
    )


def model_curve(model_params, distance_pc, matching_method, npoints=1e6, n_curve_points=1000):
    """Model curve for plotting, sampled in the same way as the matching method uses it:
    in radius (to r_low) for legacy/continuous, and in in-plane angle (to the disk midplane) for integrated.

    Returns (ra_model, dec_model, v_model, valid_mask)"""
    if matching_method in INTEGRATED_MATCHING_METHODS:
        _, ra_model, dec_model, v_model, _ = integrated_model_nodes(model_params, distance_pc, n_curve_points)
        return ra_model, dec_model, v_model, jnp.ones_like(ra_model, dtype=bool)
    ra_model, dec_model, v_model, valid_mask, _ = forward_model(model_params, distance_pc, npoints=npoints)
    return ra_model, dec_model, v_model, valid_mask.astype(bool)


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

@jax.jit(static_argnames=("loss_method", "matching_method", "matching_iterations", "integration_nodes", "npoints", "priors_keys", "priors_means", "priors_sigmas"))
def chi2_loss(
    model_params,
    distance_pc,
    prepared_data,
    loss_method=0,
    matching_method='continuous',
    matching_iterations=MATCHING_ITERATIONS,
    integration_nodes=INTEGRATION_NODES,
    npoints=1e6,
    priors_keys=(),
    priors_means=(),
    priors_sigmas=()
):
    """Generates model via forward model and calculates loss between data and model.
    Returns (chi2_total, loss_trace, err) where err is a checkify.
    
    chi2_loss = chi2_data + chi2_priors.
    
    chi2_priors is the sum of Gaussian prior penalty terms for any optimised parameters where priors were given.
    See compute_prior_penalty

    For the integrated matching methods, chi2_data is the intensity-weighted sum of each data point's integrated
    loss along the whole model curve (see integrated_match_terms), and the per-coordinate chi2 components are
    diagnostics that do not sum to chi2_total.

    For the continuous and integrated methods, if the prepared data has an outer_point, chi2_outer (the chi2
    between the start of the model and the outermost data point, see outer_anchor_chi2) is added to chi2_total."""
 
    loss_method = check_loss_method(loss_method)
    matching_method = check_matching_method(matching_method)
    distance_pc = to_float64(distance_pc)

    if 'mu' not in model_params:
        model_params = dict(model_params)
        if 'rc' in model_params:
            model_params['mu'] = model_params['rc'] / model_params['r0']
        elif 'omega' in model_params:
            model_params['mu'] = stream_lines_grad.mu_from_omega(
                omega=model_params['omega'],
                mass=model_params['mass'],
                r0=model_params['r0'],
            )
        else:
            raise ValueError("model_params must contain 'mu', 'rc', or 'omega'.")

    if matching_method == 'legacy':
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

        chi2_prior = compute_prior_penalty(model_params, priors_means, priors_sigmas, priors_keys)
        chi2_total = chi2_total + chi2_prior

        if loss_method == 0:
            chi2_components = {'chi2_ra': chi2_ra, 'chi2_dec': chi2_dec, 'chi2_v': chi2_v, 'chi2_prior': chi2_prior, 'overlap_width': overlap_max - overlap_min, 'chi2_total': chi2_total}
        else:
            chi2_components = {'chi2_r': chi2_r, 'chi2_theta': chi2_theta, 'chi2_v': chi2_v, 'chi2_prior': chi2_prior, 'overlap_width': overlap_max - overlap_min, 'chi2_total': chi2_total}

        matching_trace = {'model_points_total': model_points_total, 'model_nan_count': model_nan_count, 'model_valid_points': model_valid_points, 'model_retained_count': jnp.sum(model_keep), 'data_points_total': ra_data.size, 'data_valid_points': jnp.sum(data_finite_mask), 'data_retained_count': jnp.sum(data_keep), 'overlap_metric_min': overlap_min, 'overlap_metric_max': overlap_max, 'model_metric_span': model_metric_span, 'model_metric_min_gap': model_metric_min_gap, 'model_metric_near_tie_count': model_metric_near_tie_count, 'model_metric_duplicate_count': model_metric_duplicate_count, 'model_metric_non_monotonic_count': model_metric_non_monotonic_count}
        loss_trace = {'chi2_components': chi2_components, 'matching': matching_trace, 'loss_method': loss_method}
        return chi2_total, loss_trace, None

    if matching_method in INTEGRATED_MATCHING_METHODS:
        valid = prepared_data.data_finite_mask
        weights = jnp.where(valid, prepared_data.weights, 0.0)
        weights_sum = jnp.sum(weights)
        weights = jnp.where(weights_sum > 0.0, weights / weights_sum, 0.0)
        point_cloud_loss_scale = prepared_data.point_cloud_loss_scale

        delta_nodes, ra_nodes, dec_nodes, v_nodes, r_nodes = integrated_model_nodes(
            model_params, distance_pc, integration_nodes
        )
        terms = integrated_match_terms(
            ra_nodes, dec_nodes, v_nodes, delta_nodes,
            jnp.where(valid, prepared_data.ra_data, 0.0),
            jnp.where(valid, prepared_data.dec_data, 0.0),
            jnp.where(valid, prepared_data.v_data, 0.0),
            jnp.where(valid, prepared_data.ra_sigma, 1.0),
            jnp.where(valid, prepared_data.dec_sigma, 1.0),
            jnp.where(valid, prepared_data.v_sigma, 1.0),
            loss_method=loss_method,
        )
        chi2_data = point_cloud_loss_scale * jnp.sum(weights * terms['loss'])
        component_values = point_cloud_loss_scale * jnp.sum(weights * terms['residual2'], axis=1)
        chi2_prior = compute_prior_penalty(model_params, priors_means, priors_sigmas, priors_keys)
        # the start of the model (delta = 0, i.e. r0) is anchored to the outermost data point
        if prepared_data.outer_point is not None:
            chi2_outer = outer_anchor_chi2(
                ra_nodes[0], dec_nodes[0], v_nodes[0],
                prepared_data.outer_point, prepared_data.outer_sigma, loss_method,
            )
        else:
            chi2_outer = to_float64(0.0)
        chi2_total = chi2_data + chi2_outer + chi2_prior

        component_keys = LOSS_METHOD_COMPONENT_KEYS[loss_method][:3]
        chi2_components = {key: value for key, value in zip(component_keys, component_values)}
        chi2_components.update({'chi2_outer': chi2_outer, 'chi2_prior': chi2_prior, 'chi2_total': chi2_total})

        matched_delta = jnp.where(valid, terms['matched_delta'], jnp.nan)
        matching_trace = {
            'data_points_total': prepared_data.ra_data.size,
            'data_valid_points': jnp.sum(valid),
            'model_points_total': integration_nodes,
            'model_valid_points': integration_nodes,
            'matched_delta_min': jnp.nanmin(matched_delta),
            'matched_delta_max': jnp.nanmax(matched_delta),
            'model_radius_min': jnp.min(r_nodes),
            'model_radius_max': jnp.max(r_nodes),
            'point_cloud_loss_scale': point_cloud_loss_scale,
            'integration_nodes': integration_nodes,
            # weighted mean and max length of the segments carrying each point's integral, in sigma units.
            # Values above ~1 mean the integral is under-resolved and integration_nodes should be increased
            'segment_length_mean': jnp.sum(weights * terms['segment_length']),
            'segment_length_max': jnp.max(jnp.where(valid, terms['segment_length'], 0.0)),
            # intensity-weighted share of the data lying beyond each end of the model
            'outer_end_weight': jnp.sum(weights * terms['outer_end_weight']),
            'inner_end_weight': jnp.sum(weights * terms['inner_end_weight']),
        }
        loss_trace = {'chi2_components': chi2_components, 'matching': matching_trace, 'loss_method': loss_method}
        return chi2_total, loss_trace, None

    ra_data = prepared_data.ra_data
    dec_data = prepared_data.dec_data
    v_data = prepared_data.v_data
    int_data = prepared_data.intensity
    point_weights = prepared_data.weights
    ra_sigma = prepared_data.ra_sigma
    dec_sigma = prepared_data.dec_sigma
    v_sigma = prepared_data.v_sigma
    finite_mask = prepared_data.data_finite_mask
    point_cloud_loss_scale = prepared_data.point_cloud_loss_scale

    rmin = model_params.get('rmin', 0.0)
    if rmin is None:
        rmin = to_float64(0.0)
    r0 = model_params['r0']

    valid = finite_mask
    weights = jnp.where(valid, point_weights, 0.0)
    weights_sum = jnp.sum(weights)
    weights = jnp.where(weights_sum > 0.0, weights / weights_sum, 0.0)

    best_u, best_radius = golden_section_search_u(
        r0=r0,
        rmin=rmin,
        data_ra=jnp.where(valid, ra_data, 0.0),
        data_dec=jnp.where(valid, dec_data, 0.0),
        data_v=jnp.where(valid, v_data, 0.0),
        sigma_ra=jnp.where(valid, ra_sigma, 1.0),
        sigma_dec=jnp.where(valid, dec_sigma, 1.0),
        sigma_v=jnp.where(valid, v_sigma, 1.0),
        model_params=model_params,
        distance_pc=distance_pc,
        loss_method=loss_method,
        n_iterations=matching_iterations,
    )

    ra_model, dec_model, v_model = stream_lines_grad.forward_model_at_radius(best_radius, model_params, distance_pc)
    residual_ra = jnp.where(valid, (ra_data - ra_model) / ra_sigma, 0.0)
    residual_dec = jnp.where(valid, (dec_data - dec_model) / dec_sigma, 0.0)
    residual_v = jnp.where(valid, (v_data - v_model) / v_sigma, 0.0)

    if loss_method == 0:
        q_point = residual_ra ** 2 + residual_dec ** 2 + residual_v ** 2
        chi2_ra = point_cloud_loss_scale * jnp.sum(weights * residual_ra ** 2)
        chi2_dec = point_cloud_loss_scale * jnp.sum(weights * residual_dec ** 2)
        chi2_v = point_cloud_loss_scale * jnp.sum(weights * residual_v ** 2)
        chi2_total = chi2_ra + chi2_dec + chi2_v
    else:
        r_proj_data, theta_proj_data = extract_streamline.cartesian_to_polar(ra_data, dec_data)
        r_proj_model, theta_proj_model = extract_streamline.cartesian_to_polar(ra_model, dec_model)
        dtheta = extract_streamline.wrap_to_pi(theta_proj_data - theta_proj_model)
        sigma_r = jnp.sqrt(ra_sigma ** 2 + dec_sigma ** 2)
        r_eps = to_float64(1e-8)
        r_safe = jnp.maximum(jnp.abs(r_proj_data), r_eps)
        sigma_theta = jnp.sqrt((dec_data * ra_sigma) ** 2 + (ra_data * dec_sigma) ** 2) / (r_safe ** 2)
        sigma_theta = jnp.maximum(sigma_theta, r_eps)
        residual_r = jnp.where(valid, (r_proj_data - r_proj_model) / sigma_r, 0.0)
        residual_theta = jnp.where(valid, dtheta / sigma_theta, 0.0)
        chi2_r = point_cloud_loss_scale * jnp.sum(weights * residual_r ** 2)
        chi2_theta = point_cloud_loss_scale * jnp.sum(weights * residual_theta ** 2)
        chi2_v = point_cloud_loss_scale * jnp.sum(weights * residual_v ** 2)
        chi2_total = chi2_r + chi2_theta + chi2_v

    # the start of the model (r0) is anchored to the outermost data point. The start is evaluated with the closed
    # form at delta = 0, which is the same point as r = r0 but has no arccos to clip there
    if prepared_data.outer_point is not None:
        ra_start, dec_start, v_start, _ = stream_lines_grad.forward_model_at_delta(
            to_float64(0.0), model_params, distance_pc
        )
        chi2_outer = outer_anchor_chi2(
            ra_start, dec_start, v_start, prepared_data.outer_point, prepared_data.outer_sigma, loss_method,
        )
    else:
        chi2_outer = to_float64(0.0)
    chi2_prior = compute_prior_penalty(model_params, priors_means, priors_sigmas, priors_keys)
    chi2_total = chi2_total + chi2_outer + chi2_prior

    if loss_method == 0:
        chi2_components = {'chi2_ra': chi2_ra, 'chi2_dec': chi2_dec, 'chi2_v': chi2_v, 'chi2_outer': chi2_outer, 'chi2_prior': chi2_prior, 'chi2_total': chi2_total}
    else:
        chi2_components = {'chi2_r': chi2_r, 'chi2_theta': chi2_theta, 'chi2_v': chi2_v, 'chi2_outer': chi2_outer, 'chi2_prior': chi2_prior, 'chi2_total': chi2_total}

    matching_trace = {
        'data_points_total': ra_data.size,
        'data_valid_points': jnp.sum(valid),
        'matched_u_min': jnp.min(best_u),
        'matched_u_max': jnp.max(best_u),
        'matched_radius_min': jnp.min(best_radius),
        'matched_radius_max': jnp.max(best_radius),
        'point_cloud_loss_scale': point_cloud_loss_scale,
        'intensity_min': jnp.min(int_data),
        'intensity_max': jnp.max(int_data),
        'intensity_sum': jnp.sum(int_data),
        'search_iterations': matching_iterations,
        'data_inner_count': jnp.sum(best_u <= 0.05),
        'outer_boundary_count': jnp.sum(best_u >= 0.95),
    }
    loss_trace = {'chi2_components': chi2_components, 'matching': matching_trace, 'loss_method': loss_method}
    return chi2_total, loss_trace, None



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
    matching_method='continuous',
    matching_iterations=MATCHING_ITERATIONS,
    priors=None,
    integration_nodes=INTEGRATION_NODES,
    anchor_outer_point=True,
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
    matching_method : str
        One of MATCHING_METHOD_CHOICES. See fit_streamline.
    priors : dict or None
        Optional Gaussian priors on optimised parameters, in the form
        {param_name: (mean, sigma), ...}. Only optimised parameters can have priors.
        Mean and sigma must be in the same canonical units as the rest of the code
    integration_nodes : int
        Number of model nodes for the integrated matching methods.
    anchor_outer_point : bool
        Continuous and integrated methods: tie the start of the model (r0) to the outermost data point.
        See fit_streamline.
 
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
    matching_method = check_matching_method(matching_method)
 
    model_params, opt_params_clean, fixed_params_clean = prepare_model_params(initial_opt_params, fixed_params)
    validated_priors = validate_priors(priors, opt_params_clean, fixed_params_clean)
    priors_keys = tuple(validated_priors.keys())
    priors_means = tuple(v[0] for v in validated_priors.values())
    priors_sigmas = tuple(v[1] for v in validated_priors.values())
 
    ra_model, dec_model, v_model, valid_mask_model, err = forward_model(
        model_params, distance_pc
    )
    err.throw()
 
    if matching_method == 'legacy':
        ra_model_interp, dec_model_interp, v_model_interp, valid, _, _, _ = (
            checked_match_model_to_data_curve(
                ra_model, dec_model, v_model, valid_mask_model,
                jnp.asarray(data[0], dtype=jnp.float64),
                jnp.asarray(data[1], dtype=jnp.float64),
            )
        )
        prepared_data = extract_streamline.prepare_data(data, uncertainties, n_elements=n_elements)
    else:
        pseudo_streamer = types.SimpleNamespace(
            pc_coords=jnp.vstack((data[0], data[1], data[2], jnp.ones_like(data[0]))),
            ra_sigma=uncertainties[0], dec_sigma=uncertainties[1], v_sigma=uncertainties[2],
        )
        prepared_data = prepare_matching_data(
            pseudo_streamer, matching_method, n_elements=n_elements, anchor_outer_point=anchor_outer_point
        )
        matched = match_model_to_data(
            prepared_data, model_params, distance_pc, matching_method,
            loss_method=loss_method, matching_iterations=matching_iterations,
            integration_nodes=integration_nodes,
        )
        ra_model_interp = matched.ra_model_matched
        dec_model_interp = matched.dec_model_matched
        v_model_interp = matched.v_model_matched
        valid = matched.valid

    chi2_total, loss_trace, _ = chi2_loss(
        model_params, distance_pc, prepared_data, loss_method=loss_method,
        matching_method=matching_method,
        matching_iterations=matching_iterations,
        integration_nodes=integration_nodes,
        priors_keys=priors_keys, priors_means=priors_means, priors_sigmas=priors_sigmas
    )
    chi2_components = loss_trace['chi2_components']
 
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
                   learning_rate=0.005, param_bounds=None, n_epochs=1000,
                   beta1=0.9, beta2=0.999,
                   info_every=100, loss_threshold=None, loss_threshold_epochs=1,
                   gradient_tol=None, gradient_tol_epochs=1,
                   early_stopping_patience=50,
                   save_folder='sting_results',
                   loss_method=1, # 0: radecvel, 1: rthetavel
                   matching_method='continuous',
                   matching_iterations=MATCHING_ITERATIONS,
                   integration_nodes=INTEGRATION_NODES,
                   anchor_outer_point=True,
                   priors=None,
                   v_lsr=None,
                   show_plots=False,
                   point_sigma_ra=None,
                   point_sigma_dec=None,
                   point_sigma_v=None,
                   point_cloud_loss_scale=None,
                   yso_centre=None,
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
        Only r0 and mass need bounds (if optimised). All other parameters are normalised automatically
        (see AUTO_NORMALISATION): angles and mu use their physical ranges, v_r0 is kept >= 0
        with no upper bound, and v_lsr is unbounded.
        Bounds supplied for any other parameter are ignored.
        Provide param_bounds as a dictionary with values as (min, max) tuples for each parameter
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
        - 1: rthetavel: optimise projected radial distance, polar angle, and velocity residuals.
        Both options use the same model-data matching and overlap penalty.
    matching_method : str
        How the model is compared to the data. Options:
        - 'continuous': binned data, each bin matched to its closest model radius by golden-section search.
        - 'continuous_point_cloud': as 'continuous', using every point in the point cloud.
        - 'integrated': binned data, each bin's likelihood integrated along the whole model curve, so no
          single matched point is chosen and the loss is smooth in every parameter. The model runs from r0 to
          the disk midplane and rmin/deltar are not used. See integrated_match_terms.
        - 'integrated_point_cloud': as 'integrated', using every point in the point cloud.
        - 'legacy': binned data matched to a sampled model by the sky-plane distance metric.
    integration_nodes : int
        Number of model nodes for the integrated methods. Increase it if the fit warns that the
        integral is under-resolved.
    anchor_outer_point : bool
        Continuous and integrated methods (not legacy). If True, add chi2_outer: the chi2 between the start of the
        model (r0) and the outermost metric bin of the data. Neither loss penalises model length beyond the data,
        so without this r0 is weakly constrained from above. It assumes the observed outer end of the streamer is
        where the streamline starts; set False if the emission may be cut off by the field of view or sensitivity.
    priors : dict or None
        Optional Gaussian priors on optimised parameters, in the form
        {param_name: (mean, sigma), ...}. Only optimised parameters can have priors.
        Mean and sigma must be in the same canonical units as the rest of the code
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
        indicating convergence. Parameters held at a bound, with the gradient pushing
        past it, are left out of the norm since their gradient cannot go to zero.
    gradient_tol_epochs : int
        Number of consecutive epochs with ||grad|| < gradient_tol required to
        trigger normalised-space gradient norm-based early stopping. Must be >= 1.
    show_plots : bool
        Whether to show diagnostic plots during optimisation
    yso_centre : astropy SkyCoord or None
        Star position (the same one passed to extract_streamline.reduce_to_1D). If given, absolute
        RA/Dec (deg) are added to best_fit_trajectory.csv alongside the offsets from the star.
        
    Epoch 0: initial state before any updates, with initial_opt_params
    Epoch n (n>=1): state after applying parameter update n
    Tracking and checks are all performed at the end of each epoch. So e.g. loss n = loss after applying update n, using the updated parameters

    Returns:
    --------   
    FitResult namedtuple with fields:
    - best_opt_params : dict of best-fit optimised parameters (physical/log units)
    - loss_history : list of loss values at each epoch (float)
    - param_errors: dict of estimated 1-sigma uncertainties for each optimised parameter in the display parameterisation (or None if uncertainty estimation failed).
      Parameters in at_bound are left out.
    - at_bound: dict {param name: 'lower' or 'upper'} of parameters that finished on a bound with the loss still pushing past it.
      They are held fixed at the bound when estimating the other uncertainties, and have zero variance in the covariance.
    - covariance_result: CovarianceResult or None: full covariance information needed for sampling, or None if estimation failed. Fields:
        - covariance : 2D array of covariance matrix in physical/log units
        - opt_keys: list of parameter keys corresponding to covariance_matrix rows/columns
        - best_opt_params: dict of best-fit optimised parameters (physical/log units)
        - fixed_params: dict of fixed parameters (physical/log units)
        - param_errors: dict of 1-sigma parameter uncertainties (physical/log units)
        - transformed_cov: dict of Jacobian-transformed covariance when 'rc'/'omega' was substutied by 'mu', keys are 'keys', 'cov', 'errors'
    """
    # lazy imports to avoid circular imports
    from . import outputs
    from . import errors
    # Initialize parameters
    loss_method = check_loss_method(loss_method)
    matching_method = check_matching_method(matching_method)

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
    opt_params, fixed_params, rotation_key = with_mu_substituted(opt_params, fixed_params)

    # check priors are valid and match optimised parameters
    validated_priors = validate_priors(priors, opt_params, fixed_params)
    priors_keys = tuple(validated_priors.keys())
    priors_means = tuple(float(v[0]) for v in validated_priors.values())
    priors_sigmas = tuple(float(v[1]) for v in validated_priors.values())


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
    if matching_method == 'legacy':
        prepared_data = extract_streamline.prepare_data(data, uncertainties, n_elements=len(data[0]))
    else:
        prepared_data = prepare_matching_data(
            streamer,
            matching_method,
            n_elements=len(data[0]),
            point_sigma_ra=point_sigma_ra,
            point_sigma_dec=point_sigma_dec,
            point_sigma_v=point_sigma_v,
            point_cloud_loss_scale=point_cloud_loss_scale,
            anchor_outer_point=anchor_outer_point,
        )
    # npoints for forward model evaluation: fixed large number set by max r0 bound and deltar
    # this is necessary to ensure forward model has constant array lengths for jax/jit compatability
    if 'r0' in param_bounds:
        max_r0 = param_bounds['r0'][1]
        deltar = fixed_params['deltar'] if 'deltar' in fixed_params else 1.0
        npoints = int(jnp.ceil(max_r0 / deltar)) + 1
    else: 
        npoints = 1e10

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
            matching_method=matching_method,
            matching_iterations=matching_iterations,
            integration_nodes=integration_nodes,
            npoints=npoints,
            priors_keys=priors_keys,
            priors_means=priors_means,
            priors_sigmas=priors_sigmas
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
    best_active_bounds = {}
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
    fieldnames = ['epoch', 'loss'] + [log_header(k) for k in opt_param_keys]

    all_param_keys = set(opt_param_keys) | set(fixed_params.keys())
    if save_folder is not None and 'mu' in all_param_keys:
        # also log derived rc and omega when mu is present for convenience
        if log_header('rc') not in fieldnames:
            fieldnames.append(log_header('rc'))
        if log_header('omega') not in fieldnames:
            fieldnames.append(log_header('omega'))
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
    if validated_priors:
        print(f"Priors:")
        for key, (mu_p, sigma_p) in validated_priors.items():
            print(f"  {key}: mean={format_param(key, mu_p)}, sigma={format_param(key, sigma_p)}")
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
            row[log_header(key)] = float(opt_params[key])
        row = add_rc_omega_to_log(row, opt_params, fixed_params, all_param_keys)
        if 'rc' in row:
            row[log_header('rc')] = row.pop('rc')
        if 'omega' in row:
            row[log_header('omega')] = row.pop('omega')
        log_writer.writerow(row)
        log_file.flush()
    
    # Log epoch 0 trace if trace file is requested
    if trace_writer is not None:
        # Compute initial loss and trace
        (loss_value_trace, (loss_trace_raw, _)), norm_grads_trace = loss_and_grad_fn(opt_params_norm)
        loss_trace = trace_tree_to_python(loss_trace_raw)
        active_bounds = find_active_bounds(opt_params_norm, norm_grads_trace, normalisation_spec)
        grad_norm = float(projected_gradient_l2_norm(norm_grads_trace, active_bounds))
        
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
                spec = normalisation_spec[key]
                if spec['cyclic']:
                    # phi0, pa are cyclic; wrap to [0, 1) in normalised space
                    opt_params_norm[key] = jnp.mod(opt_params_norm[key], 1.0)
                else:
                    opt_params_norm[key] = jnp.clip(opt_params_norm[key], spec['clip_min'], spec['clip_max'])

            # Now materialize physical parameters from the (possibly clamped)
            # normalised parameters.
            opt_params = denormalise_opt_params(opt_params_norm, normalisation_spec)

            # Compute loss and gradient at the post-update state S(epoch) for logging 
            (loss_value, (loss_trace_raw, err)), norm_grads = loss_and_grad_fn(opt_params_norm)
            # parameters held at a bound can't reach zero gradient, so leave them out of the convergence check
            active_bounds = find_active_bounds(opt_params_norm, norm_grads, normalisation_spec)
            grad_norm = float(projected_gradient_l2_norm(norm_grads, active_bounds))

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
                    row[log_header(key)] = float(opt_params[key])

                row = add_rc_omega_to_log(row, opt_params, fixed_params, all_param_keys)

                if 'rc' in row:
                    row[log_header('rc')] = row.pop('rc')
                if 'omega' in row:
                    row[log_header('omega')] = row.pop('omega')
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
                best_active_bounds = active_bounds
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

    if matching_method in INTEGRATED_MATCHING_METHODS:
        _, (best_trace, _) = loss_from_normalised(best_opt_params_norm)
        segment_length_max = float(best_trace['matching']['segment_length_max'])
        if segment_length_max > INTEGRATION_RESOLUTION_WARN:
            print(
                f"WARNING: the integrated loss is under-resolved at the best fit: model segments near some data "
                f"points are {segment_length_max:.2g} sigma long (should be < {INTEGRATION_RESOLUTION_WARN:g}). "
                f"Increase integration_nodes (currently {integration_nodes})."
            )

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
            matching_method=matching_method,
            matching_iterations=matching_iterations,
            integration_nodes=integration_nodes,
            gradient_tol=gradient_tol,
            normalisation_spec=normalisation_spec,
            best_norm_opt_params=best_opt_params_norm,
            rotation_key=key_needs_transform,
            active_bounds=best_active_bounds,
            npoints=npoints,
            priors_keys=priors_keys,
            priors_means=priors_means,
            priors_sigmas=priors_sigmas
        )
    except Exception as e:
        print(f"\nWarning: parameter uncertainty estimation failed: ({e}).")
        traceback.print_exc()
        print("Continuing without error estimates")

    display_opt_params = dict(ordered_best_opt_params)
    display_fixed_params = dict(fixed_params)
    display_param_errors = dict(param_errors) if param_errors is not None else None
    display_at_bound = dict(best_active_bounds)

    if cov_transformed_dict is not None and key_needs_transform is not None and display_param_errors is not None:
        if key_needs_transform in cov_transformed_dict['keys']:
            all_params_for_transform = {**display_fixed_params, **display_opt_params}
            mu_best = float(ordered_best_opt_params['mu'])
            mass_val = float(all_params_for_transform['mass'])
            r0_val   = float(all_params_for_transform['r0'])
            display_opt_params[key_needs_transform] = rotation_param_from_mu(key_needs_transform, mu_best, mass_val, r0_val)
            display_opt_params.pop('mu', None)
            if key_needs_transform in cov_transformed_dict['errors']:
                display_param_errors[key_needs_transform] = cov_transformed_dict['errors'][key_needs_transform]
            display_param_errors.pop('mu', None)
            if 'mu' in display_at_bound:
                display_at_bound[key_needs_transform] = display_at_bound.pop('mu')

    print("\nFinal parameters at best-fit:")
    all_display_params = {**display_fixed_params, **display_opt_params}
    for key in all_display_params.keys():
        value = all_display_params[key]
        if key in display_at_bound:
            print(f"  {key}: {format_param(key, value)} (at {display_at_bound[key]} bound, no uncertainty)")
        elif display_param_errors is not None and key in display_param_errors:
            error = display_param_errors[key]
            print(f"  {key}: {format_param(key, value)} ± {format_param(key, error)}")
        else:
            print(f"  {key}: {format_param(key, value)}")
    if display_at_bound:
        print(
            f"Note: {list(display_at_bound)} finished on a bound, so held fixed for estimating the other uncertainties. "
        )

    if save_folder is not None:
        outputs.save_best_fit_params(
            display_opt_params,
            display_fixed_params,
            display_param_errors,
            save_folder=save_folder,
            at_bound=display_at_bound,
        )

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
            matching_method=matching_method,
            matching_iterations=matching_iterations,
            integration_nodes=integration_nodes,
            yso_centre=yso_centre,
        )

    # save results to CovarianceResult and FitResult namedtuples
    cov_result = None
    if cov_matrix is not None:
        cov_result = CovarianceResult(
            covariance=cov_matrix,
            opt_keys=list(ordered_best_opt_params.keys()),
            best_opt_params=ordered_best_opt_params,
            fixed_params=fixed_params,
            param_errors=param_errors,
            transformed_cov=cov_transformed_dict
        )
    
    return FitResult(
        best_opt_params=display_opt_params,
        loss_history=loss_history,
        param_errors=display_param_errors,
        covariance_result=cov_result,
        at_bound=display_at_bound,
    )
