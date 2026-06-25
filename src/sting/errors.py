# For computing the uncertainties, we must use some slightly different versions of the functions,
# because the original versions use jax.jit, which is great for optimization,
# but not compatble with taking second derivatives for the Hessian-based uncertainty estimation
# because they have a side effect of requiring computations such as argsort.

# So we put the slower but hessian-compatible versions of the relevant functions here, and use them in the uncertainty estimation.

import jax.numpy as jnp
import jax
from jax.experimental import checkify
import astropy.units as u
import math
jax.config.update("jax_enable_x64", True)
from typing import NamedTuple
from collections import namedtuple
from . import stream_lines_grad, extract_streamline, gradient_descent


## constants 
eps = 1e-8 # small value to avoid division by zero
G = 6.67430e-11 * (1e-3)**2 * (1.988416e30) / (1.4959787e11) # in au (km/s)^2 * Msol^-1
au_to_km = 1.4959787e8 #km
FLOAT_DTYPE = jnp.float64

# settings and constants
BIG = 1e30
LOSS_METHOD_CHOICES = [0, 1]
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
    "omega": 1/u.s,
    # mu = rc/r0 is dimensionless, so no units
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


def match_model_to_data_curve_hsafe(
    ra_model,
    dec_model,
    v_model,
    valid_mask_model,
    ra_data,
    dec_data,
    model_sort_idx,
    dmetric_model_frozen, #precomputed at best fit params and treated as constant
    data_valid,
):
    """Second-derivative-safe version of match_model_to_data_curve, with the argsort replaced by a precomputed static integer index array"""
    ra_model = jnp.asarray(ra_model, dtype=jnp.float64)
    dec_model = jnp.asarray(dec_model, dtype=jnp.float64)
    v_model = jnp.asarray(v_model, dtype=jnp.float64)
    ra_data = jnp.asarray(ra_data, dtype=jnp.float64)
    dec_data = jnp.asarray(dec_data, dtype=jnp.float64)

    dmetric_model = dmetric_model_frozen
    dmetric_data, _ = extract_streamline.get_distance_metric(ra_data, dec_data) # this is not precomputed because it only depends on the data
    model_valid = valid_mask_model.astype(bool)

    d_data_valid = jnp.where(data_valid, dmetric_data, jnp.inf)
    data_min = jnp.min(d_data_valid)
    model_keep = model_valid & (dmetric_model >= data_min) & (dmetric_model < BIG)
    w_model = model_keep.astype(jnp.float64)

    d_model = jnp.where(model_keep, dmetric_model, 0.0)
    d_data = jnp.where(data_valid, dmetric_data, 0.0)

    # apply precomputed sort index to model
    d_model_s = d_model[model_sort_idx]
    ra_s = ra_model[model_sort_idx]
    dec_s = dec_model[model_sort_idx]
    v_s = v_model[model_sort_idx]
    w_model_s = w_model[model_sort_idx]

    data_min_eff = jnp.min(jnp.where(data_valid, dmetric_data, jnp.inf))
    data_max_eff = jnp.max(jnp.where(data_valid, dmetric_data, -jnp.inf))
    model_min = jnp.min(jnp.where(model_keep, dmetric_model, jnp.inf))
    model_max = jnp.max(jnp.where(model_keep, dmetric_model, -jnp.inf))

    model_span = model_max - model_min
    data_span = data_max_eff - data_min_eff
    model_span_safe = jnp.where(model_span > 0.0, model_span, 1.0)
    data_span_safe = jnp.where(data_span > 0.0, data_span, 1.0)

    d_data_norm = (d_data - data_min_eff) / data_span_safe
    d_goal = model_min + d_data_norm * model_span_safe

    xp = jnp.where(w_model_s > 0, d_model_s, BIG)
    ra_interp = jnp.interp(d_goal, xp, ra_s)
    dec_interp = jnp.interp(d_goal, xp, dec_s)
    v_interp = jnp.interp(d_goal, xp, v_s)

    valid = data_valid
    return ra_interp, dec_interp, v_interp, valid

@jax.jit(static_argnames=("loss_method", "npoints", "priors_keys", "priors_means", "priors_sigmas"))
def chi2_loss_hsafe(model_params, distance_pc, prepared_data, loss_method, model_sort_idx, dmetric_model_frozen,npoints=10000,
                    priors_keys=(), priors_means=(), priors_sigmas=()):
    """Second-derivative-safe version of chi2_loss, using match_model_to_data_curve_hsafe"""
    distance_pc = jnp.asarray(distance_pc, dtype=jnp.float64)

    rmin = model_params['rmin']
    if rmin is None:
        rmin = jnp.asarray(0.0, dtype=jnp.float64)

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

    xyz_stream_checked = checkify.checkify(stream_lines_grad.xyz_stream)
    errs, ((x, y, z), (vx, vy, vz), valid_mask) = xyz_stream_checked(
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
    )

    ra_model = -x / distance_pc
    dec_model = z / distance_pc
    v_model = vy + model_params['v_lsr']
    ra_model = jnp.where(valid_mask, ra_model, jnp.nan)
    dec_model = jnp.where(valid_mask, dec_model, jnp.nan)
    v_model = jnp.where(valid_mask, v_model, jnp.nan)

    ra_data = prepared_data.ra_data
    dec_data = prepared_data.dec_data
    v_data = prepared_data.v_data
    ra_sigma = prepared_data.ra_sigma_safe
    dec_sigma = prepared_data.dec_sigma_safe
    v_sigma = prepared_data.v_sigma_safe
    data_valid = prepared_data.data_finite_mask


    ra_interp, dec_interp, v_interp, valid = match_model_to_data_curve_hsafe(
        ra_model, dec_model, v_model, valid_mask, 
        ra_data, dec_data, 
        model_sort_idx,
        dmetric_model_frozen,
        data_valid
    )

    valid_weights = valid.astype(jnp.float64)

    chi2_v = jnp.sum(valid_weights * ((v_data - v_interp) / v_sigma)**2)

    if loss_method == 0:
        chi2_ra = jnp.sum(valid_weights * ((ra_data - ra_interp) / ra_sigma)**2)
        chi2_dec = jnp.sum(valid_weights * ((dec_data - dec_interp) / dec_sigma)**2)
        chi2_total = chi2_ra + chi2_dec + chi2_v
    else:
        r_proj_data = prepared_data.r_proj_data
        theta_proj_data = prepared_data.theta_proj_data
        r_proj_model, theta_proj_model = extract_streamline.cartesian_to_polar(ra_interp, dec_interp)
        dtheta = extract_streamline.wrap_to_pi(theta_proj_data - theta_proj_model)
        sigma_r = jnp.sqrt(ra_sigma**2 + dec_sigma**2)
        r_eps = jnp.asarray(1e-8, dtype=jnp.float64)
        r_safe = jnp.maximum(jnp.abs(r_proj_data), r_eps)
        sigma_theta = jnp.sqrt((dec_data * ra_sigma)**2 + (ra_data * dec_sigma)**2) / r_safe**2
        sigma_theta = jnp.maximum(sigma_theta, r_eps)
        chi2_r = jnp.sum(valid_weights * ((r_proj_data - r_proj_model) / sigma_r)**2)
        chi2_theta = jnp.sum(valid_weights * (dtheta / sigma_theta)**2)
        chi2_total = chi2_r + chi2_theta + chi2_v

    # add prior penalty if any
    chi2_prior = gradient_descent.compute_prior_penalty(model_params, priors_means, priors_sigmas, priors_keys)
    chi2_total = chi2_total + chi2_prior

    return chi2_total

def compute_model_sort_idx(best_opt_params, fixed_params, distance_pc, prepared_data, npoints=10000):
    """Evaluate forward model once at best-fit parameters, and compute sort index that can be reused for Hessian-based uncertainty estimation"""
    model_params = {**best_opt_params, **fixed_params}
    rmin = model_params.get('rmin', None)
    if rmin is None:
        rmin = 0.0

    if 'mu' in model_params:
        mu = float(model_params['mu'])
    elif 'rc' in model_params:
        mu = float(model_params['rc']) / float(model_params['r0'])
    elif 'omega' in model_params:
        mu = stream_lines_grad.mu_from_omega(omega=model_params['omega'], mass=model_params['mass'], r0=model_params['r0'])
    else:
        raise ValueError("model_params must contain either 'rc', 'omega', or 'mu'")
    
    model_params = dict(model_params)
    model_params['mu'] = mu

    (x, y, z), (vx, vy, vz), valid_mask = stream_lines_grad.xyz_stream(
        mass=float(model_params['mass']),
        r0=float(model_params['r0']),
        theta0=float(model_params['theta0']),
        phi0=float(model_params['phi0']),
        mu=float(model_params['mu']),
        v_r0=float(model_params['v_r0']),
        inc=float(model_params['inc']),
        pa=float(model_params['pa']),
        rmin=float(rmin),
        deltar=float(model_params['deltar']),
        npoints=npoints,
    )

    ra_model = jnp.asarray(-x, dtype=jnp.float64) / float(distance_pc)
    dec_model = jnp.asarray(z, dtype=jnp.float64) / float(distance_pc)
    ra_model = jnp.where(valid_mask, ra_model, jnp.nan)
    dec_model = jnp.where(valid_mask, dec_model, jnp.nan)

    dmetric_model, _ = extract_streamline.get_distance_metric(ra_model, dec_model)

    data_valid = prepared_data.data_finite_mask
    dmetric_data = prepared_data.dmetric_data
    data_min = jnp.min(jnp.where(data_valid, dmetric_data, jnp.inf))

    model_keep = valid_mask.astype(bool) & (dmetric_model >= data_min) & (dmetric_model < BIG)
    w_model = model_keep.astype(jnp.float64)
    d_model = jnp.where(model_keep, dmetric_model, 0.0)

    # return integer array of indices that would sort the model points by distance metric, with invalid points at the end
    model_sort_key = d_model + (1.0 - w_model) * BIG    

    return jnp.argsort(model_sort_key), dmetric_model

def estimate_covariance_at_best_fit(
    best_opt_params,
    initial_opt_params,
    fixed_params,
    data,
    uncertainties,
    distance,
    param_bounds,
    loss_method=0,
    gradient_tol=1e-1,
    rotation_key=None,
    priors_keys=(),
    priors_means=(),
    priors_sigmas=()
):
    """ wrapper around estimate_parameter_errors for convenient using after fit_streamline has finished"""
    opt_keys = list(initial_opt_params.keys())
    best_for_cov = {k: float(best_opt_params[k]) for k in opt_keys}
    # prepare data-only quantities once
    prepared_data = extract_streamline.prepare_data(data, uncertainties, n_elements=len(data[0]))
    param_bounds = gradient_descent.convert_and_strip_bound_units(param_bounds)
    best_for_cov_mu, fixed_params_mu, param_bounds, inferred_rotation_key = gradient_descent.with_mu_substituted(best_for_cov, fixed_params, param_bounds)
    if rotation_key is None and inferred_rotation_key in ('rc', 'omega'):
        rotation_key = inferred_rotation_key
    param_bounds = gradient_descent.auto_fill_angle_bounds(best_for_cov_mu, param_bounds)
    normalisation_spec = gradient_descent.build_normalisation_spec(best_for_cov_mu, param_bounds)
    param_errors, cov, cov_transformed_dict = estimate_parameter_errors(
        best_for_cov_mu,
        fixed_params_mu,
        distance,
        prepared_data,
        loss_method=loss_method,
        gradient_tol=gradient_tol,
        normalisation_spec=normalisation_spec,
        rotation_key=rotation_key,
        priors_keys=priors_keys,
        priors_means=priors_means,
        priors_sigmas=priors_sigmas
    )
    mu_opt_keys = list(best_for_cov_mu.keys())
    return opt_keys, param_errors, cov, cov_transformed_dict, best_for_cov_mu, fixed_params_mu, mu_opt_keys


def estimate_parameter_errors(
    best_opt_params,
    fixed_params,
    distance_pc,
    prepared_data,
    loss_method=0,
    gradient_tol=1e-1,
    normalisation_spec=None,
    best_norm_opt_params=None,
    rotation_key=None,
    npoints=10000,
    priors_keys=(),
    priors_means=(),
    priors_sigmas=()
):
    """
    Estimate parameter uncertainties using Hessian of chi2 loss.
    Hessian is computed in normalised parameter space to keep eery parameter on O(1) scale.
    Then resulting covariance is transformed back to physical parameter space.

    Parameters
    ----------
    prepared_data : PreparedData
        Precomputed data-only quantities (created via extract_streamline.prepare_data).
    gradient_tol : float or None
        Tolerance on gradient norm in normalised space. If provided and
        normalised-space gradient norm > gradient_tol at best params, a
        warning is issued because the quadratic approximation may not be valid.
    normalisation_spec : dict
        Bounds-derived normalisation metadata for optimised parameters.
    best_norm_opt_params : dict or None
        Normalised parameters at best-fit state
        If not provided, will be computed from best_opt_params and normalisation_spec, but providing it can save a redundant computation
    rotation_key : str or None
        If provided, must be 'rc' or 'omega'. Used to transform covariance matrix from optimised 'mu' to rotation_key
    priors_keys, priors_means, priors_sigmas : tuples
        Parallel tuples of prior information for optimised parameters. If provided, a prior penalty is added to the chi2 loss before computing the Hessian. If not provided, no prior penalty is added.

    Returns
    -------
    dict
        1-sigma uncertainties for each optimisable parameter
    array
        covariance matrix
    dict or None
        If rotation_key was given and 'mu' was optimised: {'keys': new_keys, 'cov': new_cov, 'errors': error_dict} 
        where new_keys is same as original keys but with 'mu' replaced by rotation_key, 
        new_cov is the covariance matrix transformed into the original parameter space, 
        and error_dict is the dict of 1-sigma errors for each parameter in new_keys.
    """
    if gradient_tol is not None:
        gradient_tol = float(gradient_tol)
        if not math.isfinite(gradient_tol):
            raise ValueError('gradient_tol must be finite when provided.')
        if gradient_tol <= 0:
            raise ValueError('gradient_tol must be positive when provided.')
        
    if normalisation_spec is None:
        raise ValueError("normalisation_spec not provided")
    

    # evaluate hessian in raw_v_r0 space (v_r0 = softplus(raw_v_r0))
    has_v_r0 = 'v_r0' in best_opt_params
    params_for_hessian = dict(best_opt_params)
    if has_v_r0:
        v_r0_best = gradient_descent.to_float64(params_for_hessian['v_r0'])
        if not bool(v_r0_best > 0): #should never be triggered
            raise ValueError(f"best fit v_r0 must be positive, got v_r0={v_r0_best}")
        params_for_hessian['v_r0'] = gradient_descent.inv_softplus(v_r0_best)


    # convert dict -> vector
    params_vec, keys = params_dict_to_vector(params_for_hessian)
    loss_method = gradient_descent.check_loss_method(loss_method)

    # precompute model sort index at best fit params
    model_sort_idx, dmetric_model = compute_model_sort_idx(best_opt_params, fixed_params, distance_pc, prepared_data, npoints=npoints)

    if best_norm_opt_params is not None:
        norm_opt_params = best_norm_opt_params
    else:
        norm_opt_params = gradient_descent.normalise_opt_params(best_opt_params, normalisation_spec)

    norm_params_vec, norm_keys = params_dict_to_vector(norm_opt_params)
    missing_norm_keys = [key for key in keys if key not in normalisation_spec and key != 'v_r0']
    if missing_norm_keys:
        raise ValueError(
            f"normalisation_spec is missing optimised parameter keys required: {missing_norm_keys} "
        )

    def loss_vec_norm(theta_norm_vec):
        norm_params = vector_to_params_dict(theta_norm_vec, norm_keys)
        physical_params = gradient_descent.denormalise_opt_params(norm_params, normalisation_spec)
        model_params = {**physical_params, **fixed_params}
        chi2_total = chi2_loss_hsafe(
            model_params,
            distance_pc,
            prepared_data,
            loss_method=loss_method,
            model_sort_idx=model_sort_idx,
            dmetric_model_frozen=dmetric_model,
            npoints=npoints,
            priors_keys=priors_keys,
            priors_means=priors_means,
            priors_sigmas=priors_sigmas
        )
        return chi2_total
    
    def loss_vec(theta_vec):
        params = vector_to_params_dict(theta_vec, keys)
        if has_v_r0:
            params = dict(params)
            params['v_r0'] = gradient_descent.softplus(params['v_r0'])
        model_params = {**params, **fixed_params}
        chi2_total = chi2_loss_hsafe(
            model_params,
            distance_pc,
            prepared_data,
            loss_method=loss_method,
            model_sort_idx=model_sort_idx,
            dmetric_model_frozen=dmetric_model,
            npoints=npoints,
            priors_keys=priors_keys,
            priors_means=priors_means,
            priors_sigmas=priors_sigmas
        )
        return chi2_total
    

    
    # Check gradient magnitude at best-fit parameters in normalised space.
    if gradient_tol is not None:
        def norm_loss_vec(theta_norm_vec):
            norm_params = vector_to_params_dict(theta_norm_vec, norm_keys)
            physical_params = gradient_descent.denormalise_opt_params(norm_params, normalisation_spec)
            model_params = {**physical_params, **fixed_params}
            chi2_total, _, _ = gradient_descent.chi2_loss(
                model_params,
                distance_pc,
                prepared_data,
                loss_method=loss_method,
                priors_keys=priors_keys,
                priors_means=priors_means,
                priors_sigmas=priors_sigmas
            )
            return chi2_total        

        norm_grad_vec = jax.grad(norm_loss_vec)(norm_params_vec)
        
        norm_grad_norm = float(gradient_descent.gradient_l2_norm(norm_grad_vec))

        if norm_grad_norm > gradient_tol:
            print(
                "WARNING: normalised-space gradient norm at best fit = "
                f"{norm_grad_norm:.3e} exceeds tolerance {gradient_tol:.3e}"
            )
            print("Optimisation may not have reached a minimum yet.")
            print("Parameter uncertainties may be less reliable. Consider:")
            print("    - Increasing n_epochs")
            print("    - Reducing learning rate for finer convergence")
            print("    - Reducing loss_threshold, if used, to allow more optimisation steps")
            
    # compute Hessian in normalised space
    H_norm = jax.hessian(loss_vec_norm)(norm_params_vec)

    # invert to get covariance in normalised space
    cov_norm = jnp.linalg.inv(H_norm)

    # transform from normalised space to physical space
    def denormalise_vec(theta_norm_vec):
        norm_params = vector_to_params_dict(theta_norm_vec, norm_keys)
        physical_params = gradient_descent.denormalise_opt_params(norm_params, normalisation_spec)
        # denormalise_opt_params returns v_r0 already passed through softplus - convert back to raw space here so that J is correct for the transformation from raw_v_r0 to v_r0
        if has_v_r0:
            physical_params = dict(physical_params)
            physical_params['v_r0'] = gradient_descent.inv_softplus(physical_params['v_r0'])
        output = [physical_params[k] for k in keys]
        return jnp.stack(output)
    
    J = jax.jacobian(denormalise_vec)(norm_params_vec)
    cov = J @ cov_norm @ J.T
    
    if has_v_r0:
        v_r0_transformed = transform_cov_matrix(cov, keys, best_opt_params, fixed_params, rotation_key=None, v_r0_is_raw=True)
        cov = v_r0_transformed['cov']

    # parameter errors
    errors = jnp.sqrt(jnp.diag(cov))

    error_dict = {k: float(errors[i]) for i, k in enumerate(keys)}

    cov_transformed_dict = None
    if rotation_key is not None and 'mu' in keys:
        cov_transformed_dict = transform_cov_matrix(cov, keys, best_opt_params, fixed_params, rotation_key)

    return error_dict, cov, cov_transformed_dict

def transform_cov_matrix(cov, keys, best_opt_params, fixed_params, rotation_key=None, v_r0_is_raw=False):
    """Transform a covariance matrix from optimisation space to physical/output space.

    Maths:
    if A = original optimisation-space parameter vector
    and B = transformed parameter vector,
    then covariance in B space is

    cov_B = J @ cov_A @ J^T

    where J is the Jacobian of the transformation from A to B, evaluated at the best-fit parameters
    J = d(B) / d(A)  (identity except for rows being tranformed)

    Parameters:
    cov: covariance matrix from estimate_parameter_errors, in same parameter order as 'keys'
    keys: list of str. optimised parameter names
    best_opt_params: dict of best-fit optimised parameters in physical space (the point at which we evaluate J)
    fixed_params: dict of fixed parameters (the point at which we evaluate J)
    rotation_key: str of None, the original rotation parameter which we want to transform into ('rc' or 'omega')
    v_r0_is_raw: bool, whether v_r0 is in raw space (before softplus transformation)

    Returns:
    new_cov: covariance matrix transformed into original parameter space, with same order as keys but with 'mu' replaced by rotation_key if rotation_key is not None
    new_keys: list of str, same as keys but with 'mu' replaced by rotation_key if rotation_key is not None
    errors: dict of 1-sigma errors for each parameter in new_keys
    """
    if rotation_key is not None:
        if rotation_key not in ['rc', 'omega']:
            raise ValueError(f"rotation_key must be 'rc' or 'omega', got {rotation_key}")
        if 'mu' not in keys:
            raise ValueError(f"keys must include 'mu' for covariance transformation, got {keys}")
        # check that we have mass and r0 available to convert mu to rc or omega
        for required_key in ('mass', 'r0'):
            if required_key not in best_opt_params and required_key not in fixed_params:
                raise ValueError(f"'{required_key}' must be present in either best_opt_params or fixed_params")

    # build the vector at which to evaluate the jacobian
    params_list = []
    for k in keys:
        if k == 'v_r0' and v_r0_is_raw:
            v_r0_best = gradient_descent.to_float64(best_opt_params[k])
            if not bool(v_r0_best > 0): #should never be triggered
                raise ValueError(f"best fit v_r0 must be positive, got v_r0={v_r0_best}")
            params_list.append(gradient_descent.inv_softplus(v_r0_best))
        else:
            params_list.append(float(best_opt_params[k]))
    params_vec = jnp.array(params_list, dtype=jnp.float64)

    def transform(vec_A):
        opt_params = vector_to_params_dict(vec_A, keys)
        combined_params = {**fixed_params, **opt_params}
        if rotation_key is not None:
            mu = combined_params['mu']
            mass = combined_params['mass']
            r0 = combined_params['r0']
            if rotation_key == 'rc':
                rotation_val = mu * r0
            else: # 'omega'
                rotation_val = stream_lines_grad.omega_from_mu(mu=mu, mass=mass, r0=r0)
        output = []
        for k in keys:
            if k == 'mu' and rotation_key is not None:
                output.append(rotation_val)
            elif k == 'v_r0' and v_r0_is_raw:
                output.append(gradient_descent.softplus(opt_params[k]))
            else:
                output.append(opt_params[k])
        return jnp.stack(output)
    
    J = jax.jacobian(transform)(params_vec)
    new_cov = J @ cov @ J.T

    new_keys = [rotation_key if (k == 'mu' and rotation_key is not None) else k for k in keys]
    new_sigmas = jnp.sqrt(jnp.diag(new_cov))
    error_dict = {k: float(new_sigmas[i]) for i, k in enumerate(new_keys)}

    return {'keys': new_keys, 'cov': new_cov, 'errors': error_dict}

#-------------------- old gradient_descent.py ---------------------

def params_dict_to_vector(opt_params):
    """Convert parameter dict to ordered vector"""
    keys = list(opt_params.keys())
    vec = jnp.array([opt_params[k] for k in keys], dtype=jnp.float64)
    return vec, keys

def vector_to_params_dict(vec, keys):
    """Convert parameter vector back to dict"""
    return {k: vec[i] for i, k in enumerate(keys)}

@jax.jit
def match_model_to_data_curve(ra_model, dec_model, v_model, ra_data, dec_data):
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
    ra_model = extract_streamline.to_float64(ra_model)
    dec_model = extract_streamline.to_float64(dec_model)
    v_model = extract_streamline.to_float64(v_model)
    ra_data = extract_streamline.to_float64(ra_data)
    dec_data = extract_streamline.to_float64(dec_data)

    # get distance metrics
    dmetric_model, _ = extract_streamline.get_distance_metric(ra_model, dec_model)
    dmetric_data, _ = extract_streamline.get_distance_metric(ra_data, dec_data)

    # only finite values are valid
    model_valid = (
        jnp.isfinite(ra_model)
        & jnp.isfinite(dec_model)
        & jnp.isfinite(v_model)
        & jnp.isfinite(dmetric_model)
    )

    data_valid = (
        jnp.isfinite(ra_data)
        & jnp.isfinite(dec_data)
        & jnp.isfinite(dmetric_data)
    )

    # we also filter model to keep only model points with dmetric >= minimum of data dmetric
    # this is becuase the model shouldn't go further in than the innermost data point
    # as this is where we no longer observe the streamer
    d_data_valid = jnp.where(data_valid, dmetric_data, jnp.inf)
    data_min = jnp.min(d_data_valid)

    # enforce both constraints on model
    model_keep = model_valid & (dmetric_model >= data_min)

    # weights: 0 = ignore, 1 = use. This is for jax/jit compatibility
    w_model = model_keep.astype(jnp.float64)

    d_model = jnp.where(model_keep, dmetric_model, 0.0)
    d_data  = jnp.where(data_valid, dmetric_data, 0.0)

    ra = ra_model
    dec = dec_model
    v = v_model

    # ---- sort ONLY MODEL using metric + weight penalty ----
    model_sort_key = d_model + (1.0 - w_model) * BIG
    model_idx = jnp.argsort(model_sort_key)

    d_model_s = d_model[model_idx]
    ra_s = ra[model_idx]
    dec_s = dec[model_idx]
    v_s = v[model_idx]
    w_model_s = w_model[model_idx]

    # stats for trace and interpolation domain
    data_min_eff = jnp.min(jnp.where(data_valid, d_data, jnp.inf))
    data_max_eff = jnp.max(jnp.where(data_valid, d_data, -jnp.inf))

    model_min = jnp.min(jnp.where(model_keep, d_model, jnp.inf))
    model_max = jnp.max(jnp.where(model_keep, d_model, -jnp.inf))

    model_span = model_max - model_min
    data_span = data_max_eff - data_min_eff
    model_span_safe = jnp.where(model_span == 0.0, 1.0, model_span)
    data_span_safe = jnp.where(data_span == 0.0, 1.0, data_span)

    # normalise data metric
    d_data_norm = (d_data - data_min_eff) / data_span_safe
    d_goal = model_min + d_data_norm * model_span_safe

    # interpolate model at data points, using weights to ignore invalid model points 
    # by giving them huge distance values so they don't affect the interpolation
    xp = jnp.where(w_model_s > 0, d_model_s, BIG)

    ra_interp = jnp.interp(d_goal, xp, ra_s)
    dec_interp = jnp.interp(d_goal, xp, dec_s)
    v_interp = jnp.interp(d_goal, xp, v_s)

    # things for trace
    valid = data_valid

    matching_trace = {
    "model_points_total": model_idx.size,
    "model_nan_count": jnp.sum(jnp.isnan(d_model)),
    "model_valid_points": model_valid.sum(),
    "data_points_total": ra_data.size,
    "data_nan_count": jnp.sum(jnp.isnan(d_data)),
    "data_valid_points": data_valid.sum(),
    "model_metric_min": model_min,
    "model_metric_max": model_max,
    "data_metric_min": data_min_eff,
    "data_metric_max": data_max_eff,
    "model_metric_span": model_span_safe,
    "data_metric_span": data_span_safe}

    return ra_interp, dec_interp, v_interp, valid, dmetric_model, matching_trace

checked_matching = checkify.checkify(match_model_to_data_curve)

def checked_match_model_to_data_curve(*args, **kwargs):
    """Wrapper around match_model_to_data_curve with checkify checks for errors (to remain jax compatible)"""
    errors, result = checked_matching(*args, **kwargs)
    errors.throw()
    return result


    





