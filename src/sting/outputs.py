'''
This file contains functions related to outputs from streamfit optimisation,
such as saving logs and plotting results.

Last updated: 03-06-26
'''
import json
import math
from matplotlib.patches import Patch
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.family"] = "serif"
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import os


from . import gradient_descent
from . import extract_streamline

def param_for_display(key, value):
    """
    Format parameter for display in output, with units. Notably:
    - converts angles (theta0, phi0, inc, pa) from radians to degrees
    Returns (display_key, display_value, unit_str)
    """
    if key in gradient_descent.ANGLE_KEYS:
        return key, math.degrees(float(value)), 'deg'
    unit = gradient_descent.DISPLAY_UNITS.get(key, '')
    return key, float(value), unit


def evaluate_best_fit(
    best_opt_params,
    fixed_params,
    data,
    distance_pc,
    by_eye_params=None,
):
    """
    Run the forward model and match it to data for the best-fit parameters. Optionally run a by-eye parameter set through the forward model 
 
    Parameters
    ----------
    best_opt_params : dict
        Best-fit optimised parameters.
    fixed_params : dict
        Fixed model parameters.
    data : tuple of arrays (ra_data, dec_data, v_data)
    distance_pc : float
    by_eye_params : dict or None
        Optional by-eye parameters
    Returns
    -------
    dict with keys:
        ra_model, dec_model, v_model    : full forward model arrays
        ra_model_interp, dec_model_interp, v_model_interp : interpolated at data positions
        valid                           : boolean mask of retained data points
        by_eye                          : (ra, dec, v) tuple or None
    """
    ra_data, dec_data, _ = data
 
    best_opt_full_params, _, _ = gradient_descent.prepare_model_params(best_opt_params, fixed_params)
    ra_best, dec_best, v_best, valid_mask_best, _err = gradient_descent.forward_model(best_opt_full_params, distance_pc)
    valid_mask_best = valid_mask_best.astype(bool)
 
    ra_best_interp, dec_best_interp, v_best_interp, valid_interp, _, _, _ = (
        gradient_descent.checked_match_model_to_data_curve(
            ra_best, dec_best, v_best, valid_mask_best,
            jnp.asarray(ra_data, dtype=jnp.float64),
            jnp.asarray(dec_data, dtype=jnp.float64),
        )
    )
 
    by_eye = None
    if by_eye_params is not None:
        by_eye_full_params, _, _ = gradient_descent.prepare_model_params(by_eye_params, fixed_params)
        ra_by_eye, dec_by_eye, v_by_eye, _, err_by_eye = gradient_descent.forward_model(by_eye_full_params, distance_pc)
        err_by_eye.throw()
        by_eye = (ra_by_eye, dec_by_eye, v_by_eye)
 
    return dict(
        ra_model=ra_best,
        dec_model=dec_best,
        v_model=v_best,
        ra_model_interp=ra_best_interp,
        dec_model_interp=dec_best_interp,
        v_model_interp=v_best_interp,
        valid=valid_interp,
        by_eye=by_eye,
    )
 

def plot_fitting_results(
    ordered_best_opt_params,
    opt_param_keys,
    fixed_params,
    streamer,
    distance_pc,
    loss_history,
    param_errors,
    cov_matrix,
    v_lsr,
    save_folder,
    show_plots=False,
    transformed_cov_result=None,
    by_eye_params=None,
):
    """
    Generate and save the followingbest-fit diagnostic plots to save_folderafter optimisation:
    - loss_history.png          : loss vs epoch
    - best_fit_morphology.png   : RA/Dec best fit
    - best_fit_vel_radius.png   : velocity-radius best fit
    - parameter_uncertainties.png  : sizes of error bars for each optimised param (if param_errors given)
    - parameter_correlation_matrix.png : parameter correlation heatmap (if cov_matrix given)
 
    Parameters
    ----------
    ordered_best_opt_params : dict, best-fit optimised parameters
    opt_param_keys : list of str, list of optimised parameter names
    fixed_params : dict, fixed model parameters.
    streamer : NamedTuple with fields:
        pc_coords, ra_data, dec_data, v_data, ra_sigma, dec_sigma, v_sigma, data, uncertainties
    distance_pc : float
    loss_history : list of float, loss value at each epoch.
    param_errors : dict or None, 1-sigma parameter uncertainties keyed by parameter name, or None if uncertainty estimation failed
    cov_matrix : array or None, parameter covariance matrix, or None if uncertainty estimation failed.
    v_lsr : float or None, km/s
    save_folder : str, directory to write figures into (created if absent).
    show_plots : bool, whether to display plots (in addition to saving). Default False
    transformed_cov_result: dict or None. keys expected: 'keys', 'cov', 'errors'.
    by_eye_params: dict or None, optional by-eye parameter guess (with the same params as order_best_opt_params). If provided, will be plotted alongside the best-fit model in the morphology and velocity-radius plots.
    """
 
    ra_data, dec_data, v_data = streamer.data
    ra_sigma, dec_sigma, v_sigma = streamer.uncertainties
 
    # Loss
    plot_loss(loss_history, save_folder=save_folder, show=show_plots)
 
    # evaluate best fit morphology and belocity-radius
    best_fit = evaluate_best_fit(ordered_best_opt_params, fixed_params, streamer.data, distance_pc, by_eye_params=by_eye_params)

    plot_morphology(
        streamer=streamer,
        ra_model=best_fit['ra_model'],
        dec_model=best_fit['dec_model'],
        ra_model_interp=best_fit['ra_model_interp'],
        dec_model_interp=best_fit['dec_model_interp'],
        valid=best_fit['valid'],
        by_eye=best_fit['by_eye'],
        save_folder=save_folder,
        save_name='best_fit_morphology',
        show=show_plots,
    )
 
    plot_vel_radius(
        streamer=streamer,
        ra_model=best_fit['ra_model'],
        dec_model=best_fit['dec_model'],
        v_model=best_fit['v_model'],
        ra_model_interp=best_fit['ra_model_interp'],
        dec_model_interp=best_fit['dec_model_interp'],
        v_model_interp=best_fit['v_model_interp'],
        valid=best_fit['valid'],
        by_eye=best_fit['by_eye'],
        velocity_reference=v_lsr,
        save_folder=save_folder,
        save_name='best_fit_vel_radius',
        show=show_plots,
    )
 
    # Uncertainty plots (only if error estimation succeeeded)
    if param_errors is not None and cov_matrix is not None:
        if transformed_cov_result is not None:
            plot_keys = transformed_cov_result['keys']   # 'mu' replaced by 'rc'/'omega'
            plot_cov  = transformed_cov_result['cov']    # Jacobian-transformed covariance with 'mu' replaced by 'rc'/'omega'
            plot_errors = transformed_cov_result['errors'] # 'mu' error transformed to 'rc'/'omega' error
        else:
            plot_keys   = opt_param_keys
            plot_cov    = cov_matrix
            plot_errors = param_errors
        param_vals = np.array([float(ordered_best_opt_params.get(k, 0.0)) for k in opt_param_keys], dtype=float)
        param_errs = np.array([float(plot_errors[k]) for k in plot_keys], dtype=float)
        plot_param_uncertainties(plot_keys, param_vals, param_errs, save_folder=save_folder, show=show_plots)
        plot_param_correlations(plot_keys, plot_cov, save_folder=save_folder, show=show_plots)


def save_best_fit_params(best_opt_params, fixed_params, param_errors, save_folder='sting_results'):
    """
    saves parameters from the best-fit epoch (lowest loss) and their uncertainties 
    (when available, fixed params will not have uncertainties) to a JSON
    """
    output_path = os.path.join(save_folder, 'best_fit_params.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
 
    # parameters that were optimised
    optimised_section = {}
    for raw_key, raw_val in best_opt_params.items():
        display_key, display_val, unit = param_for_display(raw_key, raw_val)
        entry = {
            'value': display_val,
            'unit': unit,
        }
        if param_errors is not None and raw_key in param_errors:
            # same conversions for the errors as for the values
            # for log_omega: propagate via omega * sigma_log_omega).
            raw_err = float(param_errors[raw_key])
            if raw_key in gradient_descent.ANGLE_KEYS:
                display_err = math.degrees(raw_err)
            else:
                display_err = raw_err
            entry['sigma'] = display_err
        optimised_section[display_key] = entry
 
    # parameters that were fixed (no uncertainties)
    fixed_section = {}
    for raw_key, raw_val in fixed_params.items():
        if raw_val is None:
            fixed_section[raw_key] = {
                'value': None, 
                'unit': gradient_descent.DISPLAY_UNITS.get(raw_key, '')}
            continue
        display_key, display_val, unit = param_for_display(raw_key, raw_val)
        fixed_section[display_key] = {
            'value': display_val,
            'unit': unit,
        }
 
    output = {
        'optimised_parameters': optimised_section,
        'fixed_parameters': fixed_section,
    }
 
    with open(output_path, 'w') as file:
        json.dump(output, file, indent=4)


def _ensure_clean_dir(path):
    """Create directory if missing and remove any files inside it.

    Keeps behaviour consistent across plotting functions that write epoch frames.
    """
    os.makedirs(path, exist_ok=True)
    for filename in os.listdir(path):
        fp = os.path.join(path, filename)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except OSError:
                pass

def _opt_params_from_log(optimisation_log):
    skip = ['epoch', 'loss']
    cols = [c for c in optimisation_log.columns if c not in skip]
    if 'mu' in cols:
        cols = [c for c in cols if c not in ('rc', 'omega')]
    return cols


def create_video_from_images(save_folder, input_pattern, output_name, fps=5):
    """Call ffmpeg to make a video from numbered image frames.
    If you don't have ffmpeg, will print an error message instead of crashing
    """
    import subprocess
    import shutil

    ffmpeg_exe = shutil.which("ffmpeg") 
    if ffmpeg_exe is None:
        print(
            "ffmpeg not found, can't create video.\n"
            "To install: see https://ffmpeg.org/download.html and ensure ffmpeg is in your system PATH."
        )
        return

    output_video = os.path.join(save_folder, output_name)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-vf",
        "setpts='PTS/(1+0.01*N)',pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        output_video,
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video saved to {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")

def plot_loss(loss_history, save_folder='sting_results', show=False):
    '''Plot loss as a function of epochs'''
    # plot loss vs epoch nicely
    # matplotlib serif font
    plt.rcParams['font.family'] = 'serif'
    # Plot loss history
    # Epoch indexing: epoch 0 = initial state, epoch i (i >= 1) = after update i
    # loss_history is 0-indexed: loss_history[i] = loss at epoch i
    plt.figure(figsize=(12, 3))
    epochs = range(len(loss_history))
    plt.plot(epochs, loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimisation Progress')
    plt.yscale('log')
    plt.grid(True, alpha=0.5)
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(f'{save_folder}/loss_history.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def make_morphology_background(pc_coords, metric_boundaries, ra_lim, dec_lim, figsize=(6.5, 7)):
    """
    Pre-make the background image of the point cloud and metric boundaries for the morphology plots.
    This function is caleld by plot_morphology_by_epoch
 
    Returns
    -------
    bg_rgba : ndarray, shape (H, W, 4)
        RGBA image of the background at the target figure resolution.
    extent : list [left, right, bottom, top]
        Data-space extent to pass to ax.imshow so the image aligns correctly.
    """
    fig, ax = plt.subplots(figsize=figsize)
    pc_coords_np = np.asarray(pc_coords, dtype=float)
    ax.scatter(pc_coords_np[0], pc_coords_np[1], s=1, color='gray', alpha=0.3)
    if metric_boundaries is not None:
        extract_streamline.plot_metric_boundaries(ax, pc_coords_np, metric_boundaries,
                                                  color='gray', linewidth=1, alpha=0.3)
    ax.set_xlim(ra_lim)
    ax.set_ylim(dec_lim)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    buffer = fig.canvas.buffer_rgba()
    bg_rgba = np.asarray(buffer).copy()
    plt.close(fig)
    extent = [ra_lim[0], ra_lim[1], dec_lim[0], dec_lim[1]]
    return bg_rgba, extent

def plot_morphology_by_epoch(
    gradient_descent,
    fixed_params,
    distance,
    streamer=None,
    n_points=None,
    save_folder="sting_results",
    make_video=False
):
    """
    Create and save one streamline morphology plot per optimisation epoch, in save_folder/epochs/morphology,
    and optionally compile into a video
    """
    try:
        optimisation_log = load_optimisation_log(save_folder)
    except FileNotFoundError:
        print(f"Error: Could not find 'optimisation_log.csv' in {save_folder}")
        return
    param_names = _opt_params_from_log(optimisation_log)

    epochs = optimisation_log['epoch'].values

    # create the models
    epoch_models = []
    for idx, epoch in enumerate(epochs):

        row = optimisation_log.iloc[idx]
        opt_params_epoch = {param: float(row[param]) for param in param_names}
        opt_params_epoch_full, opt_params_epoch, fixed_params = gradient_descent.prepare_model_params(opt_params_epoch, fixed_params)
        ra_model, dec_model, v_model, valid_mask_model, err = gradient_descent.forward_model(opt_params_epoch_full, distance)
        valid_mask_model = valid_mask_model.astype(bool)

        (ra_model_interp, dec_model_interp, _, valid, model_keep, dmetric_model, matching_trace) = gradient_descent.checked_match_model_to_data_curve(
            ra_model,
            dec_model,
            v_model,
            valid_mask_model,
            streamer.ra_data,
            streamer.dec_data,
        )

        epoch_models.append(
            dict(
                epoch=epoch,
                opt_params_epoch=opt_params_epoch,
                ra_model=np.asarray(ra_model),
                dec_model=np.asarray(dec_model),
                ra_model_interp=np.asarray(ra_model_interp),
                dec_model_interp=np.asarray(dec_model_interp),
                valid=np.asarray(valid),
                model_keep=np.asarray(model_keep),
            )
        )

    # get constant axis limits
    all_ra = np.concatenate([
        *[e['ra_model'] for e in epoch_models],
        np.asarray(streamer.ra_data),
        np.asarray(streamer.pc_coords[0]),
    ])
    all_dec = np.concatenate([
        *[e['dec_model'] for e in epoch_models],
        np.asarray(streamer.dec_data),
        np.asarray(streamer.pc_coords[1]),
    ])
    mask = np.isfinite(all_ra) & np.isfinite(all_dec)
    all_ra = all_ra[mask]
    all_dec = all_dec[mask]

    pad_ra = 0.05 * (all_ra.max() - all_ra.min())
    pad_dec = 0.05 * (all_dec.max() - all_dec.min())

    ra_lim = (all_ra.min() - pad_ra, all_ra.max() + pad_ra)
    dec_lim = (all_dec.min() - pad_dec, all_dec.max() + pad_dec)

    # prepare clean output folder for epoch frames
    output_dir = os.path.join(save_folder, "epochs", "morphology")
    _ensure_clean_dir(output_dir)

    partitions = extract_streamline.get_metric_partitions(streamer.pc_coords, n_points)
    metric_boundaries, trace = extract_streamline.sample_metric_boundaries(streamer.pc_coords, partitions)

    # pre-make the background image (point cloud and metric boundaries)
    bg_rgba, bg_extent = make_morphology_background(streamer.pc_coords, metric_boundaries, ra_lim, dec_lim)

    # plot and save for each epoch
    for model in epoch_models:
        plot_morphology(
            ra_model=model["ra_model"],
            dec_model=model["dec_model"],
            streamer=streamer,
            ra_model_interp=model["ra_model_interp"],
            dec_model_interp=model["dec_model_interp"],
            valid=model["valid"],
            bg_rgba=bg_rgba,
            bg_extent=bg_extent,
            title=f"Epoch: {int(model['epoch'])}",
            xlim=ra_lim,
            ylim=dec_lim,
            save_folder=output_dir,
            save_name=f"morphology_epoch_{int(model['epoch']):03d}",
            show=False
        )

    if make_video:
        input_pattern = os.path.join(output_dir, "morphology_epoch_%03d.png")
        create_video_from_images(output_dir, input_pattern, "streamline_morphology_evolution.mp4", fps=5)

def plot_morphology(
    ra_model=None,
    dec_model=None,
    streamer=None,
    ra_model_interp=None,
    dec_model_interp=None,
    valid=None,
    by_eye=None,
    metric_boundaries=None,
    bg_rgba=None,
    bg_extent=None,
    title=None,
    xlim=None,
    ylim=None,
    legend_loc='lower right',
    save_folder='sting_results',
    save_name='streamline_morphology',
    show=True,
):
    '''Plot offsets in RA/Dec. Optionally include: model, model points, data points, best fit, background overlay, metric partitions.
    
    For a single plot, pass pc_corords and metric_boundaries directly. For per-epoch plotting use plot_morphology_by_epoch, 
    which will call this function and pass pre-rendered background images for speed.'''
    ra_data = None
    dec_data = None
    ra_sigma = None
    dec_sigma = None
    pc_coords = None
    if streamer is not None:
        ra_data = streamer.ra_data
        dec_data = streamer.dec_data
        ra_sigma = streamer.ra_sigma
        dec_sigma = streamer.dec_sigma
        pc_coords = streamer.pc_coords

    fig, ax = plt.subplots(figsize=(6.5, 7))
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)

    # Static background: prefer pre-rendered image, fall back to live drawing
    if bg_rgba is not None and bg_extent is not None:
        ax.imshow(
            bg_rgba,
            extent=bg_extent,
            aspect='auto',
            origin='upper',
            zorder=1,
        )
    elif pc_coords is not None:
        pc_coords_np = np.asarray(pc_coords, dtype=float)
        ax.scatter(pc_coords_np[0], pc_coords_np[1], s=1, color='gray',
                   alpha=0.3, label='Point cloud', zorder=4)
        if metric_boundaries is not None:
            ax_limits = ax.get_xlim(), ax.get_ylim()
            extract_streamline.plot_metric_boundaries(
                ax, pc_coords_np, metric_boundaries,
                color='gray', linewidth=1, alpha=0.3,
            )
            ax.set_xlim(ax_limits[0])
            ax.set_ylim(ax_limits[1])
 

    # model curve if given
    if ra_model is not None and dec_model is not None:
        ax.plot(ra_model, dec_model, color='blue', linewidth=2, label='STING', zorder=7)

    # model points if given
    if ra_model_interp is not None and dec_model_interp is not None and valid is not None:
        if valid is not None:
            ax.scatter(
                ra_model_interp[valid],
                dec_model_interp[valid],
                s=25,
                color='blue',
                zorder=7,
            )

    # by eye model if given
    if by_eye is not None:
        ra_by_eye, dec_by_eye, _ = by_eye
        ax.plot(
            np.asarray(ra_by_eye, dtype=float),
            np.asarray(dec_by_eye, dtype=float),
            color='tab:green',
            linewidth=2,
            label='By-eye',
            zorder=8,
        )

    # data points streamline if given
    if ra_data is not None and dec_data is not None:
        if ra_sigma is not None and dec_sigma is not None:
            ax.errorbar(
                ra_data,
                dec_data,
                xerr=ra_sigma,
                yerr=dec_sigma,
                fmt='o-',
                label='Extracted 1D Streamline',
                color='red',
                zorder=5,
            )
        else:
            ax.plot(
                ra_data,
                dec_data,
                'o-',
                label='Extracted 1D Streamline',
                color='red',
                zorder=5,
            )

    star_ra = 0
    star_dec = 0
    ax.scatter(star_ra, star_dec, marker='*', s=100, color='yellow', edgecolor='black', zorder=10)
    ax.set_xlabel('RA Offset (arcsec)')
    ax.set_ylabel('Dec Offset (arcsec)')

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        if pc_coords is not None:
            all_ra = np.concatenate([*[e for e in [ra_model, ra_data, pc_coords[0], np.array([star_ra])] if e is not None]])
        else:
            all_ra = np.concatenate([*[e for e in [ra_model, ra_data, np.array([star_ra])] if e is not None]])
        pad_ra = 0.05 * (all_ra.max() - all_ra.min())
        ra_lim = (all_ra.min() - pad_ra, all_ra.max() + pad_ra)
        ax.set_xlim(ra_lim)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        if pc_coords is not None:
            all_dec = np.concatenate([*[e for e in [dec_model, dec_data, pc_coords[1], np.array([star_dec])] if e is not None]])
        else:
            all_dec = np.concatenate([*[e for e in [dec_model, dec_data, np.array([star_dec])] if e is not None]])
        pad_dec = 0.05 * (all_dec.max() - all_dec.min())
        dec_lim = (all_dec.min() - pad_dec, all_dec.max() + pad_dec)
        ax.set_ylim(dec_lim)
    ax.invert_xaxis()
    ax.set_title(title)
    ax.legend(loc=legend_loc)
    if save_folder is not None:
        # make dir if it doesn't exist
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(f'{save_folder}/{save_name}.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_ra_vel_by_epoch(
    gradient_descent,
    fixed_params,
    distance,
    streamer=None,
    save_folder="sting_results",
    make_video=False,
):
    """
    Create RA–velocity plots for every epoch
    """
    try:
        optimisation_log = load_optimisation_log(save_folder)
    except FileNotFoundError:
        print(f"Error: Could not find 'optimisation_log.csv' in {save_folder}")
        return
    param_names = _opt_params_from_log(optimisation_log)

    epochs = optimisation_log['epoch'].values

    epoch_models = []

    # make models
    for idx, epoch in enumerate(epochs):

        row = optimisation_log.iloc[idx]
        opt_params_epoch = {p: float(row[p]) for p in param_names}
        opt_params_epoch_full, _, _ = gradient_descent.prepare_model_params(opt_params_epoch, fixed_params)
        ra_model, dec_model, v_model, valid_mask_model, err = gradient_descent.forward_model(opt_params_epoch_full, distance)
        valid_mask_model = valid_mask_model.astype(bool)
        ra_model_interp, _, v_model_interp, valid, model_keep, dmetric_model, matching_trace = (
            gradient_descent.checked_match_model_to_data_curve(ra_model, dec_model, v_model, valid_mask_model, streamer.ra_data, streamer.dec_data)
        )

        epoch_models.append({
            "epoch": epoch,
            "ra_model": ra_model,
            "v_model": v_model,
            "ra_model_interp": ra_model_interp,
            "v_model_interp": v_model_interp,
            "valid": valid,
            "model_keep": model_keep,
        })

    # global velocity limits
    v_list = [m["v_model"] for m in epoch_models]
    if streamer is not None:
        v_list.append(streamer.v_data)
    all_v = np.concatenate(v_list)
    vlim = (np.nanmin(all_v), np.nanmax(all_v))

    # global RA limits
    ra_list = [m["ra_model"] for m in epoch_models]
    if streamer is not None:
        ra_list.append(streamer.ra_data)
    all_ra = np.concatenate(ra_list)
    ralim = (np.nanmin(all_ra), np.nanmax(all_ra))

    # make clean output folder
    output_dir = os.path.join(save_folder, "epochs", "ra_vel")
    _ensure_clean_dir(output_dir)


    # make the plots
    for model in epoch_models:
        plot_ra_vel(
            ra_model=model["ra_model"],
            v_model=model["v_model"],
            streamer=streamer,
            ra_model_interp=model["ra_model_interp"],
            v_model_interp=model["v_model_interp"],
            valid=model["valid"],
            model_keep=model["model_keep"],
            title=f"Epoch: {int(model['epoch'])}",
            vlim=vlim,
            ralim=ralim,
            save_folder=output_dir,
            save_name=f"ra_vel_epoch_{int(model['epoch']):03d}",
        )

    if make_video:
        input_pattern = os.path.join(output_dir, "ra_vel_epoch_%03d.png")
        create_video_from_images(output_dir, input_pattern, "streamline_ra_vel_evolution.mp4", fps=5)
        

def plot_ra_vel(
    ra_model,
    v_model,
    *,
    streamer=None,
    ra_model_interp=None,
    v_model_interp=None,
    valid=None,
    model_keep=None,
    title=None,
    vlim=None,
    ralim=None,
    legend_loc='lower right',
    save_folder='sting_results',
    save_name='streamline_ra_vel',
    show=False,
):
    ra_data = None
    v_data = None
    ra_sigma = None
    v_sigma = None
    pc_coords = None
    if streamer is not None:
        ra_data = streamer.ra_data
        v_data = streamer.v_data
        ra_sigma = streamer.ra_sigma
        v_sigma = streamer.v_sigma
        pc_coords = streamer.pc_coords

    ra_model = np.asarray(ra_model, dtype=float)
    v_model = np.asarray(v_model, dtype=float)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
    if model_keep is not None:
        model_keep = np.asarray(model_keep, dtype=bool)
    fig, ax = plt.subplots(figsize=(6, 5))
    

    # point cloud (RA vs velocity)
    if pc_coords is not None:
        ax.scatter(pc_coords[0], pc_coords[2], s=1, alpha=0.3, color='grey', label='Point cloud')

    # data
    if ra_data is not None and v_data is not None:
        if ra_sigma is not None and v_sigma is not None:
            ax.errorbar(ra_data, v_data, xerr=ra_sigma, yerr=v_sigma, fmt='o', color='red', ecolor='red', ms=4, alpha=0.9, label='Data')
        else:
            ax.plot(ra_data, v_data, 'o', color='red', label='Data')

    # model curve
    if ra_model is not None and v_model is not None:
        ax.plot(ra_model, v_model, color='blue', linewidth=2, label='Model Streamline', zorder=7)

    # interpolated points
    if ra_model_interp is not None and v_model_interp is not None and valid is not None:
        ax.scatter(np.asarray(ra_model_interp)[valid], np.asarray(v_model_interp)[valid], s=25, color='blue', zorder=5, label='Model at data positions')

    ax.set_xlabel("RA Offset (arcsec)")
    ax.set_ylabel("Velocity (km/s)")
    ax.set_title(title or "RA vs Velocity")

    if vlim is not None:
        ax.set_ylim(vlim)
    if ralim is not None:
        ax.set_xlim(ralim)

    # flip RA axis to match astronomical convention
    ax.invert_xaxis()

    ax.legend(loc=legend_loc)
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f"{save_name}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


#########
def plot_dec_vel_by_epoch(
    gradient_descent,
    fixed_params,
    distance,
    streamer=None,
    save_folder="sting_results",
    make_video=False,
):
    """
    Create DEC–velocity plots for every epoch
    """
    try:
        optimisation_log = load_optimisation_log(save_folder)
    except FileNotFoundError:
        print(f"Error: Could not find 'optimisation_log.csv' in {save_folder}")
        return
    param_names = _opt_params_from_log(optimisation_log)

    epochs = optimisation_log['epoch'].values

    epoch_models = []

    # make models
    for idx, epoch in enumerate(epochs):

        row = optimisation_log.iloc[idx]
        opt_params_epoch = {p: float(row[p]) for p in param_names}
        opt_params_epoch_full, _, _ = gradient_descent.prepare_model_params(opt_params_epoch, fixed_params)
        ra_model, dec_model, v_model, valid_mask_model, err = gradient_descent.forward_model(opt_params_epoch_full, distance)
        valid_mask_model = valid_mask_model.astype(bool)
        ra_model_interp, dec_model_interp, v_model_interp, valid, model_keep, dmetric_model, matching_trace = (
            gradient_descent.checked_match_model_to_data_curve(ra_model, dec_model, v_model, valid_mask_model, streamer.ra_data, streamer.dec_data)
        )

        epoch_models.append({
            "epoch": epoch,
            "ra_model": ra_model,
            "dec_model": dec_model,
            "v_model": v_model,
            "ra_model_interp": ra_model_interp,
            "dec_model_interp": dec_model_interp,
            "v_model_interp": v_model_interp,
            "valid": valid,
            "model_keep": model_keep,
        })

    # global velocity limits
    v_list = [m["v_model"] for m in epoch_models]
    if streamer is not None:
        v_list.append(streamer.v_data)
    all_v = np.concatenate(v_list)
    vlim = (np.nanmin(all_v), np.nanmax(all_v))

    # global dec limits
    dec_list = [m["dec_model"] for m in epoch_models]
    if streamer is not None:
        dec_list.append(streamer.dec_data)
    all_dec = np.concatenate(dec_list)
    declim = (np.nanmin(all_dec), np.nanmax(all_dec))

    # make clean output folder
    output_dir = os.path.join(save_folder, "epochs", "dec_vel")
    _ensure_clean_dir(output_dir)


    # make the plots
    for model in epoch_models:
        plot_dec_vel(
            dec_model=model["dec_model"],
            v_model=model["v_model"],
            streamer=streamer,
            dec_model_interp=model["dec_model_interp"],
            v_model_interp=model["v_model_interp"],
            valid=model["valid"],
            model_keep=model["model_keep"],
            title=f"Epoch: {int(model['epoch'])}",
            vlim=vlim,
            declim=declim,
            save_folder = output_dir,
            save_name = f"dec_vel_epoch_{int(model['epoch']):03d}",
        )

    if make_video:
        input_pattern = os.path.join(output_dir, "dec_vel_epoch_%03d.png")
        create_video_from_images(output_dir, input_pattern, "streamline_dec_vel_evolution.mp4", fps=5)
        

def plot_dec_vel(
    dec_model,
    v_model,
    *,
    streamer=None,
    dec_model_interp=None,
    v_model_interp=None,
    valid=None,
    model_keep=None,
    title=None,
    vlim=None,
    declim=None,
    legend_loc='lower right',
    save_folder='sting_results',
    save_name='streamline_dec_vel',
    show=False,
):
    dec_data = None
    v_data = None
    dec_sigma = None
    v_sigma = None
    pc_coords = None
    if streamer is not None:
        dec_data = streamer.dec_data
        v_data = streamer.v_data
        dec_sigma = streamer.dec_sigma
        v_sigma = streamer.v_sigma
        pc_coords = streamer.pc_coords

    dec_model = np.asarray(dec_model, dtype=float)
    v_model = np.asarray(v_model, dtype=float)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
    if model_keep is not None:
        model_keep = np.asarray(model_keep, dtype=bool)

    fig, ax = plt.subplots(figsize=(6, 5))

    # point cloud (DEC vs velocity)
    if pc_coords is not None:
        # pc_coords layout: [ra, dec, velocity]
        ax.scatter(pc_coords[1], pc_coords[2], s=1, alpha=0.3, color='grey', label='Point cloud')

    # data
    if dec_data is not None and v_data is not None:
        if dec_sigma is not None and v_sigma is not None:
            ax.errorbar(dec_data, v_data, xerr=dec_sigma, yerr=v_sigma, fmt='o', color='red', ecolor='red', ms=4, alpha=0.9, label='Data')
        else:
            ax.plot(dec_data, v_data, 'o', color='red', label='Data')

    # model curve
    if dec_model is not None and v_model is not None:
        ax.plot(dec_model, v_model, color='blue', linewidth=2, label='Model Streamline')

    # interpolated points
    if dec_model_interp is not None and v_model_interp is not None and valid is not None:
        ax.scatter(np.asarray(dec_model_interp)[valid], np.asarray(v_model_interp)[valid], s=25, color='blue', zorder=5, label='Model at data positions')

    ax.set_xlabel("DEC Offset (arcsec)")
    ax.set_ylabel("Velocity (km/s)")
    ax.set_title(title or "DEC vs Velocity")

    if vlim is not None:
        ax.set_ylim(vlim)
    if declim is not None:
        ax.set_xlim(declim)

    ax.legend(loc=legend_loc)

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f"{save_name}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)


def build_velocity_radius_kde(
    ra_data,
    dec_data,
    vlos_data,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    grid_size=100,
    sigma_levels=None,
):
    """Build a KDE background grid for projected radius vs velocity plots.

    Parameters
    ----------
    rproj_data : array-like
        Projected radial distance samples (arcsec).
    vlos_data : array-like
        Line-of-sight velocity samples (km/s).
    xmin, xmax, ymin, ymax : float, optional
        Plot limits for KDE grid. If omitted, finite data limits are used.
    grid_size : int, optional
        Number of grid points per axis.
    sigma_levels : array-like, optional
        Sigma values used to build cumulative Gaussian-like contour levels.

    Returns
    -------
    dict
        Dictionary with keys: "xx", "yy", "zz", "levels", "xlim", "ylim".
    """
    from scipy import stats
    ra = np.asarray(ra_data)
    dec = np.asarray(dec_data)
    rproj = np.sqrt(ra**2 + dec**2)
    vlos = np.asarray(vlos_data, dtype=float)
    finite = np.isfinite(rproj) & np.isfinite(vlos)
    if np.sum(finite) < 3:
        raise ValueError("Need at least 3 finite samples to build KDE background.")

    rproj = rproj[finite]
    vlos = vlos[finite]

    if xmin is None:
        xmin = float(np.nanmin(rproj) - 1)
    if xmax is None:
        xmax = float(np.nanmax(rproj) + 1)
    if ymin is None:
        ymin = float(np.nanmin(vlos) - 1)
    if ymax is None:
        ymax = float(np.nanmax(vlos) + 1)


    xx, yy = np.mgrid[xmin:xmax:complex(grid_size), ymin:ymax:complex(grid_size)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([rproj, vlos])

    kernel = stats.gaussian_kde(values)
    zz = np.reshape(kernel(positions).T, xx.shape)
    zmax = np.nanmax(zz)
    if np.isfinite(zmax) and zmax > 0:
        zz = zz / zmax

    if sigma_levels is None:
        sigma_levels = np.arange(1.0, 2.1, 0.5)
    sigma_levels = np.asarray(sigma_levels, dtype=float)
    levels = np.append(np.exp(-0.5 * sigma_levels**2)[::-1], [1.0])
    kde_levels = np.append(np.exp(-0.5 * np.arange(1.0, 2.1, 0.5)**2)[::-1], [1.0])

    return {
        "xx": xx,
        "yy": yy,
        "zz": zz,
        "levels": levels,
        "xlim": (xmin, xmax),
        "ylim": (ymin, ymax),
    }


def plot_vel_radius(
    ra_model,
    dec_model,
    v_model,
    streamer,
    *,
    ra_model_interp=None,
    dec_model_interp=None,
    v_model_interp=None,
    valid=None,
    by_eye=None,
    model_keep=None,
    kde_background=None,
    velocity_reference=None,
    title=None,
    xlim=None,
    ylim=None,
    legend_loc='lower right',
    save_folder='sting_results',
    save_name=None,
    show=False,
):
    """Plot velocity vs projected radius for one model (optionally with KDE background)."""
    ra_data = streamer.ra_data
    dec_data = streamer.dec_data
    v_data = streamer.v_data
    ra_sigma = streamer.ra_sigma
    dec_sigma = streamer.dec_sigma
    v_sigma = streamer.v_sigma
    pc_coords = streamer.pc_coords

    ra_model = np.asarray(ra_model, dtype=float)
    dec_model = np.asarray(dec_model, dtype=float)
    v_model = np.asarray(v_model, dtype=float)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
    if model_keep is not None:
        model_keep = np.asarray(model_keep, dtype=bool)
        ra_model = ra_model[model_keep]
        dec_model = dec_model[model_keep]
        v_model = v_model[model_keep]

    rproj_model = np.sqrt(ra_model**2 + dec_model**2)
    order_model = np.argsort(rproj_model)

    fig, ax = plt.subplots(figsize=(6.5 * 1.3, 4 * 1.3))
    data_handle = None
    model_handle = None
    background_handle = None
    by_eye_handle = None

    if kde_background is None and pc_coords is not None:
        # make the kde background
        kde_background = build_velocity_radius_kde(
            ra_data=ra_data,
            dec_data=dec_data,
            vlos_data=v_data,
        )

    if kde_background is not None:
        ax.contourf(
            kde_background["xx"],
            kde_background["yy"],
            kde_background["zz"],
            levels=kde_background["levels"],
            cmap='Greys',
            vmin=0,
            vmax=1.2,
            zorder=1,
        )

        background_handle = Patch(
            facecolor='lightgray',
            edgecolor='none',
            label='Data KDE',
        )

    # Central source marker in this projection (r=0, v=v_lsr)
    if velocity_reference is not None:
        ax.scatter(
            0,
            float(velocity_reference),
            marker='*',
            s=100,
            color='yellow',
            edgecolor='black',
            zorder=10,
            label='Central Source',
        )

    if ra_data is not None and dec_data is not None and v_data is not None:
        ra_data = np.asarray(ra_data, dtype=float)
        dec_data = np.asarray(dec_data, dtype=float)
        v_data = np.asarray(v_data, dtype=float)
        rproj_data = np.sqrt(ra_data**2 + dec_data**2)

        if ra_sigma is not None and dec_sigma is not None and v_sigma is not None:
            ra_sigma = np.asarray(ra_sigma, dtype=float)
            dec_sigma = np.asarray(dec_sigma, dtype=float)
            v_sigma = np.asarray(v_sigma, dtype=float)
            denom = np.maximum(rproj_data, 1e-8)
            rproj_sigma = np.sqrt((ra_data * ra_sigma) ** 2 + (dec_data * dec_sigma) ** 2) / denom
            data_handle = ax.errorbar(
                rproj_data,
                v_data,
                xerr=rproj_sigma,
                yerr=v_sigma,
                fmt='o',
                color='red',
                ecolor='red',
                ms=4,
                alpha=0.9,
                label='Extracted 1D Streamline',
                zorder=6,
            )
        else:
            data_handle = ax.plot(
                rproj_data,
                v_data,
                'o',
                color='red',
                label='Extracted 1D Streamline',
                zorder=6,
            )[0]

    model_handle, = ax.plot(
        rproj_model[order_model],
        v_model[order_model],
        color='blue',
        linewidth=2,
        label='STING',
        zorder=7,
    )

    if (
        ra_model_interp is not None
        and dec_model_interp is not None
        and v_model_interp is not None
        and valid is not None
    ):
        ra_model_interp = np.asarray(ra_model_interp, dtype=float)
        dec_model_interp = np.asarray(dec_model_interp, dtype=float)
        v_model_interp = np.asarray(v_model_interp, dtype=float)
        valid = np.asarray(valid, dtype=bool)
        rproj_interp = np.sqrt(ra_model_interp**2 + dec_model_interp**2)
        ax.scatter(
            rproj_interp[valid],
            v_model_interp[valid],
            s=25,
            color='blue',
            label='Model at retained data arc lengths',
            zorder=8,
        )

    if velocity_reference is not None:
        ax.axhline(
            float(velocity_reference),
            color='black',
            linestyle='--',
            label='Systemic Velocity',
            zorder=4,
        )

    if by_eye is not None:
        ra_by_eye, dec_by_eye, v_by_eye = by_eye
        ra_by_eye = np.asarray(ra_by_eye, dtype=float)
        dec_by_eye = np.asarray(dec_by_eye, dtype=float)
        v_by_eye = np.asarray(v_by_eye, dtype=float)
        rproj_by_eye = np.sqrt(ra_by_eye**2 + dec_by_eye**2)
        by_eye_handle, = ax.plot(
            rproj_by_eye,
            v_by_eye,
            color='tab:green',
            linewidth=2,
            label='By-eye',
            zorder=9,
        )

    ax.set_xlabel('Projected Distance from Source (arcsec)')
    ax.set_ylabel('Velocity (km/s)')
    ax.set_title(title or 'Velocity vs Projected Radius')

    if xlim is not None:
        ax.set_xlim(xlim)
    elif kde_background is not None:
        ax.set_xlim(kde_background["xlim"])

    if ylim is not None:
        ax.set_ylim(ylim)
    elif kde_background is not None:
        ax.set_ylim(kde_background["ylim"])

    all_handles = [data_handle, model_handle, by_eye_handle, background_handle]
    handles = [h for h in all_handles if h is not None]
    if handles:
        ax.legend(handles=handles, loc=legend_loc)
    else:
        ax.legend(loc=legend_loc)

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(f'{save_folder}/{save_name}.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_vel_radius_by_epoch(
    gradient_descent,
    fixed_params,
    distance,
    streamer,
    *,
    grid_size=100,
    levels=None,
    velocity_reference=None,
    save_folder="sting_results",
    make_video=False,
):
    """Create velocity vs projected radius plots for every epoch."""

    try:
        optimisation_log = load_optimisation_log(save_folder)
    except FileNotFoundError:
        print(f"Error: Could not find 'optimisation_log.csv' in {save_folder}")
        return
    param_names = _opt_params_from_log(optimisation_log)
    
    epochs = optimisation_log['epoch'].values
    epoch_models = []

    kde_background = None
    if streamer is not None:
        kde_background = build_velocity_radius_kde(
            ra_data=streamer.ra_data,
            dec_data=streamer.dec_data,
            vlos_data=streamer.v_data,
            grid_size=grid_size,
            sigma_levels=levels,
        )

    for idx, epoch in enumerate(epochs):
        row = optimisation_log.iloc[idx]
        opt_params_epoch = {p: float(row[p]) for p in param_names}
        opt_params_epoch_full, _, _ = gradient_descent.prepare_model_params(opt_params_epoch, fixed_params)
        ra_model, dec_model, v_model, valid_mask_model, err = gradient_descent.forward_model(
            opt_params_epoch_full,
            distance,
        )

        valid_mask_model = valid_mask_model.astype(bool)

        ra_model_interp, dec_model_interp, v_model_interp, valid, model_keep, dmetric_model, matching_trace = (
            gradient_descent.checked_match_model_to_data_curve(
                ra_model,
                dec_model,
                v_model,
                valid_mask_model,
                streamer.ra_data,
                streamer.dec_data,
            )
        )

        if model_keep is not None:
            model_keep = model_keep.astype(bool)

        epoch_models.append({
            "epoch": epoch,
            "ra_model": ra_model,
            "dec_model": dec_model,
            "v_model": v_model,
            "ra_model_interp": ra_model_interp,
            "dec_model_interp": dec_model_interp,
            "v_model_interp": v_model_interp,
            "valid": valid,
            "model_keep": model_keep,
        })

    # Set consistent axis limits across epochs
    rproj_list = []
    v_list = [np.asarray(m["v_model"], dtype=float) for m in epoch_models]
    for model in epoch_models:
        ra_m = np.asarray(model["ra_model"], dtype=float)
        dec_m = np.asarray(model["dec_model"], dtype=float)
        rproj_list.append(np.sqrt(ra_m**2 + dec_m**2))

    if streamer is not None:
        rproj_list.append(np.sqrt(np.asarray(streamer.ra_data, dtype=float) ** 2 + np.asarray(streamer.dec_data, dtype=float) ** 2))
    if streamer is not None and streamer.v_data is not None:
        v_list.append(np.asarray(streamer.v_data, dtype=float))

    all_rproj = np.concatenate(rproj_list)
    all_v = np.concatenate(v_list)
    xlim = (np.nanmin(all_rproj), np.nanmax(all_rproj))
    ylim = (np.nanmin(all_v), np.nanmax(all_v))

    # make or clean output folder
    output_dir = os.path.join(save_folder, "epochs", "vel_radius")
    _ensure_clean_dir(output_dir)

    for model in epoch_models:
        plot_vel_radius(
            ra_model=model["ra_model"],
            dec_model=model["dec_model"],
            v_model=model["v_model"],
            streamer=streamer,
            ra_model_interp=model["ra_model_interp"],
            dec_model_interp=model["dec_model_interp"],
            v_model_interp=model["v_model_interp"],
            valid=model["valid"],
            model_keep=model["model_keep"],
            kde_background=kde_background,
            velocity_reference=velocity_reference,
            title=f"Epoch: {int(model['epoch'])}",
            xlim=xlim,
            ylim=ylim,
            save_folder=output_dir,
            save_name=f"vel_radius_epoch_{int(model['epoch']):03d}",
        )

    if make_video:
        input_pattern = os.path.join(output_dir, "vel_radius_epoch_%03d.png")
        create_video_from_images(
            output_dir,
            input_pattern,
            "streamline_vel_radius_evolution.mp4",
            fps=5,
        )


def plot_param_uncertainties(opt_keys, opt_params, opt_sigmas, save_folder=None, show=False):
    eps = 1e-12
    norm_errs = np.abs(opt_sigmas / (opt_params + eps))
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ypos = np.arange(len(opt_keys))
    ax.barh(
        ypos,
        norm_errs,
        color='tab:blue',
        alpha=0.8
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels(opt_keys)
    ax.set_xlabel('Relative uncertainty ($\\sigma / |x|$)')
    ax.set_title('Normalized Parameter Uncertainties')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(f'{save_folder}/parameter_uncertainties.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_param_correlations(param_names, covariance, annotate=True, save_folder=None, show=False):
    '''
    Plot a parameter correlation matrix derived from the covariance matrix, as a heatmpa.
    
    Parameters:
    ----------
    param_names: list of str
        Names of the parameters, in the same order as the covariance matrix.
    covariance: 2D array
        Covariance matrix of the parameters.
    annotate: bool, optional
        Whether to annotate the heatmap with correlation values.
    '''
    cov_np = np.array(covariance, dtype=float)

    diag = np.sqrt(np.clip(np.diag(cov_np), 1e-30, None))
    corr = cov_np / np.outer(diag, diag)
    corr = np.clip(corr, -1.0, 1.0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm_r')

    ax.set_xticks(np.arange(len(param_names)))
    ax.set_yticks(np.arange(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(param_names, fontsize=11)
    ax.set_title('Parameter Correlation Matrix')

    # Create colorbar axis with matched height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Correlation coefficient')

    if annotate:
        for i in range(len(param_names)):
            for j in range(len(param_names)):
                ax.text(
                    j, i,
                    f'{corr[i, j]:.2f}',
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='black'
                )

    plt.tight_layout()
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(f'{save_folder}/parameter_correlation_matrix.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_streamline_covariance_samples(best_opt_params,
                                       fixed_params,
                                       data,
                                       uncertainties,
                                       distance,
                                       covariance_result,
                                       v_lsr=None,
                                       n_samples=100,
                                       save_folder=None):
    """
    Sample parameter sets from the covariance matrix, evaluate streamlines from those sets, and plot them all together

        Parameters
    ----------
    best_opt_params : dict
        Best-fit optimised parameters in the original user-supplied parameterisation
        (e.g. with 'rc' or 'omega'). Used only to evaluate and plot the best-fit streamline.
    fixed_params : dict
        Fixed model parameters in the original user-supplied parameterisation.
        Used only to evaluate and plot the best-fit streamline.
    data : tuple of arrays (ra_data, dec_data, v_data)
    uncertainties : tuple of arrays (ra_sigma, dec_sigma, v_sigma)
    distance : float, distance in pc
    covariance_result : namedtuple from 
        result = fit_streamline(...)
        covariance_result = result.cov_result
    v_lsr : float or None, km/s
    n_samples : int
    save_folder : str or None
    """
    cov_best_params = covariance_result.best_opt_params
    cov_opt_keys = covariance_result.opt_keys
    cov_fixed_params = covariance_result.fixed_params
    cov = covariance_result.covariance
    _, streamline_samples = generate_streamline_samples(
        best_opt_params=cov_best_params,
        covariance=cov,
        opt_keys=cov_opt_keys,
        fixed_params=cov_fixed_params,
        distance=distance,
        param_bounds=None,
        n_samples=n_samples
    )

    fig, (ax_sky, ax_v) = plt.subplots(1, 2, figsize=(10, 5))
    ra_data, dec_data, v_data = data
    ra_sigma, dec_sigma, v_sigma = uncertainties

    # plot samples streamlines
    for streamline in streamline_samples:
        ra = streamline['ra']
        dec = streamline['dec']
        vel = streamline['v']

        rproj = np.sqrt(ra**2 + dec**2)
        order = np.argsort(rproj)
        ax_sky.plot(ra, dec, color='tab:blue', alpha=0.1, lw=1)
        ax_v.plot(rproj[order], vel[order], color='tab:blue', alpha=0.1, lw=1)

    # plot best fit streamline
    best_opt_full_params, best_opt_params, fixed_params = gradient_descent.prepare_model_params(best_opt_params, fixed_params)
    ra_best, dec_best, v_best, valid_mask_best, err = gradient_descent.forward_model(best_opt_full_params, distance)
    ra_best = np.asarray(ra_best, dtype=float)
    dec_best = np.asarray(dec_best, dtype=float)
    v_best = np.asarray(v_best, dtype=float)
    valid_mask_best = valid_mask_best.astype(bool)
    ra_best = ra_best[valid_mask_best]
    dec_best = dec_best[valid_mask_best]
    v_best = v_best[valid_mask_best]
    rproj_best = np.sqrt(ra_best**2 + dec_best**2)
    order_best = np.argsort(rproj_best)
    ax_sky.plot(ra_best, dec_best, color='blue', lw=2, label='Best-fit')
    ax_v.plot(rproj_best[order_best], v_best[order_best], color='blue', lw=2, label='Best-fit')

    # plot data
    ax_sky.errorbar(
        ra_data, dec_data, xerr=ra_sigma, yerr=dec_sigma,
        fmt='o', color='red', ecolor='red', ms=4, alpha=0.9, label='Data'
        )
    rproj_data = np.sqrt(ra_data**2 + dec_data**2)
    # get errors in rproj_data
    rproj_sigma = np.sqrt((ra_data * ra_sigma)**2 + (dec_data * dec_sigma)**2) / rproj_data
    order_data = np.argsort(rproj_data)
    ax_v.errorbar(
        rproj_data[order_data], np.asarray(v_data)[order_data], yerr=np.asarray(v_sigma)[order_data], xerr=np.asarray(rproj_sigma)[order_data],
        fmt='o', color='red', ecolor='red', ms=4, alpha=0.9, label='Data'
        )
    if v_lsr is not None:
        xmin, xmax = ax_v.get_xlim()
        ax_v.hlines(v_lsr, xmin=xmin, xmax=xmax, colors='k', linestyles='--', alpha=0.6,)
        ax_v.set_xlim(xmin, xmax)

    # finalise plots
    ax_sky.invert_xaxis()
    ax_sky.set_xlabel('RA Offset (arcsec)')
    ax_sky.set_ylabel('Dec Offset (arcsec)')
    ax_sky.set_title('Covariance Sampling')
    ax_sky.legend()

    ax_v.set_xlabel('Projected distance(arcsec)')
    ax_v.set_ylabel('Velocity (km/s)')
    ax_v.set_title('Covariance Sampling')
    ax_v.legend()

    plt.tight_layout()

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(f'{save_folder}/streamline_covariance_samples.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.show()


def generate_streamline_samples(best_opt_params, covariance, opt_keys, fixed_params, distance, param_bounds=None, n_samples=100):
    """
    wrapper of sample_parameter_sets_from_covariance() and evaluate_streamlines_samples() to generate streamline samples from covariance matrix.
    """
    samples = sample_parameter_sets_from_covariance(
        best_opt_params,
        covariance,
        opt_keys,
        param_bounds=param_bounds,
        n_samples=n_samples
    )
    streamlines = evaluate_streamlines_samples(
        samples,
        opt_keys,
        fixed_params,
        distance,
    )
    return samples, streamlines

def evaluate_streamlines_samples(param_samples, opt_keys, fixed_params, distance):
    """
    Evaluate streamline models for sampled parameter vectors
    Returns
    -------
    streamlines : list of dict
        Each entry contains:
        {
            "ra": ...,
            "dec": ...,
            "v": ...,
            "dmetric": ...
        }
    """
    streamlines = []
    for sample in param_samples:
        sample_params = {
            key: float(value)
            for key, value in zip(opt_keys, sample)
        }
        sample_params_full, _, _ = gradient_descent.prepare_model_params(sample_params, fixed_params)
        ra, dec, vel, valid_mask, err = gradient_descent.forward_model(sample_params_full, distance)
        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)
        vel = np.asarray(vel, dtype=float)
        valid_mask = valid_mask.astype(bool)
        ra = ra[valid_mask]
        dec = dec[valid_mask]
        vel = vel[valid_mask]

        dmetric, trace = extract_streamline.get_distance_metric(ra, dec)
        dmetric = np.asarray(dmetric, dtype=float)

        streamlines.append(
            {
                "ra": ra,
                "dec": dec,
                "v": vel,
                "dmetric": dmetric,
            }
        )

    return streamlines

def sample_parameter_sets_from_covariance(best_params, covariance, opt_keys, param_bounds=None, n_samples=100, seed=42):
    """
    Draw parameter samples from a covariance matrix.
    Returns
    -------
    samples : ndarray, shape (n_samples, n_params)
        Sampled parameter vectors
    """
    rng = np.random.default_rng(seed)
    mu = np.array(
        [best_params[key] for key in opt_keys],
        dtype=float,
    )
    cov = np.asarray(covariance, dtype=float)
    samples = rng.multivariate_normal(
        mu,
        cov,
        size=n_samples,
    )
    param_bounds = gradient_descent.auto_fill_angle_bounds(set(opt_keys), param_bounds)
    param_bounds = gradient_descent.convert_and_strip_bound_units(param_bounds)
    for j, key in enumerate(opt_keys):
        if key in param_bounds:
            low, high = param_bounds[key]
            samples[:, j] = np.clip(samples[:, j], low, high)

    return samples

def plot_param_optimisation_history(save_folder='sting_results'):
    '''Plot the history of parameter optimisation from logs saved during optimisation.
    save_folder should be the same as the one used during optimisation, 
    and should contain "optimisation_log.csv" and optionally "optimisation_trace.csv"'''
    
    try:
        optimisation_log = load_optimisation_log(save_folder)
    except FileNotFoundError:
        print(f"Error: Could not find 'optimisation_log.csv' and/or 'optimisation_trace.csv' in {save_folder}")
        return

    epochs = optimisation_log["epoch"].values
    loss = optimisation_log["loss"].values

    param_names = []
    for c in optimisation_log.columns:
        if c not in ("epoch", "loss"):
            param_names.append(c)

    fig, axes = plt.subplots(len(param_names), 1, figsize=(8, 2 * (len(param_names) + 1)), sharex=True)

    plot_loss_panel(axes[0], epochs, loss)

    for ax, param in zip(axes[1:], param_names):
        values = optimisation_log[param].values
        ax.plot(epochs, values)
        ax.set_ylabel(param)
        ax.grid(True)

    plt.tight_layout()
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(f'{save_folder}/parameter_optimisation_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.show()

def plot_loss_panel(ax, epochs, loss):
    lowest_loss = np.min(loss)
    best_idx = np.argmin(loss)
    best_epoch = epochs[best_idx]
    ax.plot(epochs, loss, color="black")
    ax.scatter(best_epoch, lowest_loss, color="green", label=f"Best Epoch: {best_epoch}")
    ax.set_yscale("log")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()


def load_optimisation_log(logs_dir):
    log_path = os.path.join(logs_dir, "optimisation_log.csv")
    optimisation_log = pd.read_csv(log_path)
    return optimisation_log


