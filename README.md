# STING

## **ST**reamer **IN**fall with **G**radients

[![PyPI version](https://img.shields.io/pypi/v/sting.svg)](https://pypi.org/project/sting/)
[![codecov](https://codecov.io/gh/Lauren4476/sting/graph/badge.svg?token=QABJ72IB52)](https://codecov.io/gh/Lauren4476/sting)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Lauren4476/sting/blob/main/LICENCE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://github.com/Lauren4476/sting)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://github.com/Lauren4476/sting)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://github.com/Lauren4476/sting)

STING (STreamer INfall with Gradients) is a Python package for quickly fitting streamline models to molecular line position-position-velocity (PPV) data of asymmetric infalling material around young protostars. It uses a gradient descent method, powered by [JAX](https://github.com/jax-ml/jax) and [Optax](https://github.com/google-deepmind/optax), and can find best-fit streamline parameters of a streamer, and their uncertainties, in 10-20 seconds on a normal astronomy laptop (e.g. MacBook). 

The streamline model used is the analytic solutions of [Mendoza et al. (2009)](https://doi.org/10.1111/j.1365-2966.2008.14210.x). 

## What does STING do?

With STING you can:

1. **Extract a 1D streamline from a spectral cube.** Given a PPV cube containing candidate streamer emission, and the coordinates of its protostar, STING reduces the PPV cube to a set of `npoints` representative (RA offset, Dec offset, velocity) points with corresponding uncertainties
2. **Model a streamline.** The streamline model used by STING is taken from [Mendoza et al. (2009)](https://doi.org/10.1111/j.1365-2966.2008.14210.x) — a ballistic model of rotating infall, dominated by the gravitational force of a sink mass at the star position.
3. **Quickly fit the model to your data.** Streamline parameters are optimised with the Adam optimiser to minimise a chi-squared loss between the model and the observed streamline. Current parameters supported for optimisation: `r0`, `theta0`, `phi0`, `v_r0`, `omega` or `rc`, `mass`, `inc`, `pa`. You can optimise any combination of these.
4. **Quantify uncertainties.** Parameter uncertainties are estimated from the Hessian of the loss at the best-fit point, and propagated into the model (e.g. as "streamline spaghetti" plots).
5. **Visualise the result.** Pre-built plotting functions include: morphology, velocity vs. projected radius on sky plane, loss curves, parameter correlation and uncertainty sampling, and per-epoch animations of the fit converging.


## Installation

It is recommended to install STING in a virtual environment:

```bash
python -m venv /path/to/new/venv
source /path/to/new/venv/bin/activate
```

### pip (recommended)

The quickest way to install STING is 

```bash
pip install sting
```

which also installs all of STING's dependencies automatically.

If you want to install with additional development dependencies (e.g. for running the test suite):

```bash
pip install sting[dev]
```

### from source

```bash
git clone https://github.com/Lauren4476/sting.git
cd sting
pip install .
```
installs STING and all of its dependencies. 

### Notes on dependencies

STING requires **Python ≥ Python 3.12**.

Dependencies (installed automatically when using either of the methods above):
- `numpy>=2.0.2`
- `astropy>=7.0.0`
- `jax>=0.9.0`
- `optax>=0.2.6`
- `matplotlib>=3.9.0`
- `pandas>=2.2.0`
- `spectral_cube>=0.6.6`

## Quick start

A full  worked example with data is provided in [`examples/sting_example_run.ipynb`](https://github.com/Lauren4476/sting/tree/main/examples). The core workflow is:

```python
# --- Imports ---
from astropy import units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from spectral_cube import SpectralCube
from sting import extract_streamline, gradient_descent, outputs

# --- Settings ---
star_position = SkyCoord("3h28m55.569s", "+31d14m37.025s", frame='fk5')
distance = 293        # distance to protostar star in parsecs
v_lsr = 7.5           # km/s, systemic velocity

# --- 1. Extract 1D streamline from the cube ---
hdu = fits.open("data/example_streamer_cluster_data.fits")[0]
cube = SpectralCube.read(hdu).with_spectral_unit(
    u.km / u.s, rest_value=hdu.header["RESTFRQ"] * u.Hz
)
streamer_cube = extract_streamline.extract_streamer_subcube(
    cube,
    vmin=6 * u.km / u.s, vmax=8 * u.km / u.s,
    xmin=-5 * u.arcsec, xmax=5 * u.arcsec,
    ymin=-12 * u.arcsec, ymax=0.5 * u.arcsec,
    rms_thresh=4,
)
streamer = extract_streamline.reduce_to_1D(streamer_cube, star_position, n_elements=10)

# --- 2. Set an initial guess ---
# parameters you want STING to optimise
initial_opt_params = {
    'r0': 1500.0 * u.au,
    'theta0': 40.0 * u.deg,
    'phi0': 100.0 * u.deg,
    'omega': 5e-13 * (1 / u.s),
    'v_r0': 0.1 * u.km / u.s,
    'mass': 4.0 * u.Msun,
    'inc': -45 * u.deg,
    'pa': 194 * u.deg
}
# parameters you want to keep fixed
fixed_params = {
    'rmin': 50.0 * u.au,
    'deltar': 30.0 * u.au,
    'v_lsr': v_lsr * u.km / u.s,
}
model_params, initial_opt_params, fixed_params = gradient_descent.prepare_model_params(
    initial_opt_params, fixed_params
)

# --- 3. Set bounds for 'r0', 'mass', if any of these are in initial_opt_params ---
param_bounds = {
    'r0': (200.0, 10000.0) * u.au,
    'mass': (3.0, 5.0) * u.Msun,
}

# --- 4. (optional) It is recommended to set priors for 'mass', 'inc', 'pa' if any of these are in initial_opt_params, to help break degeneracy ---
# format 'parameter': (mean, sigma) * u.Unit
priors = {
    'mass': (4.0, 1.0) * u.Msun,
    'inc': (-45, 10) * u.deg,
    'pa': (194, 10) * u.deg
}

# --- 5. Fit the streamline ---
best_opt_params, loss_history, param_errors = gradient_descent.fit_streamline(
    initial_opt_params,
    fixed_params,
    streamer,
    distance,
    param_bounds=param_bounds,
    priors=priors
    n_epochs=1000,
    v_lsr=v_lsr,
    save_folder="sting_results",
)
```

This saves best-fit parameters, an optimisation log, and diagnostic plots to `sting_results/`. From there, the `outputs` module offers further plotting and analaysis tools.

## Recommended use case (best practice)

STING is intended for fitting kinematic streamer models to interferometric molecular line observations (e.g. ALMA, NOEMA) of young stellar objects with asymmetric infall candidates visible in PPV space. It is best suited to:

- High spectral and spatial resolution observations (~0.1 km/s, ~300 au resolution or better)
- Sources with a well-constrained distance and systemic velocity
- Sources with some prior constraints on sink mass, inclination, and position axis.
- Streamer candidates that have already been isolated from the rest of the cube by preprocessing.

### Priors

It is possible to run STING without specifying a `priors` argument. However, PPV observations only contain data in 3 dimensions (RA, Dec, LOS velocity), while the streamline model has 6 dimensions (3D position, 3D velocity). As a result, some parameter combinations are "coupled" and may be only weakly constrained by the observations.

Common degeneracies:
- sink mass, `mass`, and angular velocity, `omega` or centrifugal radius, `rc`
- source inclination, `inc`, and streamline initial polar angle, `theta0`
- source position angle, `pa`, and streamline starting azimuthal angle, `phi0`

As a result, when both parameters in one of these pairs are allowed to vary optimisation may converge more slowly as the optimiser drifts along a nearly-flat direction, and **uncertainty estimates will become large**.

STING supports Gaussian priors on any optimisable parameter, specified as (mean, sigma). These are included in the loss function as additional chi-squared penalty terms, penalising the optimiser for adjusting parameters further from the prior mean. Where independent constraints are available from previous work, **I recommend applying priors to `mass`, `inc`, and `pa`**. This helps constrain otherwise degenerate parameter combinations and reduces uncertainties.

### Streamer preprocessing

STING's `extract_streamline` module assumes you hand it a cube that already contains mostly streamer emission — it only applies simple velocity/spatial cuts and an RMS threshold (in `extract_streamer_subcube`). **It cannot yet** separate streamer emission from overlapping envelope, outflow, or disk emission for you. (stay tuned)

The preprocessing steps I use and recommend to isolate a **pure streamer cube** for input to STING are:

- **Multi-Gaussian spectral fitting** for the spectrum at each pixel in the original PPV cube to separate kinematically distinct components (e.g. streamer vs. envelope vs. disc) that overlap in velocity. Tools like [`pyspeckit`](https://pyspeckit.readthedocs.io/) or [`scousepy`](https://github.com/jdhenshaw/scousepy) work well for this. I use the [AIC criterion](https://en.wikipedia.org/wiki/Akaike_information_criterion) to determine which number of Gaussians best fit each spectrum.
- **DBSCAN clustering** (e.g. with [`scikit-learn`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)) on the resulting collection of gaussians, to extract the spatially and kinematically coherent cluster of gaussians that corresponds to the streamer candidate, discarding noise and unrelated structures.

## Source code overview

| Module | Contains |
|---|---|
| `extract_streamline` | Extracts streamer sub-cubes and reduces them to a 1D weighted-mean streamline with uncertainties. |
| `stream_lines_grad` | JAX-differentiable forward model of the streamline, following [Mendoza et al. (2009)](https://doi.org/10.1111/j.1365-2966.2008.14210.x). |
| `gradient_descent` | Main body. Parameter preparation, loss function, and Adam-based `fit_streamline` optimisation loop. |
| `errors` | Streamline parameter uncertainty estimation at the best fit parameters. |
| `outputs` | Result-saving and plotting: morphology, position–velocity diagrams, loss curves, correlation/uncertainty plots, and per-epoch animations. |

## Contributing / Issues

Contributions of all kinds are very welcome! Here's how to do it.

### Reporting bugs and getting support

Reports of bugs or unexpected behaviour are welcome via opening an Issue on the [GitHub issue tracker](https://github.com/Lauren4476/sting/issues). To help diagnose the problem, please include:
- a short description of the problem and what you expected to happen instead
- a code example that reproduces the issue
- the versions of STING and its dependencies (listed above) you are using

For general questions about how to use STING, open an Issue with the label `question`. You can also get in touch directly at [lmason@mpe.mpg.de](mailto:lmason@mpe.mpg.de).


### Contributing code

Bug fixes and contributions of new features are very welcome. To contribute code:
- fork the repository on GitHub, and create a branch for your change
- install STING in development mode:
```pip install -e ".[dev]"```
- make your changes, and add or update the tests as needed
- run the test suite to check nothing is broken: ```pytest```
- open a [pull request](https://github.com/Lauren4476/sting/pulls) against the `main` branch with a description of what your change does and why

If you have an idea for a feature but are unsure how to approach it, feel free to open an Issue to discuss it first!

All contributors who submit an accepted pull request will be listed in the repository's Contributor list on GitHub.

### Known issues

- **The radius sampling grid must reach down to `r_low`.** `r_low = max(rmin, 0.5*rc)`. For jit compilation (what makes STING so fast, see [this explanation](https://docs.jax.dev/en/latest/jit-compilation.html)), arrays must be constant length across optimisation epochs. If `npoints` and `deltar` are too small for the chosen `r0`, the sampled radius grid won't extend down to `r_low`, and the model will raise an error. If you encounter this, increase `npoints` and/or `deltar`.


## Citing STING

tbc, get in touch at [lmason@mpe.mpg.de](mailto:lmason@mpe.mpg.de)

## Credits

STING is written and maintained by **Lauren Mason** ([lmason@mpe.mpg.de](mailto:lmason@mpe.mpg.de), [@Lauren4476](https://github.com/Lauren4476)).

## Licence

STING is released under the [MIT Licence](https://github.com/Lauren4476/sting/blob/main/LICENCE).
