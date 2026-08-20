import types

import jax.numpy as jnp

from sting import extract_streamline
from sting import gradient_descent
from sting import stream_lines_grad


def test_continuous_loss_matches_synthetic_point_cloud():
    model_params = {
        'mass': 1.0,
        'r0': 1000.0,
        'theta0': 0.5,
        'phi0': 0.2,
        'mu': 0.3,
        'v_r0': -2.0,
        'inc': 0.0,
        'pa': 0.0,
        'rmin': 20.0,
        'v_lsr': 0.0,
    }
    radii = jnp.linspace(500.0, 800.0, 6)
    ra, dec, velocity = stream_lines_grad.forward_model_at_radius(
        radii,
        model_params,
        1000.0,
    )
    streamer = types.SimpleNamespace(
        pc_coords=jnp.stack((ra, dec, velocity, jnp.arange(1, 7, dtype=jnp.float64))),
        ra_sigma=jnp.full(6, 0.1),
        dec_sigma=jnp.full(6, 0.1),
        v_sigma=jnp.full(6, 0.1),
        ra_data=ra,
        dec_data=dec,
    )

    prepared = extract_streamline.prepare_point_cloud_data(streamer)
    loss, trace, _ = gradient_descent.chi2_loss(
        model_params,
        1000.0,
        prepared,
        loss_method=0,
        matching_method='continuous',
    )

    assert bool(jnp.isfinite(loss))
    assert float(loss) < 1e-10
    assert int(trace['matching']['data_valid_points']) == 6
