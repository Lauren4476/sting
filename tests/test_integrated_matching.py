import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sting import extract_streamline
from sting import gradient_descent
from sting import stream_lines_grad

jax.config.update("jax_enable_x64", True)

DISTANCE = 300.0

MODEL_PARAMS = {
    'mass': 1.0,
    'r0': 1500.0,
    'theta0': 0.7,
    'phi0': 3.5,
    'mu': 0.2,
    'v_r0': 0.5,
    'inc': -0.5,
    'pa': 2.0,
    'rmin': 20.0,
    'deltar': 0.1,
    'v_lsr': 0.0,
}

SPHERICAL_KEYS = ('mass', 'r0', 'theta0', 'phi0', 'mu', 'v_r0', 'inc', 'pa')


def _spherical_params(**overrides):
    params = {key: MODEL_PARAMS[key] for key in SPHERICAL_KEYS}
    params.update(overrides)
    return params


def _synthetic_streamer(n=10, sigma_pos=0.3, sigma_v=0.2):
    """Data sitting exactly on the model streamline, one point per bin"""
    delta = jnp.linspace(0.05, 1.4, n)
    ra, dec, v, _ = stream_lines_grad.forward_model_at_delta(delta, MODEL_PARAMS, DISTANCE)
    ra_sigma = jnp.full(n, sigma_pos)
    dec_sigma = jnp.full(n, sigma_pos)
    v_sigma = jnp.full(n, sigma_v)
    return types.SimpleNamespace(
        pc_coords=jnp.stack((ra, dec, v, jnp.ones(n))),
        ra_data=ra, dec_data=dec, v_data=v,
        ra_sigma=ra_sigma, dec_sigma=dec_sigma, v_sigma=v_sigma,
        data=(ra, dec, v),
        uncertainties=(ra_sigma, dec_sigma, v_sigma),
    ), delta


# ---------------------------------------------------------------------------
# closed-form model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('theta0', [0.3, 0.7, 1.3, 1.9, 2.6])
def test_closed_form_matches_radius_model(theta0):
    params = _spherical_params(theta0=theta0)
    delta = jnp.linspace(0.0, jnp.pi / 2, 200)
    pos, vel, r = stream_lines_grad.evaluate_streamline_at_delta(delta, **params)
    state = stream_lines_grad.build_stream_quantities(
        params['mass'], params['r0'], params['theta0'], params['mu'], params['v_r0']
    )
    pos_r, vel_r, valid = stream_lines_grad.evaluate_streamline_at_radius(r, **params)
    # the radius model is only defined above rc/2, and loses accuracy at r0 through safe_arccos
    keep = np.asarray(valid) & (np.asarray(r) > 0.5 * float(state.rc)) & (np.asarray(r) < 0.999 * params['r0'])
    assert keep.sum() > 50
    for a, b in zip(pos, pos_r):
        np.testing.assert_allclose(np.asarray(a)[keep], np.asarray(b)[keep], atol=1e-3)
    for a, b in zip(vel, vel_r):
        np.testing.assert_allclose(np.asarray(a)[keep], np.asarray(b)[keep], atol=1e-4)


@pytest.mark.parametrize('theta0', [0.7, 2.1])
def test_closed_form_velocity_is_time_derivative_of_position(theta0):
    # dx/dt = dx/ddelta * ddelta/dt, with ddelta/dt = h / r^2 and h = vk0 rc sin(theta0)
    params = _spherical_params(theta0=theta0)
    state = stream_lines_grad.build_stream_quantities(
        params['mass'], params['r0'], params['theta0'], params['mu'], params['v_r0']
    )
    angular_momentum = state.vk0 * state.rc * jnp.sin(params['theta0'])
    for delta in (0.0, 0.4, 1.2):
        dpos = jax.jacfwd(
            lambda d: jnp.stack(stream_lines_grad.evaluate_streamline_at_delta(d, **params)[0])
        )(jnp.asarray(delta))
        _, vel, r = stream_lines_grad.evaluate_streamline_at_delta(delta, **params)
        np.testing.assert_allclose(dpos * angular_momentum / r ** 2, jnp.stack(vel), atol=1e-12)


def test_closed_form_endpoints():
    params = _spherical_params(inc=0.0, pa=0.0)  # identity sky rotation
    state = stream_lines_grad.build_stream_quantities(
        params['mass'], params['r0'], params['theta0'], params['mu'], params['v_r0']
    )
    (_, _, _), _, r_start = stream_lines_grad.evaluate_streamline_at_delta(0.0, **params)
    (_, _, z_end), _, r_end = stream_lines_grad.evaluate_streamline_at_delta(jnp.pi / 2, **params)
    sin_theta0 = jnp.sin(params['theta0'])
    assert float(r_start) == pytest.approx(params['r0'], rel=1e-12)
    assert float(r_end) == pytest.approx(float(state.rc * sin_theta0 ** 2 / (1 + state.nu * sin_theta0)), rel=1e-12)
    assert float(z_end) == pytest.approx(0.0, abs=1e-9)


def test_closed_form_gradients_finite_at_start():
    # the radius model uses arccos(1) at r0; the closed form has no such singularity
    grads = jax.grad(
        lambda p: jnp.sum(jnp.stack(stream_lines_grad.evaluate_streamline_at_delta(0.0, **p)[1]))
    )(_spherical_params())
    assert all(bool(jnp.isfinite(g)) for g in grads.values())


# ---------------------------------------------------------------------------
# integral along the curve
# ---------------------------------------------------------------------------

def test_log_diff_ndtr_matches_direct_and_is_finite_in_tails():
    a = jnp.array([-1.0, -3.0, 0.2])
    b = jnp.array([0.5, 2.0, 1.7])
    direct = jnp.log(jax.scipy.stats.norm.cdf(b) - jax.scipy.stats.norm.cdf(a))
    np.testing.assert_allclose(gradient_descent.log_diff_ndtr(a, b), direct, rtol=1e-12)
    tails = gradient_descent.log_diff_ndtr(jnp.array([40.0, -60.0, 3.0]), jnp.array([41.0, -59.0, 3.0]))
    assert bool(jnp.all(jnp.isfinite(tails)))


def _straight_line_terms(point, sigma, loss_method=0):
    nodes = jnp.linspace(-20.0, 20.0, 41)
    zeros = jnp.zeros_like(nodes)
    return gradient_descent.integrated_match_terms(
        nodes, zeros, zeros, jnp.linspace(0.0, 1.0, 41),
        *[jnp.array([value]) for value in point],
        *[jnp.array([value]) for value in sigma],
        loss_method=loss_method,
    )


def test_straight_line_gives_perpendicular_chi2():
    terms = _straight_line_terms((1.3, 0.5, 0.3), (1.0, 0.5, 0.3))
    # perpendicular chi2 = (0.5/0.5)^2 + (0.3/0.3)^2
    assert float(terms['loss'][0]) == pytest.approx(2.0, abs=1e-10)
    assert float(terms['matched_ra'][0]) == pytest.approx(1.3, abs=1e-8)


def test_point_at_end_of_line_is_penalised_by_2log2():
    terms = _straight_line_terms((20.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert float(terms['loss'][0]) == pytest.approx(2 * np.log(2.0), abs=1e-10)
    beyond = _straight_line_terms((23.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert float(beyond['loss'][0]) > float(terms['loss'][0])


@pytest.mark.parametrize('loss_method', [0, 1])
def test_integrated_match_recovers_data_on_the_model(loss_method):
    streamer, delta_true = _synthetic_streamer(sigma_pos=0.05, sigma_v=0.05)
    # the synthetic data starts inside r0, so leave out the outer anchor and test the integral alone
    prepared = gradient_descent.prepare_matching_data(streamer, 'integrated_point_cloud', anchor_outer_point=False)
    # small sigmas need more nodes than the default to keep the segments shorter than sigma
    matched = gradient_descent.match_model_to_data(
        prepared, MODEL_PARAMS, DISTANCE, 'integrated_point_cloud', loss_method=loss_method,
        integration_nodes=1024,
    )
    np.testing.assert_allclose(matched.best_u * jnp.pi / 2, delta_true, atol=5e-3)
    np.testing.assert_allclose(matched.ra_model_matched, streamer.ra_data, atol=5e-3)
    np.testing.assert_allclose(matched.v_model_matched, streamer.v_data, atol=5e-3)

    loss, trace, _ = gradient_descent.chi2_loss(
        MODEL_PARAMS, DISTANCE, prepared, loss_method=loss_method, matching_method='integrated_point_cloud',
        integration_nodes=1024,
    )
    # points on a smooth curve have loss ~0, up to curvature corrections
    assert abs(float(loss)) < 0.1 * prepared.ra_data.size
    assert float(trace['matching']['segment_length_max']) < gradient_descent.INTEGRATION_RESOLUTION_WARN


def test_integrated_loss_converges_with_nodes():
    streamer, _ = _synthetic_streamer()
    prepared = gradient_descent.prepare_matching_data(streamer, 'integrated')
    params = dict(MODEL_PARAMS, phi0=3.2, inc=-0.3)
    losses = [
        float(gradient_descent.chi2_loss(
            params, DISTANCE, prepared, matching_method='integrated', integration_nodes=k
        )[0])
        for k in (64, 128, 1024)
    ]
    assert abs(losses[1] - losses[2]) < 0.01 * abs(losses[2])
    assert abs(losses[1] - losses[2]) < abs(losses[0] - losses[2])


def test_integrated_gradient_matches_finite_difference_and_hessian_is_finite():
    streamer, _ = _synthetic_streamer()
    prepared = gradient_descent.prepare_matching_data(streamer, 'integrated')
    keys = ('phi0', 'inc', 'v_r0', 'mu')

    def loss(vec):
        params = dict(MODEL_PARAMS, **dict(zip(keys, vec)))
        return gradient_descent.chi2_loss(params, DISTANCE, prepared, loss_method=1, matching_method='integrated')[0]

    x0 = jnp.array([3.3, -0.4, 0.7, 0.25])
    grad = jax.grad(loss)(x0)
    for i in range(len(keys)):
        step = jnp.zeros_like(x0).at[i].set(1e-6)
        fd = (loss(x0 + step) - loss(x0 - step)) / 2e-6
        assert float(grad[i]) == pytest.approx(float(fd), rel=1e-5, abs=1e-6)
    assert bool(jnp.all(jnp.isfinite(jax.hessian(loss)(x0))))


def test_integrated_loss_has_no_jumps_over_phi0():
    # the 'continuous' method jumps by ~12 in this sweep, when the golden-section search swaps local minima
    streamer, _ = _synthetic_streamer()
    prepared = gradient_descent.prepare_matching_data(streamer, 'integrated')
    loss = jax.jit(lambda phi0: gradient_descent.chi2_loss(
        dict(MODEL_PARAMS, phi0=phi0), DISTANCE, prepared, loss_method=0, matching_method='integrated'
    )[0])
    grid = np.linspace(0.0, 2 * np.pi, 1441)
    values = np.array([float(loss(x)) for x in grid])
    worst = int(np.argmax(np.abs(np.diff(values))))
    lower, upper = grid[worst], grid[worst + 1]
    # bisect towards the steepest part of the worst step: a jump would keep its size, a smooth step shrinks
    for _ in range(20):
        mid = 0.5 * (lower + upper)
        if abs(float(loss(mid)) - float(loss(lower))) > abs(float(loss(upper)) - float(loss(mid))):
            upper = mid
        else:
            lower = mid
    assert abs(float(loss(upper)) - float(loss(lower))) < 1e-3


# ---------------------------------------------------------------------------
# outer anchor
# ---------------------------------------------------------------------------

def _streamer_from_delta(delta, sigma=0.2):
    ra, dec, v, _ = stream_lines_grad.forward_model_at_delta(delta, MODEL_PARAMS, DISTANCE)
    sigmas = jnp.full(delta.size, sigma)
    return types.SimpleNamespace(
        pc_coords=jnp.stack((ra, dec, v, jnp.ones(delta.size))),
        ra_data=ra, dec_data=dec, v_data=v,
        ra_sigma=sigmas, dec_sigma=sigmas, v_sigma=sigmas,
        data=(ra, dec, v), uncertainties=(sigmas, sigmas, sigmas),
    )


def test_outer_anchor_uses_outermost_bin():
    streamer = _streamer_from_delta(jnp.linspace(0.0, 1.4, 10))
    prepared = gradient_descent.prepare_matching_data(streamer, 'integrated')
    ra0, dec0, v0, _ = stream_lines_grad.forward_model_at_delta(0.0, MODEL_PARAMS, DISTANCE)
    np.testing.assert_allclose(prepared.outer_point, jnp.stack([ra0, dec0, v0]), atol=1e-12)
    for method in ('continuous', 'continuous_point_cloud', 'integrated_point_cloud'):
        np.testing.assert_allclose(
            gradient_descent.prepare_matching_data(streamer, method).outer_point, prepared.outer_point
        )
    for method in ('continuous', 'integrated'):
        assert gradient_descent.prepare_matching_data(streamer, method, anchor_outer_point=False).outer_point is None


@pytest.mark.parametrize('theta0', [0.7, 2.1])
def test_model_start_matches_radius_model_at_r0(theta0):
    # the continuous anchor evaluates the start with the closed form; it must be the radius model's start too
    params = dict(MODEL_PARAMS, theta0=theta0)
    start_delta = jnp.stack(stream_lines_grad.forward_model_at_delta(0.0, params, DISTANCE)[:3])
    start_radius = jnp.stack(stream_lines_grad.forward_model_at_radius(params['r0'], params, DISTANCE))
    # the radius model clips its arccos at r0 (safe_arccos), which moves its start by ~1e-3 au
    np.testing.assert_allclose(start_delta, start_radius, atol=1e-4)


@pytest.mark.parametrize('method', ['continuous', 'continuous_point_cloud'])
def test_continuous_outer_anchor_is_zero_at_truth_and_constrains_r0(method):
    streamer = _streamer_from_delta(jnp.linspace(0.0, 1.0, 10))

    def loss_and_trace(r0, anchor):
        prepared = gradient_descent.prepare_matching_data(streamer, method, anchor_outer_point=anchor)
        return gradient_descent.chi2_loss(
            dict(MODEL_PARAMS, r0=r0), DISTANCE, prepared, loss_method=1, matching_method=method
        )

    trace = loss_and_trace(MODEL_PARAMS['r0'], True)[1]
    assert float(trace['chi2_components']['chi2_outer']) == pytest.approx(0.0, abs=1e-10)
    step = 300.0
    rise_anchor = float(loss_and_trace(MODEL_PARAMS['r0'] + step, True)[0] - loss_and_trace(MODEL_PARAMS['r0'], True)[0])
    rise_free = float(loss_and_trace(MODEL_PARAMS['r0'] + step, False)[0] - loss_and_trace(MODEL_PARAMS['r0'], False)[0])
    assert rise_anchor > 10 * max(rise_free, 1e-3)
    grad = jax.grad(lambda r0: loss_and_trace(r0, True)[0])(MODEL_PARAMS['r0'] + step)
    assert bool(jnp.isfinite(grad)) and float(grad) > 0.0


@pytest.mark.parametrize('loss_method', [0, 1])
def test_outer_anchor_is_zero_at_truth_and_constrains_r0(loss_method):
    streamer = _streamer_from_delta(jnp.linspace(0.0, 1.4, 10))

    def loss_and_trace(r0, anchor):
        prepared = gradient_descent.prepare_matching_data(streamer, 'integrated', anchor_outer_point=anchor)
        return gradient_descent.chi2_loss(
            dict(MODEL_PARAMS, r0=r0), DISTANCE, prepared, loss_method=loss_method, matching_method='integrated'
        )

    _, trace = loss_and_trace(MODEL_PARAMS['r0'], True)[:2]
    assert float(trace['chi2_components']['chi2_outer']) == pytest.approx(0.0, abs=1e-20)
    # extending the model beyond the data costs much more with the anchor
    step = 300.0
    rise_anchor = float(loss_and_trace(MODEL_PARAMS['r0'] + step, True)[0] - loss_and_trace(MODEL_PARAMS['r0'], True)[0])
    rise_free = float(loss_and_trace(MODEL_PARAMS['r0'] + step, False)[0] - loss_and_trace(MODEL_PARAMS['r0'], False)[0])
    assert rise_anchor > 10 * max(rise_free, 1e-3)


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------

def test_prepare_matching_data_dispatch():
    streamer, _ = _synthetic_streamer()
    assert gradient_descent.prepare_matching_data(streamer, 'legacy') is None
    binned = gradient_descent.prepare_matching_data(streamer, 'integrated')
    cloud = gradient_descent.prepare_matching_data(streamer, 'integrated_point_cloud')
    assert isinstance(binned, extract_streamline.PreparedContinuousData)
    assert isinstance(cloud, extract_streamline.PreparedContinuousData)
    with pytest.raises(ValueError):
        gradient_descent.prepare_matching_data(streamer, 'nearest')


def test_model_curve_for_integrated_ends_at_disk_midplane():
    ra, dec, v, valid = gradient_descent.model_curve(MODEL_PARAMS, DISTANCE, 'integrated', n_curve_points=50)
    assert ra.shape == (50,) and bool(jnp.all(valid))
    ra0, dec0, _, _ = stream_lines_grad.forward_model_at_delta(0.0, MODEL_PARAMS, DISTANCE)
    assert float(ra[0]) == pytest.approx(float(ra0)) and float(dec[0]) == pytest.approx(float(dec0))


def test_evaluate_initial_guess_integrated():
    streamer, _ = _synthetic_streamer()
    opt_params = {key: MODEL_PARAMS[key] for key in ('r0', 'theta0', 'phi0', 'mu', 'v_r0', 'inc', 'pa', 'mass')}
    fixed_params = {key: MODEL_PARAMS[key] for key in ('rmin', 'deltar', 'v_lsr')}
    result = gradient_descent.evaluate_initial_guess(
        opt_params, fixed_params, streamer.data, streamer.uncertainties, DISTANCE,
        n_elements=10, loss_method=0, matching_method='integrated',
    )
    assert np.isfinite(result.chi2_total)
    assert set(result.chi2_components) == {'chi2_ra', 'chi2_dec', 'chi2_v', 'chi2_outer', 'chi2_prior', 'chi2_total'}
    assert np.asarray(result.ra_model_interp).shape == (10,)


def test_fit_streamline_integrated_runs_and_estimates_errors():
    streamer, _ = _synthetic_streamer()
    opt_params = {'phi0': 3.4, 'inc': -0.45}
    fixed_params = {key: value for key, value in MODEL_PARAMS.items() if key not in opt_params}
    result = gradient_descent.fit_streamline(
        opt_params, fixed_params, streamer, DISTANCE,
        learning_rate=0.01, n_epochs=5, info_every=100,
        save_folder=None, loss_method=1, matching_method='integrated',
    )
    assert len(result.loss_history) >= 2
    assert result.loss_history[-1] < result.loss_history[0]
    assert result.param_errors is not None
    assert all(np.isfinite(value) for value in result.param_errors.values())



