"""Benchmark binned and raw point-cloud matching, for the continuous and integrated methods.

Run with:
    $HOME/.pyenv/versions/.stingenv/bin/python benchmarks/benchmark_matching.py
"""

import time
import types
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from sting import extract_streamline, gradient_descent, stream_lines_grad


MODEL_PARAMS = {
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


def make_streamer(n_raw=400):
    radii = jnp.linspace(500.0, 800.0, n_raw)
    ra, dec, velocity = stream_lines_grad.forward_model_at_radius(
        radii, MODEL_PARAMS, 1000.0
    )
    intensity = jnp.linspace(1.0, 2.0, n_raw)
    sigma = jnp.full(n_raw, 0.1)
    return types.SimpleNamespace(
        pc_coords=jnp.stack((ra, dec, velocity, intensity)),
        ra_sigma=sigma,
        dec_sigma=sigma,
        v_sigma=sigma,
        ra_data=ra,
        dec_data=dec,
        v_data=velocity,
        data=(ra, dec, velocity),
        uncertainties=(sigma, sigma, sigma),
    )


def benchmark(prepared, matching_method):
    loss_fn = jax.jit(
        lambda params: gradient_descent.chi2_loss(
            params,
            1000.0,
            prepared,
            loss_method=0,
            matching_method=matching_method,
        )[0]
    )
    value_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    start = time.perf_counter()
    loss_fn(MODEL_PARAMS).block_until_ready()
    jit_loss = time.perf_counter() - start
    start = time.perf_counter()
    loss_fn(MODEL_PARAMS).block_until_ready()
    steady_loss = time.perf_counter() - start

    start = time.perf_counter()
    value, grad = value_grad_fn(MODEL_PARAMS)
    jax.block_until_ready((value, grad))
    jit_value_grad = time.perf_counter() - start
    start = time.perf_counter()
    value, grad = value_grad_fn(MODEL_PARAMS)
    jax.block_until_ready((value, grad))
    steady_value_grad = time.perf_counter() - start
    return float(value), jit_loss, steady_loss, jit_value_grad, steady_value_grad


def save_plots(rows, output_dir):
    """Save the benchmark timing plot."""
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    for colour, method in (('C0', 'continuous'), ('C1', 'integrated')):
        binned = [row for row in rows if row['mode'] == method]
        raw = next(row for row in rows if row['mode'] == f'{method}_point_cloud')
        bins = [row['bins'] for row in binned]
        ax.plot(bins, [row['steady_value_grad_s'] for row in binned], 'o-', color=colour,
                label=f'{method}: value and gradient')
        ax.axhline(raw['steady_value_grad_s'], color=colour, linestyle='--', alpha=0.7,
                   label=f'{method}: raw point cloud')
    ax.set_xlabel('Number of metric bins')
    ax.set_ylabel('Steady-state time (s)')
    ax.set_title('Matching timing')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.25)
    ax.legend()

    timing_path = output_dir / 'benchmark_timings.png'
    fig.savefig(timing_path, dpi=160)
    plt.close(fig)
    return timing_path


def main():
    streamer = make_streamer()
    fieldnames = [
        'mode', 'bins', 'loss', 'jit_loss_s', 'steady_loss_s',
        'jit_value_grad_s', 'steady_value_grad_s',
    ]
    rows = []
    for method in ('continuous', 'integrated'):
        for n_bins in (5, 10, 20, 40):
            prepared = extract_streamline.prepare_binned_continuous_data(streamer, n_bins)
            result = benchmark(prepared, method)
            rows.append(dict(zip(fieldnames, (method, n_bins, *result))))

        prepared = extract_streamline.prepare_point_cloud_data(streamer)
        result = benchmark(prepared, f'{method}_point_cloud')
        rows.append(dict(zip(fieldnames, (f'{method}_point_cloud', streamer.pc_coords.shape[1], *result))))

    output_path = Path(__file__).with_name('benchmark_results.csv')
    with output_path.open('w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    timing_path = save_plots(rows, output_path.parent)

    print(','.join(fieldnames))
    for row in rows:
        print(','.join(str(row[field]) for field in fieldnames))
    print(f'Benchmark results saved to: {output_path}')
    print(f'Timing plot saved to: {timing_path}')


if __name__ == '__main__':
    main()
