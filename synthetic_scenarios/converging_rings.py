"""Create synthetic experiment 04: converging latitude rings.

The image ring starts near the north pole and the audio ring near the south
pole. Both move toward a shared latitude at 45 degrees from the north pole,
where the two modality distributions overlap.

Outputs: unit_spheres.png and metrics.csv in the output directory.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.extra_plots.plot_synthetic_east_west_convergence import (  # noqa: E402
    METRICS,
    NUM_PAIRS,
    calculate_metrics,
    mildly_suboptimal_pairing,
    plot_arrangements,
    validate_arrangements,
    write_metrics_csv,
)


IMAGE_START_POLAR_DEGREES = 20.0
AUDIO_START_POLAR_DEGREES = 160.0
TARGET_POLAR_DEGREES = 45.0
CONVERGENCE_FRACTIONS = np.linspace(0.0, 1.0, 5)
AZIMUTH_NOISE_STD = np.deg2rad(0.45)
POLAR_NOISE_STD = np.deg2rad(1.25)

IMAGE_POLAR_DEGREES = tuple(
    IMAGE_START_POLAR_DEGREES
    + fraction * (
        TARGET_POLAR_DEGREES - IMAGE_START_POLAR_DEGREES)
    for fraction in CONVERGENCE_FRACTIONS
)
AUDIO_POLAR_DEGREES = tuple(
    AUDIO_START_POLAR_DEGREES
    + fraction * (
        TARGET_POLAR_DEGREES - AUDIO_START_POLAR_DEGREES)
    for fraction in CONVERGENCE_FRACTIONS
)
RING_SEPARATION_DEGREES = tuple(
    abs(image_polar - audio_polar)
    for image_polar, audio_polar in zip(
        IMAGE_POLAR_DEGREES, AUDIO_POLAR_DEGREES)
)

STAGE_NAMES = (
    'stage_1_polar_rings',
    'stage_2_approaching',
    'stage_3_midway',
    'stage_4_nearby',
    'stage_5_overlapping',
)
STAGE_TITLES = tuple(
    '{}. {} rings\n{:.0f}° separation'.format(
        stage_index,
        stage_label,
        separation_degrees,
    )
    for stage_index, stage_label, separation_degrees in zip(
        range(1, 6),
        ('Polar', 'Approaching', 'Midway', 'Nearby', 'Overlapping'),
        RING_SEPARATION_DEGREES,
    )
)
FIGURE_TITLE = (
    'Synthetic experiment 04: north–south latitude rings converge'
)
FIGURE_SUBTITLE = (
    'Both rings meet at 45° from the north pole '
    '(the middle of the upper hemisphere)'
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / 'outputs'
    / 'extra_plots'
    / 'synthetic_experiments'
    / '04_latitude_ring_convergence'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create progressively converging spherical latitude rings.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Directory receiving unit_spheres.png and metrics.csv.',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Output image resolution (default: 150).',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Synthetic-data seed (default: 0).',
    )
    return parser.parse_args()


def spherical_ring(polar_angle, azimuths, polar_noise):
    noisy_polar = np.clip(
        polar_angle + polar_noise,
        1e-4,
        np.pi - 1e-4,
    )
    sine = np.sin(noisy_polar)
    return np.column_stack((
        sine * np.cos(azimuths),
        sine * np.sin(azimuths),
        np.cos(noisy_polar),
    ))


def make_ring_coordinates(seed):
    rng = np.random.default_rng(seed)
    base_azimuths = np.linspace(
        0.0, 2.0 * np.pi, NUM_PAIRS, endpoint=False)
    image_azimuths = (
        base_azimuths
        + rng.normal(scale=AZIMUTH_NOISE_STD, size=NUM_PAIRS)
    )
    audio_azimuths = (
        base_azimuths
        + rng.normal(scale=AZIMUTH_NOISE_STD, size=NUM_PAIRS)
    )
    image_polar_noise = rng.normal(
        scale=POLAR_NOISE_STD, size=NUM_PAIRS)
    audio_polar_noise = rng.normal(
        scale=POLAR_NOISE_STD, size=NUM_PAIRS)
    return (
        image_azimuths,
        audio_azimuths,
        image_polar_noise,
        audio_polar_noise,
    )


def make_arrangements(seed=0):
    (
        image_azimuths,
        audio_azimuths,
        image_polar_noise,
        audio_polar_noise,
    ) = make_ring_coordinates(seed)

    arrangements = []
    for image_degrees, audio_degrees, separation_degrees in zip(
            IMAGE_POLAR_DEGREES,
            AUDIO_POLAR_DEGREES,
            RING_SEPARATION_DEGREES):
        image_emb = spherical_ring(
            np.deg2rad(image_degrees),
            image_azimuths,
            image_polar_noise,
        )
        audio_emb = spherical_ring(
            np.deg2rad(audio_degrees),
            audio_azimuths,
            audio_polar_noise,
        )
        paired_audio, optimal_cost, paired_cost = mildly_suboptimal_pairing(
            image_emb, audio_emb)

        arrangements.append({
            'image_emb': image_emb,
            'audio_emb': paired_audio,
            'image_polar_degrees': image_degrees,
            'audio_polar_degrees': audio_degrees,
            'separation_degrees': separation_degrees,
            'optimal_pairing_cost': optimal_cost,
            'actual_pairing_cost': paired_cost,
        })
    return arrangements


def validate_ring_experiment(arrangements, metric_values):
    validate_arrangements(arrangements, metric_values)

    final_arrangement = arrangements[-1]
    if not np.isclose(
            final_arrangement['image_polar_degrees'],
            TARGET_POLAR_DEGREES):
        raise ValueError('Final image ring did not reach the target latitude.')
    if not np.isclose(
            final_arrangement['audio_polar_degrees'],
            TARGET_POLAR_DEGREES):
        raise ValueError('Final audio ring did not reach the target latitude.')
    if not np.isclose(final_arrangement['separation_degrees'], 0.0):
        raise ValueError('Final rings do not overlap.')


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    arrangements = make_arrangements(args.seed)
    metric_values = calculate_metrics(arrangements)
    validate_ring_experiment(arrangements, metric_values)

    csv_path = output_dir / 'metrics.csv'
    figure_path = output_dir / 'unit_spheres.png'
    write_metrics_csv(
        csv_path,
        metric_values,
        stage_names=STAGE_NAMES,
    )
    plot_arrangements(
        arrangements,
        metric_values,
        figure_path,
        args.dpi,
        stage_titles=STAGE_TITLES,
        figure_title=FIGURE_TITLE,
        figure_subtitle=FIGURE_SUBTITLE,
        view_elev=24,
        view_azim=-58,
    )

    for stage_name, arrangement in zip(STAGE_NAMES, arrangements):
        print(
            '{}: image={:.2f}°, audio={:.2f}°, separation={:.2f}°, '
            'optimal paired distance={:.6f}, actual={:.6f}'
            .format(
                stage_name,
                arrangement['image_polar_degrees'],
                arrangement['audio_polar_degrees'],
                arrangement['separation_degrees'],
                arrangement['optimal_pairing_cost'],
                arrangement['actual_pairing_cost'],
            ))
    for metric_name, _metric_label, _metric_fn in METRICS:
        print(
            '{}: {}'.format(
                metric_name,
                ', '.join(
                    '{:.6f}'.format(value)
                    for value in metric_values[metric_name]
                ),
            ))
    print('Saved metrics to {}'.format(csv_path))
    print('Saved unit-sphere plot to {}'.format(figure_path))


if __name__ == '__main__':
    main()
