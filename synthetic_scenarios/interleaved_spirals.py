"""Create synthetic experiment 05: progressively spreading double spirals.

Two interleaved spiral arms lie on the camera-facing hemisphere. Their
centerlines remain fixed while modality-specific transverse spread increases,
eventually making the arms overlap substantially without covering the full
sphere.

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
    write_metrics_csv,
)


VIEW_ELEVATION = 24.0
VIEW_AZIMUTH = -58.0
SPIRAL_AXIS_ELEVATION = 36.0
SPIRAL_AXIS_AZIMUTH = -42.0
NUM_TURNS = 1.75
MINIMUM_DISK_RADIUS = 0.06
MAXIMUM_DISK_RADIUS = 0.93
MAXIMUM_NOISY_RADIUS = 0.975
SPREAD_STANDARD_DEVIATIONS = (0.004, 0.025, 0.050, 0.075, 0.105)

STAGE_NAMES = (
    'stage_1_thin',
    'stage_2_light_spread',
    'stage_3_moderate_spread',
    'stage_4_broad',
    'stage_5_overlapping',
)
STAGE_TITLES = (
    '1. Thin spirals\nσ = 0.004',
    '2. Light spread\nσ = 0.025',
    '3. Moderate spread\nσ = 0.050',
    '4. Broad spirals\nσ = 0.075',
    '5. Strong overlap\nσ = 0.105',
)
FIGURE_TITLE = (
    'Synthetic experiment 05: interleaved spirals progressively spread'
)
FIGURE_SUBTITLE = (
    'The spiral plane is tilted relative to the camera; '
    'only transverse spread increases'
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / 'outputs'
    / 'extra_plots'
    / 'synthetic_experiments'
    / '05_interleaved_spiral_spread'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create progressively spreading interleaved spirals.',
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


def direction_from_elevation_azimuth(elevation_degrees, azimuth_degrees):
    elevation = np.deg2rad(elevation_degrees)
    azimuth = np.deg2rad(azimuth_degrees)
    return np.asarray([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])


def spiral_basis():
    spiral_axis = direction_from_elevation_azimuth(
        SPIRAL_AXIS_ELEVATION,
        SPIRAL_AXIS_AZIMUTH,
    )
    first_tangent = np.cross(np.asarray([0.0, 0.0, 1.0]), spiral_axis)
    first_tangent /= np.linalg.norm(first_tangent)
    second_tangent = np.cross(spiral_axis, first_tangent)
    second_tangent /= np.linalg.norm(second_tangent)
    return spiral_axis, first_tangent, second_tangent


def make_spiral_centerlines():
    progress = np.linspace(0.0, 1.0, NUM_PAIRS)
    radii = (
        MINIMUM_DISK_RADIUS
        + progress * (MAXIMUM_DISK_RADIUS - MINIMUM_DISK_RADIUS)
    )
    angles = 2.0 * np.pi * NUM_TURNS * progress
    image_centerline = np.column_stack((
        radii * np.cos(angles),
        radii * np.sin(angles),
    ))
    audio_centerline = np.column_stack((
        radii * np.cos(angles + np.pi),
        radii * np.sin(angles + np.pi),
    ))
    return image_centerline, audio_centerline


def keep_inside_disk(coordinates):
    radii = np.linalg.norm(coordinates, axis=1)
    outside = radii > MAXIMUM_NOISY_RADIUS
    if np.any(outside):
        coordinates = coordinates.copy()
        coordinates[outside] *= (
            MAXIMUM_NOISY_RADIUS / radii[outside]
        )[:, None]
    return coordinates


def disk_to_visible_hemisphere(coordinates):
    spiral_axis, first_tangent, second_tangent = spiral_basis()
    squared_radii = np.sum(coordinates ** 2, axis=1)
    spiral_axis_components = np.sqrt(
        np.clip(1.0 - squared_radii, 0.0, None))
    return (
        coordinates[:, :1] * first_tangent
        + coordinates[:, 1:] * second_tangent
        + spiral_axis_components[:, None] * spiral_axis
    )


def make_arrangements(seed=0):
    rng = np.random.default_rng(seed)
    image_centerline, audio_centerline = make_spiral_centerlines()
    image_noise = rng.normal(size=image_centerline.shape)
    audio_noise = rng.normal(size=audio_centerline.shape)

    arrangements = []
    for spread_std in SPREAD_STANDARD_DEVIATIONS:
        image_disk = keep_inside_disk(
            image_centerline + spread_std * image_noise)
        audio_disk = keep_inside_disk(
            audio_centerline + spread_std * audio_noise)
        image_emb = disk_to_visible_hemisphere(image_disk)
        audio_emb = disk_to_visible_hemisphere(audio_disk)
        paired_audio, optimal_cost, paired_cost = mildly_suboptimal_pairing(
            image_emb, audio_emb)

        arrangements.append({
            'image_emb': image_emb,
            'audio_emb': paired_audio,
            'spread_std': spread_std,
            'optimal_pairing_cost': optimal_cost,
            'actual_pairing_cost': paired_cost,
        })
    return arrangements


def validate_spiral_experiment(arrangements, metric_values):
    if len(arrangements) != 5:
        raise ValueError(
            'Expected five arrangements, got {}.'.format(
                len(arrangements)))

    spiral_axis, _first_tangent, _second_tangent = spiral_basis()
    camera_axis = direction_from_elevation_azimuth(
        VIEW_ELEVATION,
        VIEW_AZIMUTH,
    )
    for arrangement in arrangements:
        image_emb = arrangement['image_emb']
        audio_emb = arrangement['audio_emb']
        if image_emb.shape != (NUM_PAIRS, 3):
            raise ValueError(
                'Unexpected image shape: {}.'.format(image_emb.shape))
        if audio_emb.shape != image_emb.shape:
            raise ValueError(
                'Image/audio shapes differ: {} and {}.'
                .format(image_emb.shape, audio_emb.shape))
        np.testing.assert_allclose(
            np.linalg.norm(image_emb, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(
            np.linalg.norm(audio_emb, axis=1), 1.0, atol=1e-12)
        if np.min(image_emb @ spiral_axis) <= 0.0:
            raise ValueError('Image spiral left its supporting hemisphere.')
        if np.min(audio_emb @ spiral_axis) <= 0.0:
            raise ValueError('Audio spiral left its supporting hemisphere.')
        camera_facing_fraction = np.mean(
            np.concatenate((
                image_emb @ camera_axis,
                audio_emb @ camera_axis,
            )) > 0.0
        )
        if camera_facing_fraction < 0.90:
            raise ValueError(
                'Only {:.1%} of embeddings face the camera.'
                .format(camera_facing_fraction))

    spreads = [
        arrangement['spread_std']
        for arrangement in arrangements
    ]
    if not np.all(np.diff(spreads) > 0.0):
        raise ValueError(
            'Spiral spread is not strictly increasing: {}.'
            .format(spreads))
    if (
            metric_values['nn_overlap_at_50'][-1]
            <= metric_values['nn_overlap_at_50'][0]):
        raise ValueError('Final NNO@50 did not improve over the thin spirals.')
    if (
            metric_values['modality_silhouette_score'][-1]
            >= metric_values['modality_silhouette_score'][0]):
        raise ValueError(
            'Final silhouette score did not decrease as spirals overlapped.')


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    arrangements = make_arrangements(args.seed)
    metric_values = calculate_metrics(arrangements)
    validate_spiral_experiment(arrangements, metric_values)

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
        view_elev=VIEW_ELEVATION,
        view_azim=VIEW_AZIMUTH,
        pair_line_width=0.50,
        pair_line_alpha=0.14,
    )

    for stage_name, arrangement in zip(STAGE_NAMES, arrangements):
        print(
            '{}: spread_std={:.3f}, optimal paired distance={:.6f}, '
            'actual={:.6f}'
            .format(
                stage_name,
                arrangement['spread_std'],
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
