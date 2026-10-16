"""Create synthetic experiment 02: fixed poles with increasing spread.

The image and audio distributions remain anchored at the west and east
poles. Across five arrangements their angular spread increases until both
modalities cover the complete unit sphere.

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
    normalize_rows,
    plot_arrangements,
    rotation_about_y,
    validate_arrangements,
    write_metrics_csv,
)


TARGET_MODALITY_NOISE_STD = 0.025
SPREAD_FRACTIONS = (0.18, 0.38, 0.58, 0.79, 1.0)
STAGE_NAMES = (
    'stage_1_tight_caps',
    'stage_2_broader_caps',
    'stage_3_hemispherical',
    'stage_4_nearly_global',
    'stage_5_full_sphere',
)
STAGE_TITLES = (
    '1. Tight pole caps\n18% angular spread',
    '2. Broader caps\n38% angular spread',
    '3. Hemispherical\n58% angular spread',
    '4. Nearly global\n79% angular spread',
    '5. Full sphere\n100% angular spread',
)
FIGURE_TITLE = (
    'Synthetic experiment 02: fixed east–west modalities progressively spread'
)
FIGURE_SUBTITLE = (
    'The modality centers remain fixed while both distributions broaden'
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / 'outputs'
    / 'extra_plots'
    / 'synthetic_experiments'
    / '02_east_west_progressive_spread'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Create fixed-pole synthetic embeddings with increasing spread.'
        ),
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


def make_global_targets(seed):
    """Return two nearly matched samples from the uniform sphere."""
    rng = np.random.default_rng(seed)
    image_targets = normalize_rows(
        rng.normal(size=(NUM_PAIRS, 3)))
    audio_targets = normalize_rows(
        image_targets
        + rng.normal(
            scale=TARGET_MODALITY_NOISE_STD,
            size=image_targets.shape,
        )
    )
    return image_targets, audio_targets


def compress_toward_north(local_directions, spread_fraction):
    """Scale every polar angle while preserving its azimuth."""
    polar_angles = np.arccos(
        np.clip(local_directions[:, 2], -1.0, 1.0))
    azimuths = np.arctan2(
        local_directions[:, 1],
        local_directions[:, 0],
    )
    scaled_polar_angles = spread_fraction * polar_angles
    sine = np.sin(scaled_polar_angles)
    return np.column_stack((
        sine * np.cos(azimuths),
        sine * np.sin(azimuths),
        np.cos(scaled_polar_angles),
    ))


def make_arrangements(seed=0):
    image_targets, audio_targets = make_global_targets(seed)
    west_rotation = rotation_about_y(-0.5 * np.pi)
    east_rotation = rotation_about_y(0.5 * np.pi)

    # Express each full-sphere target relative to its modality's fixed pole.
    image_local_targets = image_targets @ west_rotation
    audio_local_targets = audio_targets @ east_rotation

    arrangements = []
    for spread_fraction in SPREAD_FRACTIONS:
        image_local = compress_toward_north(
            image_local_targets, spread_fraction)
        audio_local = compress_toward_north(
            audio_local_targets, spread_fraction)
        image_emb = image_local @ west_rotation.T
        audio_emb = audio_local @ east_rotation.T
        paired_audio, optimal_cost, paired_cost = mildly_suboptimal_pairing(
            image_emb, audio_emb)

        arrangements.append({
            'image_emb': image_emb,
            'audio_emb': paired_audio,
            'spread_fraction': spread_fraction,
            'optimal_pairing_cost': optimal_cost,
            'actual_pairing_cost': paired_cost,
        })
    return arrangements


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    arrangements = make_arrangements(args.seed)
    metric_values = calculate_metrics(arrangements)
    validate_arrangements(arrangements, metric_values)

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
    )

    for stage_name, arrangement in zip(STAGE_NAMES, arrangements):
        print(
            '{}: spread={:.0%}, optimal paired distance={:.6f}, '
            'actual={:.6f}'
            .format(
                stage_name,
                arrangement['spread_fraction'],
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
