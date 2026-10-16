"""Create synthetic experiment 06: one-sided semicircle overlap.

Two long, horizontally oriented small-circle arcs begin as narrow, separated
modality bands.  Across five stages, 0%, 25%, 50%, 75%, and 100% of the arcs
are transformed, always starting from the same end.  The transformed section
moves onto a shared centerline and becomes broader, while every untransformed
point remains exactly at its stage-one position.

Outputs: unit_spheres.png and metrics.csv in the output directory.
"""

import argparse
import sys
from importlib import import_module
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Load the numbered scenario module; its filename is not a Python identifier.
_shared_clouds = import_module('1_converging_clouds')
EMBEDDING_DIMENSION = _shared_clouds.EMBEDDING_DIMENSION
METRICS = _shared_clouds.METRICS
NUM_PAIRS = _shared_clouds.NUM_PAIRS
calculate_metrics = _shared_clouds.calculate_metrics
mildly_suboptimal_pairing = _shared_clouds.mildly_suboptimal_pairing
plot_arrangements = _shared_clouds.plot_arrangements
write_metrics_csv = _shared_clouds.write_metrics_csv


VIEW_ELEVATION = 25.0
VIEW_AZIMUTH = 40.0
ARC_HALF_ANGLE_DEGREES = 65.0
ARC_BOW_HEIGHT = 0.12
IMAGE_VERTICAL_CENTER = -0.25
AUDIO_VERTICAL_CENTER = 0.25
BASE_VERTICAL_NOISE_STD = 0.008
BASE_ANGULAR_NOISE_STD = 0.005
OVERLAP_VERTICAL_NOISE_STD = 0.060
OVERLAP_ANGULAR_NOISE_STD = 0.018
OVERLAP_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)

STAGE_NAMES = (
    'stage_1_separate',
    'stage_2_25_percent_overlap',
    'stage_3_50_percent_overlap',
    'stage_4_75_percent_overlap',
    'stage_5_full_overlap',
)
STAGE_TITLES = (
    '1. Separate semicircles\n0% transformed',
    '2. One-quarter overlap\n25% transformed',
    '3. Half overlap\n50% transformed',
    '4. Three-quarter overlap\n75% transformed',
    '5. Full overlap\n100% transformed',
)
FIGURE_TITLE = (
    'Synthetic experiment 06: semicircles progressively overlap from one end'
)
FIGURE_SUBTITLE = (
    'Each transformed section broadens and meets at the midpoint; '
    'the remainder stays exactly fixed'
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / 'outputs' / 'hybrid_bands'


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Create two horizontal semicircles that progressively overlap '
            'from one end.'
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
        help='Synthetic-data and pairing seed (default: 0).',
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


def camera_basis():
    camera_axis = direction_from_elevation_azimuth(
        VIEW_ELEVATION,
        VIEW_AZIMUTH,
    )
    horizontal_axis = np.cross(
        np.asarray([0.0, 0.0, 1.0]),
        camera_axis,
    )
    horizontal_axis /= np.linalg.norm(horizontal_axis)
    vertical_axis = np.cross(camera_axis, horizontal_axis)
    vertical_axis /= np.linalg.norm(vertical_axis)
    return camera_axis, horizontal_axis, vertical_axis


def arc_coordinates_to_unit_sphere(angles, vertical_coordinates):
    """Lift screen-horizontal small-circle coordinates to the unit sphere."""
    camera_axis, horizontal_axis, vertical_axis = camera_basis()
    horizontal_radii = np.sqrt(
        np.clip(1.0 - vertical_coordinates ** 2, 0.0, None)
    )
    return (
        (
            horizontal_radii * np.sin(angles)
        )[:, None] * horizontal_axis
        + vertical_coordinates[:, None] * vertical_axis
        + (
            horizontal_radii * np.cos(angles)
        )[:, None] * camera_axis
    )


def make_base_coordinates(seed):
    rng = np.random.default_rng(seed)
    arc_progress = np.linspace(0.0, 1.0, NUM_PAIRS)
    half_angle = np.deg2rad(ARC_HALF_ANGLE_DEGREES)
    base_angles = np.linspace(-half_angle, half_angle, NUM_PAIRS)
    midline_vertical = (
        ARC_BOW_HEIGHT
        * (1.0 - (2.0 * arc_progress - 1.0) ** 2)
    )

    image_angles = (
        base_angles
        + rng.normal(scale=BASE_ANGULAR_NOISE_STD, size=NUM_PAIRS)
    )
    audio_angles = (
        base_angles
        + rng.normal(scale=BASE_ANGULAR_NOISE_STD, size=NUM_PAIRS)
    )
    image_vertical = (
        midline_vertical
        + IMAGE_VERTICAL_CENTER
        + rng.normal(scale=BASE_VERTICAL_NOISE_STD, size=NUM_PAIRS)
    )
    audio_vertical = (
        midline_vertical
        + AUDIO_VERTICAL_CENTER
        + rng.normal(scale=BASE_VERTICAL_NOISE_STD, size=NUM_PAIRS)
    )

    return {
        'arc_progress': arc_progress,
        'midline_vertical': midline_vertical,
        'image_angles': image_angles,
        'audio_angles': audio_angles,
        'image_vertical': image_vertical,
        'audio_vertical': audio_vertical,
        'image_overlap_angular_noise': rng.normal(size=NUM_PAIRS),
        'audio_overlap_angular_noise': rng.normal(size=NUM_PAIRS),
        'image_overlap_vertical_noise': rng.normal(size=NUM_PAIRS),
        'audio_overlap_vertical_noise': rng.normal(size=NUM_PAIRS),
    }


def transform_coordinates(base_coordinates, overlap_fraction):
    """Transform an exact prefix of each arc and leave its suffix untouched."""
    transformed_count = int(round(overlap_fraction * NUM_PAIRS))
    transformed = np.zeros(NUM_PAIRS, dtype=bool)
    transformed[:transformed_count] = True

    image_angles = base_coordinates['image_angles'].copy()
    audio_angles = base_coordinates['audio_angles'].copy()
    image_vertical = base_coordinates['image_vertical'].copy()
    audio_vertical = base_coordinates['audio_vertical'].copy()

    image_angles[transformed] += (
        OVERLAP_ANGULAR_NOISE_STD
        * base_coordinates['image_overlap_angular_noise'][transformed]
    )
    audio_angles[transformed] += (
        OVERLAP_ANGULAR_NOISE_STD
        * base_coordinates['audio_overlap_angular_noise'][transformed]
    )
    image_vertical[transformed] = (
        base_coordinates['midline_vertical'][transformed]
        + OVERLAP_VERTICAL_NOISE_STD
        * base_coordinates['image_overlap_vertical_noise'][transformed]
    )
    audio_vertical[transformed] = (
        base_coordinates['midline_vertical'][transformed]
        + OVERLAP_VERTICAL_NOISE_STD
        * base_coordinates['audio_overlap_vertical_noise'][transformed]
    )

    image_emb = arc_coordinates_to_unit_sphere(
        image_angles,
        image_vertical,
    )
    audio_emb = arc_coordinates_to_unit_sphere(
        audio_angles,
        audio_vertical,
    )
    return image_emb, audio_emb, transformed


def make_arrangements(seed=0):
    base_coordinates = make_base_coordinates(seed)
    arrangements = []

    for overlap_fraction in OVERLAP_FRACTIONS:
        image_emb, unpaired_audio_emb, transformed = transform_coordinates(
            base_coordinates,
            overlap_fraction,
        )
        paired_audio_emb, optimal_cost, paired_cost = (
            mildly_suboptimal_pairing(image_emb, unpaired_audio_emb)
        )
        arrangements.append({
            'image_emb': image_emb,
            'audio_emb': paired_audio_emb,
            'canonical_audio_emb': unpaired_audio_emb,
            'transformed': transformed,
            'overlap_fraction': overlap_fraction,
            'optimal_pairing_cost': optimal_cost,
            'actual_pairing_cost': paired_cost,
        })

    return arrangements


def validate_progressive_overlap(arrangements, metric_values):
    if len(arrangements) != len(OVERLAP_FRACTIONS):
        raise ValueError(
            'Expected {} arrangements, got {}.'
            .format(len(OVERLAP_FRACTIONS), len(arrangements))
        )

    initial_image = arrangements[0]['image_emb']
    initial_audio = arrangements[0]['canonical_audio_emb']
    previous_transformed_count = -1

    for arrangement, expected_fraction in zip(
            arrangements, OVERLAP_FRACTIONS):
        image_emb = arrangement['image_emb']
        audio_emb = arrangement['audio_emb']
        canonical_audio_emb = arrangement['canonical_audio_emb']
        transformed = arrangement['transformed']

        if image_emb.shape != (NUM_PAIRS, EMBEDDING_DIMENSION):
            raise ValueError(
                'Unexpected image embedding shape: {}.'
                .format(image_emb.shape)
            )
        if audio_emb.shape != image_emb.shape:
            raise ValueError(
                'Image/audio shapes differ: {} and {}.'
                .format(image_emb.shape, audio_emb.shape)
            )
        np.testing.assert_allclose(
            np.linalg.norm(image_emb, axis=1),
            1.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.linalg.norm(audio_emb, axis=1),
            1.0,
            atol=1e-12,
        )

        expected_count = int(round(expected_fraction * NUM_PAIRS))
        if transformed.sum() != expected_count:
            raise ValueError(
                'Expected {} transformed pairs, got {}.'
                .format(expected_count, transformed.sum())
            )
        transformed_count = int(transformed.sum())
        if transformed_count <= previous_transformed_count:
            raise ValueError('Transformed section did not grow by stage.')
        previous_transformed_count = transformed_count

        untouched = ~transformed
        np.testing.assert_array_equal(
            image_emb[untouched],
            initial_image[untouched],
        )
        np.testing.assert_array_equal(
            canonical_audio_emb[untouched],
            initial_audio[untouched],
        )
        if not (
                arrangement['actual_pairing_cost']
                > arrangement['optimal_pairing_cost'] + 1e-12):
            raise ValueError('A stage pairing is not measurably suboptimal.')

    if (
            metric_values['nn_overlap_at_50'][-1]
            <= metric_values['nn_overlap_at_50'][0]):
        raise ValueError('NNO@50 did not increase after full overlap.')
    if (
            metric_values['minimum_cosine_distance'][-1]
            >= metric_values['minimum_cosine_distance'][0]):
        raise ValueError(
            'Minimum cosine distance did not decrease after full overlap.'
        )
    if (
            metric_values['modality_silhouette_score'][-1]
            >= metric_values['modality_silhouette_score'][0]):
        raise ValueError(
            'Silhouette score did not decrease after full overlap.'
        )


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    arrangements = make_arrangements(args.seed)
    metric_values = calculate_metrics(arrangements)
    validate_progressive_overlap(arrangements, metric_values)

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
        pair_line_width=0.55,
        pair_line_alpha=0.18,
    )

    for stage_name, arrangement in zip(STAGE_NAMES, arrangements):
        print(
            '{}: overlap={:.0%}, optimal paired distance={:.6f}, '
            'actual={:.6f}'
            .format(
                stage_name,
                arrangement['overlap_fraction'],
                arrangement['optimal_pairing_cost'],
                arrangement['actual_pairing_cost'],
            )
        )
    for metric_name, _metric_label, _metric_fn in METRICS:
        print(
            '{}: {}'.format(
                metric_name,
                ', '.join(
                    '{:.6f}'.format(value)
                    for value in metric_values[metric_name]
                ),
            )
        )
    print('Saved metrics to {}'.format(csv_path))
    print('Saved unit-sphere plot to {}'.format(figure_path))


if __name__ == '__main__':
    main()
