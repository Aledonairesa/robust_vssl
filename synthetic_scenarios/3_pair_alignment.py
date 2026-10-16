"""Create synthetic experiment 03: improve only positive pairings.

All five arrangements use the exact image and audio positions from
``toy_nearby_semicircles``. The audio rows are permuted to progress from
completely random positive pairings to the globally optimal assignment.

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
METRICS = _shared_clouds.METRICS
plot_arrangements = _shared_clouds.plot_arrangements
validate_arrangements = _shared_clouds.validate_arrangements
write_metrics_csv = _shared_clouds.write_metrics_csv
from nearby_semicircle_bands import (  # noqa: E402
    make_nearby_semicircle_embeddings,
)
from optimal_pair_matching import (  # noqa: E402
    minimum_distance_pairing,
)


PAIRING_BLOCK_SIZES = (None, 250, 80, 20, 1)
STAGE_NAMES = (
    'stage_1_random',
    'stage_2_coarse',
    'stage_3_moderate',
    'stage_4_near_optimal',
    'stage_5_optimized',
)
STAGE_TITLES = (
    '1. Random pairing\nNo correspondence structure',
    '2. Coarse pairing\n250-point arc blocks',
    '3. Moderate pairing\n80-point arc blocks',
    '4. Near-optimal pairing\n20-point arc blocks',
    '5. Optimized pairing\nGlobal assignment',
)
FIGURE_TITLE = (
    'Synthetic experiment 03: fixed nearby semicircles, improving pairings'
)
FIGURE_SUBTITLE = (
    'Embedding positions are identical in every panel; '
    'only positive-pair connections change'
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / 'outputs' / 'pair_alignment'

PAIRING_SENSITIVE_METRICS = (
    'relative_modality_gap',
    'mean_paired_cosine_distance',
)
PAIRING_INVARIANT_METRICS = tuple(
    metric_name
    for metric_name, _metric_label, _metric_fn in METRICS
    if metric_name not in PAIRING_SENSITIVE_METRICS
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Create fixed nearby semicircles with improving pair quality.'
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


def shuffle_within_blocks(optimal_audio_emb, block_size, rng):
    """Randomly permute an optimal assignment within image-index blocks."""
    shuffled = optimal_audio_emb.copy()
    for block_start in range(0, len(shuffled), block_size):
        block_end = min(block_start + block_size, len(shuffled))
        block = optimal_audio_emb[block_start:block_end]
        shuffled[block_start:block_end] = block[
            rng.permutation(len(block))]
    return shuffled


def mean_paired_cosine_distance(image_emb, audio_emb):
    return float(
        1.0 - np.sum(image_emb * audio_emb, axis=1).mean())


def make_arrangements(seed=0):
    image_emb, random_audio_emb = make_nearby_semicircle_embeddings(seed)
    optimal_audio_emb = minimum_distance_pairing(
        image_emb, random_audio_emb)
    pairing_rng = np.random.default_rng(seed + 10_003)

    arrangements = []
    for block_size in PAIRING_BLOCK_SIZES:
        if block_size is None:
            paired_audio_emb = random_audio_emb.copy()
        elif block_size == 1:
            paired_audio_emb = optimal_audio_emb.copy()
        else:
            paired_audio_emb = shuffle_within_blocks(
                optimal_audio_emb,
                block_size,
                pairing_rng,
            )

        arrangements.append({
            'image_emb': image_emb.copy(),
            'audio_emb': paired_audio_emb,
            'pairing_block_size': block_size,
            'paired_cosine_distance': mean_paired_cosine_distance(
                image_emb, paired_audio_emb),
        })
    return arrangements


def canonical_row_order(embeddings):
    return np.lexsort((
        embeddings[:, 2],
        embeddings[:, 1],
        embeddings[:, 0],
    ))


def calculate_pairing_metrics(arrangements):
    """Keep position-only metrics exactly invariant across permutations."""
    metric_functions = {
        metric_name: metric_fn
        for metric_name, _metric_label, metric_fn in METRICS
    }
    canonical_image = arrangements[0]['image_emb']
    canonical_audio = arrangements[0]['audio_emb']
    metric_values = {}

    for metric_name, _metric_label, metric_fn in METRICS:
        if metric_name in PAIRING_SENSITIVE_METRICS:
            metric_values[metric_name] = [
                float(metric_fn(
                    arrangement['image_emb'],
                    arrangement['audio_emb'],
                ))
                for arrangement in arrangements
            ]
            continue

        if metric_name == 'linear_separability_accuracy':
            sorted_image = canonical_image[
                canonical_row_order(canonical_image)]
            sorted_audio = canonical_audio[
                canonical_row_order(canonical_audio)]
            fixed_value = float(metric_functions[metric_name](
                sorted_image, sorted_audio))
        else:
            fixed_value = float(metric_fn(
                canonical_image, canonical_audio))
        metric_values[metric_name] = [fixed_value] * len(arrangements)

    return metric_values


def sorted_rows(embeddings):
    return embeddings[canonical_row_order(embeddings)]


def validate_pairing_experiment(arrangements, metric_values):
    validate_arrangements(arrangements, metric_values)

    reference_image = sorted_rows(arrangements[0]['image_emb'])
    reference_audio = sorted_rows(arrangements[0]['audio_emb'])
    for arrangement in arrangements[1:]:
        np.testing.assert_array_equal(
            sorted_rows(arrangement['image_emb']),
            reference_image,
        )
        np.testing.assert_array_equal(
            sorted_rows(arrangement['audio_emb']),
            reference_audio,
        )

    for metric_name in PAIRING_INVARIANT_METRICS:
        if len(set(metric_values[metric_name])) != 1:
            raise ValueError(
                '{} changed even though positions were fixed: {}.'
                .format(metric_name, metric_values[metric_name]))

    for metric_name in PAIRING_SENSITIVE_METRICS:
        differences = np.diff(metric_values[metric_name])
        if not np.all(differences < 0.0):
            raise ValueError(
                '{} did not improve strictly at every stage: {}.'
                .format(metric_name, metric_values[metric_name]))

    final_cost = arrangements[-1]['paired_cosine_distance']
    optimal_audio = minimum_distance_pairing(
        arrangements[-1]['image_emb'],
        arrangements[-1]['audio_emb'],
    )
    optimal_cost = mean_paired_cosine_distance(
        arrangements[-1]['image_emb'],
        optimal_audio,
    )
    if not np.isclose(final_cost, optimal_cost, atol=1e-14, rtol=0.0):
        raise ValueError(
            'Final pairing cost {:.16f} is not globally optimal {:.16f}.'
            .format(final_cost, optimal_cost))


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    arrangements = make_arrangements(args.seed)
    metric_values = calculate_pairing_metrics(arrangements)
    validate_pairing_experiment(arrangements, metric_values)

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
        view_elev=25,
        view_azim=40,
        pair_line_width=0.60,
        pair_line_alpha=0.24,
    )

    for stage_name, arrangement in zip(STAGE_NAMES, arrangements):
        print(
            '{}: paired cosine distance={:.6f}'
            .format(
                stage_name,
                arrangement['paired_cosine_distance'],
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
