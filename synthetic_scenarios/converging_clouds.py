"""Create synthetic experiment 01: east-west modality convergence.

Five arrangements place 500 image and 500 audio unit embeddings in
three dimensions. Their cap centers move symmetrically from opposite
east/west poles to complete overlap. Pairings are deliberately close to,
but not equal to, the globally optimal cosine-distance assignment.

Outputs: unit_spheres.png and metrics.csv in the output directory.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.embedding_metrics import (  # noqa: E402
    centroid_cosine_similarity,
    linear_separability_accuracy,
    mean_paired_cosine_similarity,
    minimum_cosine_distance,
    modality_silhouette_score,
    nn_overlap_at_50,
    relative_modality_gap,
)
from analysis.extra_plots.plot_toy_opposite_poles import (  # noqa: E402
    draw_unit_sphere,
)


NUM_PAIRS = 500
EMBEDDING_DIMENSION = 3
CAP_SPREAD = 0.28
MODALITY_NOISE_STD = 0.035
NUM_SUBOPTIMAL_SWAPS = 50
SUBOPTIMAL_SWAP_QUANTILE = 0.05
CONVERGENCE_FRACTIONS = np.linspace(0.0, 1.0, 5)
SEPARATION_DEGREES = (180, 135, 90, 45, 0)
STAGE_NAMES = (
    'stage_1_180_deg',
    'stage_2_135_deg',
    'stage_3_90_deg',
    'stage_4_45_deg',
    'stage_5_overlapping',
)
STAGE_TITLES = (
    '1. Opposite poles\n180° separation',
    '2. Moving together\n135° separation',
    '3. Halfway\n90° separation',
    '4. Nearly mixed\n45° separation',
    '5. Complete overlap\n0° separation',
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / 'outputs'
    / 'extra_plots'
    / 'synthetic_experiments'
    / '01_east_west_convergence'
)

METRICS = (
    (
        'centroid_cosine_similarity',
        'Centroid cosine similarity',
        centroid_cosine_similarity,
    ),
    (
        'relative_modality_gap',
        'Relative modality gap',
        relative_modality_gap,
    ),
    (
        'nn_overlap_at_50',
        'NNO@50',
        nn_overlap_at_50,
    ),
    (
        'linear_separability_accuracy',
        'Linear separability',
        linear_separability_accuracy,
    ),
    (
        'mean_paired_cosine_distance',
        'Mean paired cosine distance',
        lambda image_emb, audio_emb: (
            1.0 - mean_paired_cosine_similarity(image_emb, audio_emb)
        ),
    ),
    (
        'minimum_cosine_distance',
        'Minimum cosine distance',
        minimum_cosine_distance,
    ),
    (
        'modality_silhouette_score',
        'Modality silhouette score',
        modality_silhouette_score,
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create synthetic east-west modality convergence plots.',
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


def normalize_rows(embeddings):
    return embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True)


def rotation_about_y(angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ])


def make_local_caps(seed):
    """Return two statistically matched noisy caps around the north pole."""
    rng = np.random.default_rng(seed)
    semantic_tangent = rng.normal(
        scale=CAP_SPREAD,
        size=(NUM_PAIRS, EMBEDDING_DIMENSION - 1),
    )
    semantic_tangent = np.clip(semantic_tangent, -0.75, 0.75)
    image_tangent = semantic_tangent + rng.normal(
        scale=MODALITY_NOISE_STD,
        size=semantic_tangent.shape,
    )
    audio_tangent = semantic_tangent + rng.normal(
        scale=MODALITY_NOISE_STD,
        size=semantic_tangent.shape,
    )

    image_local = normalize_rows(np.column_stack((
        image_tangent,
        np.ones(NUM_PAIRS),
    )))
    audio_local = normalize_rows(np.column_stack((
        audio_tangent,
        np.ones(NUM_PAIRS),
    )))
    return image_local, audio_local


def mildly_suboptimal_pairing(image_emb, audio_emb):
    """Return a strong pairing made slightly worse than the global optimum."""
    cosine_distances = 1.0 - image_emb @ audio_emb.T
    image_indices, audio_indices = linear_sum_assignment(cosine_distances)
    if not np.array_equal(image_indices, np.arange(NUM_PAIRS)):
        raise RuntimeError('Unexpected non-canonical assignment order.')

    optimal_costs = cosine_distances[np.arange(NUM_PAIRS), audio_indices]
    left_indices, right_indices = np.triu_indices(NUM_PAIRS, k=1)
    swap_cost_increases = (
        cosine_distances[left_indices, audio_indices[right_indices]]
        + cosine_distances[right_indices, audio_indices[left_indices]]
        - optimal_costs[left_indices]
        - optimal_costs[right_indices]
    )

    # The globally optimal assignment makes all swap increases non-negative
    # up to roundoff. Start at the fifth percentile rather than taking
    # numerically negligible swaps, then select mildly damaging disjoint
    # swaps. This keeps pairings strong while making them clearly non-optimal.
    candidate_order = np.argsort(swap_cost_increases)
    candidate_start = int(
        SUBOPTIMAL_SWAP_QUANTILE * len(candidate_order))
    used_indices = np.zeros(NUM_PAIRS, dtype=bool)
    paired_audio_indices = audio_indices.copy()
    selected_increases = []
    for candidate_index in candidate_order[candidate_start:]:
        left = left_indices[candidate_index]
        right = right_indices[candidate_index]
        increase = swap_cost_increases[candidate_index]
        if increase <= 1e-10:
            continue
        if used_indices[left] or used_indices[right]:
            continue

        (
            paired_audio_indices[left],
            paired_audio_indices[right],
        ) = (
            paired_audio_indices[right],
            paired_audio_indices[left],
        )
        used_indices[left] = True
        used_indices[right] = True
        selected_increases.append(float(increase))
        if len(selected_increases) == NUM_SUBOPTIMAL_SWAPS:
            break

    if len(selected_increases) != NUM_SUBOPTIMAL_SWAPS:
        raise RuntimeError(
            'Could select only {} of {} requested suboptimal swaps.'
            .format(len(selected_increases), NUM_SUBOPTIMAL_SWAPS))

    paired_audio = audio_emb[paired_audio_indices]
    paired_cost = np.mean(1.0 - np.sum(image_emb * paired_audio, axis=1))
    optimal_cost = float(optimal_costs.mean())
    if paired_cost <= optimal_cost + 1e-12:
        raise RuntimeError('Pairing was not made measurably suboptimal.')
    return paired_audio, optimal_cost, float(paired_cost)


def make_arrangements(seed=0):
    image_local, audio_local = make_local_caps(seed)
    arrangements = []
    for fraction, separation_degrees in zip(
            CONVERGENCE_FRACTIONS, SEPARATION_DEGREES):
        half_separation = 0.5 * np.pi * (1.0 - fraction)
        image_emb = image_local @ rotation_about_y(-half_separation).T
        audio_emb = audio_local @ rotation_about_y(half_separation).T
        paired_audio, optimal_cost, paired_cost = mildly_suboptimal_pairing(
            image_emb, audio_emb)

        arrangements.append({
            'image_emb': image_emb,
            'audio_emb': paired_audio,
            'separation_degrees': separation_degrees,
            'optimal_pairing_cost': optimal_cost,
            'actual_pairing_cost': paired_cost,
        })
    return arrangements


def calculate_metrics(arrangements):
    metric_values = {
        metric_name: []
        for metric_name, _metric_label, _metric_fn in METRICS
    }
    for arrangement in arrangements:
        image_emb = arrangement['image_emb']
        audio_emb = arrangement['audio_emb']
        for metric_name, _metric_label, metric_fn in METRICS:
            metric_values[metric_name].append(
                float(metric_fn(image_emb, audio_emb)))
    return metric_values


def validate_arrangements(arrangements, metric_values):
    if len(arrangements) != 5:
        raise ValueError(
            'Expected five arrangements, got {}.'.format(
                len(arrangements)))
    for arrangement in arrangements:
        image_emb = arrangement['image_emb']
        audio_emb = arrangement['audio_emb']
        if image_emb.shape != (NUM_PAIRS, EMBEDDING_DIMENSION):
            raise ValueError(
                'Unexpected image embedding shape: {}.'
                .format(image_emb.shape))
        if audio_emb.shape != image_emb.shape:
            raise ValueError(
                'Image/audio embedding shapes differ: {} and {}.'
                .format(image_emb.shape, audio_emb.shape))
        np.testing.assert_allclose(
            np.linalg.norm(image_emb, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(
            np.linalg.norm(audio_emb, axis=1), 1.0, atol=1e-12)

    increasing_metrics = (
        'centroid_cosine_similarity',
        'nn_overlap_at_50',
    )
    decreasing_metrics = (
        'relative_modality_gap',
        'linear_separability_accuracy',
        'mean_paired_cosine_distance',
        'minimum_cosine_distance',
        'modality_silhouette_score',
    )
    for metric_name in increasing_metrics:
        differences = np.diff(metric_values[metric_name])
        if np.any(differences < -1e-6):
            raise ValueError(
                '{} is not monotonic increasing: {}.'
                .format(metric_name, metric_values[metric_name]))
    for metric_name in decreasing_metrics:
        differences = np.diff(metric_values[metric_name])
        if np.any(differences > 1e-6):
            raise ValueError(
                '{} is not monotonic decreasing: {}.'
                .format(metric_name, metric_values[metric_name]))


def write_metrics_csv(path, metric_values, stage_names=STAGE_NAMES):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['metric'] + list(stage_names)
    with open(path, 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for metric_name, _metric_label, _metric_fn in METRICS:
            row = {'metric': metric_name}
            row.update({
                stage_name: repr(metric_value)
                for stage_name, metric_value in zip(
                    stage_names, metric_values[metric_name])
            })
            writer.writerow(row)


def format_metric_text(metric_values, stage_index):
    return '\n'.join(
        '{}: {:.4f}'.format(
            metric_label,
            metric_values[metric_name][stage_index],
        )
        for metric_name, metric_label, _metric_fn in METRICS
    )


def plot_arrangements(
        arrangements,
        metric_values,
        path,
        dpi,
        stage_titles=STAGE_TITLES,
        figure_title=(
            'Synthetic experiment 01: '
            'east–west modalities progressively overlap'
        ),
        figure_subtitle=(
            '1,000 unit embeddings per arrangement '
            '(500 image + 500 audio, dimension 3)'
        ),
        view_elev=18,
        view_azim=-58,
        pair_line_width=0.45,
        pair_line_alpha=0.10):
    fig = plt.figure(figsize=(24, 8.8))
    grid = fig.add_gridspec(
        2,
        5,
        height_ratios=[11.0, 3.0],
        hspace=0.02,
        wspace=0.02,
    )

    for stage_index, (arrangement, stage_title) in enumerate(zip(
            arrangements, stage_titles)):
        ax = fig.add_subplot(grid[0, stage_index], projection='3d')
        metric_ax = fig.add_subplot(grid[1, stage_index])
        image_emb = arrangement['image_emb']
        audio_emb = arrangement['audio_emb']

        draw_unit_sphere(ax)
        for image_point, audio_point in zip(image_emb, audio_emb):
            ax.plot(
                [image_point[0], audio_point[0]],
                [image_point[1], audio_point[1]],
                [image_point[2], audio_point[2]],
                color='0.65',
                linewidth=pair_line_width,
                alpha=pair_line_alpha,
                zorder=1,
            )

        ax.scatter(
            image_emb[:, 0],
            image_emb[:, 1],
            image_emb[:, 2],
            s=10,
            c='#1f77b4',
            alpha=0.76,
            edgecolors='none',
            zorder=2,
        )
        ax.scatter(
            audio_emb[:, 0],
            audio_emb[:, 1],
            audio_emb[:, 2],
            s=10,
            c='#ff7f0e',
            alpha=0.76,
            edgecolors='none',
            zorder=3,
        )

        limit = 1.05
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=view_elev, azim=view_azim)
        ax.set_title(stage_title, fontsize=13, pad=8)
        ax.set_xlabel('Dimension 1', labelpad=-8)
        ax.set_ylabel('Dimension 2', labelpad=-8)
        ax.set_zlabel('Dimension 3', labelpad=-8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(True, alpha=0.18)

        metric_ax.axis('off')
        metric_ax.text(
            0.5,
            0.95,
            format_metric_text(metric_values, stage_index),
            ha='center',
            va='top',
            fontsize=8.2,
            linespacing=1.35,
            transform=metric_ax.transAxes,
        )

    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            markerfacecolor='#1f77b4',
            markeredgecolor='none',
            markersize=7,
            label='Image',
        ),
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            markerfacecolor='#ff7f0e',
            markeredgecolor='none',
            markersize=7,
            label='Audio',
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc='upper right',
        bbox_to_anchor=(0.985, 0.945),
        frameon=True,
        fontsize=11,
    )
    fig.suptitle(
        figure_title,
        fontsize=18,
        y=0.98,
    )
    fig.text(
        0.5,
        0.945,
        figure_subtitle,
        ha='center',
        va='center',
        fontsize=11,
        color='0.32',
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    arrangements = make_arrangements(args.seed)
    metric_values = calculate_metrics(arrangements)
    validate_arrangements(arrangements, metric_values)

    csv_path = output_dir / 'metrics.csv'
    figure_path = output_dir / 'unit_spheres.png'
    write_metrics_csv(csv_path, metric_values)
    plot_arrangements(arrangements, metric_values, figure_path, args.dpi)

    for stage_index, arrangement in enumerate(arrangements):
        print(
            '{}: optimal paired distance={:.6f}, actual={:.6f}'
            .format(
                STAGE_NAMES[stage_index],
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
