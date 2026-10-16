"""Plot image and audio embeddings clustered around opposite sphere poles."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from embedding_metrics import (  # noqa: E402
    nn_overlap_at_10,
    relative_modality_gap,
)


NUM_PAIRS = 500
CLUSTER_SPREAD = 0.22
PAIR_NOISE_STD = 0.035
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / 'outputs'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot a toy opposite-poles modality-gap example.')
    parser.add_argument(
        '--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
        help='Directory receiving the output image.')
    parser.add_argument(
        '--dpi', type=int, default=150,
        help='Output image resolution (default: 150).')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed used for synthetic noise (default: 0).')
    return parser.parse_args()


def _normalize_rows(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def make_opposite_pole_embeddings(seed=0):
    """Return paired unit embeddings clustered around north and south poles."""
    rng = np.random.default_rng(seed)
    tangent = rng.normal(scale=CLUSTER_SPREAD, size=(NUM_PAIRS, 2))
    image_tangent = tangent + rng.normal(
        scale=PAIR_NOISE_STD, size=tangent.shape)
    audio_tangent = -tangent + rng.normal(
        scale=PAIR_NOISE_STD, size=tangent.shape)

    image_emb = np.column_stack((image_tangent, np.ones(NUM_PAIRS)))
    audio_emb = np.column_stack((audio_tangent, -np.ones(NUM_PAIRS)))
    return _normalize_rows(image_emb), _normalize_rows(audio_emb)


def calculate_metrics(image_emb, audio_emb):
    return {
        'Relative modality gap': relative_modality_gap(image_emb, audio_emb),
        'Nearest-neighbor overlap @ 10': nn_overlap_at_10(
            image_emb, audio_emb),
    }


def draw_unit_sphere(ax):
    """Draw a fine lavender wireframe marking the unit sphere."""
    azimuth = np.linspace(0.0, 2.0 * np.pi, 121)
    polar = np.linspace(0.0, np.pi, 61)
    x = np.outer(np.cos(azimuth), np.sin(polar))
    y = np.outer(np.sin(azimuth), np.sin(polar))
    z = np.outer(np.ones_like(azimuth), np.cos(polar))
    ax.plot_wireframe(
        x, y, z,
        rstride=3, cstride=3,
        color='#8F86B3', linewidth=0.25, alpha=0.30,
        zorder=0,
    )


def plot_example(image_emb, audio_emb, metrics, output_path, dpi,
                 title='Toy modality gap: embeddings at opposite poles',
                 view_elev=18, view_azim=-58):
    fig = plt.figure(figsize=(8, 9.0))
    grid = fig.add_gridspec(2, 1, height_ratios=[11, 1.5], hspace=0.18)
    ax = fig.add_subplot(grid[0], projection='3d')
    metric_ax = fig.add_subplot(grid[1])

    draw_unit_sphere(ax)

    for image_point, audio_point in zip(image_emb, audio_emb):
        ax.plot(
            [image_point[0], audio_point[0]],
            [image_point[1], audio_point[1]],
            [image_point[2], audio_point[2]],
            color='0.65', linewidth=0.60, alpha=0.16, zorder=1,
        )

    ax.scatter(
        image_emb[:, 0], image_emb[:, 1], image_emb[:, 2],
        s=12, c='#1f77b4', alpha=0.8, label='Image',
        edgecolors='none', zorder=2,
    )
    ax.scatter(
        audio_emb[:, 0], audio_emb[:, 1], audio_emb[:, 2],
        s=12, c='#ff7f0e', alpha=0.8, label='Audio',
        edgecolors='none', zorder=3,
    )

    limit = 1.05
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_title(title)
    ax.set_xlabel('Embedding dimension 1')
    ax.set_ylabel('Embedding dimension 2')
    ax.set_zlabel('Embedding dimension 3')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.18)

    metric_text = '\n'.join(
        '{}: {:.4f}'.format(name, value)
        for name, value in metrics.items()
    )
    metric_ax.axis('off')
    metric_ax.text(
        0.5, 0.55, metric_text,
        ha='center', va='center', fontsize=10, linespacing=1.5,
        transform=metric_ax.transAxes,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    image_emb, audio_emb = make_opposite_pole_embeddings(args.seed)
    metrics = calculate_metrics(image_emb, audio_emb)
    output_path = args.output_dir / 'toy_opposite_poles.png'
    plot_example(image_emb, audio_emb, metrics, output_path, args.dpi)

    for name, value in metrics.items():
        print('{}: {:.6f}'.format(name, value))
    print('Saved toy example to {}'.format(output_path))


if __name__ == '__main__':
    main()
