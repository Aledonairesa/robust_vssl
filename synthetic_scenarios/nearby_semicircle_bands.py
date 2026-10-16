"""Plot two nearby modality semicircles in one octant of the unit sphere."""

import argparse
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from opposite_pole_clouds import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    NUM_PAIRS,
    calculate_metrics,
    plot_example,
)


IMAGE_Y = 0.10
AUDIO_Y = 0.36
TRANSVERSE_NOISE_STD = 0.012
ANGULAR_NOISE_STD = 0.005


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot nearby semicircular modality distributions.')
    parser.add_argument(
        '--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
        help='Directory receiving the output image.')
    parser.add_argument(
        '--dpi', type=int, default=150,
        help='Output image resolution (default: 150).')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed used for data and pair shuffling (default: 0).')
    return parser.parse_args()


def _normalize_rows(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def make_nearby_semicircle_embeddings(seed=0):
    """Return shuffled pairs from two quarter-arcs in the positive octant."""
    rng = np.random.default_rng(seed)
    base_angles = np.linspace(0.0, 0.5 * np.pi, NUM_PAIRS)

    def make_arc(y_center):
        angles = np.clip(
            base_angles + rng.normal(
                scale=ANGULAR_NOISE_STD, size=NUM_PAIRS),
            1e-4,
            0.5 * np.pi - 1e-4,
        )
        y = np.clip(
            y_center + rng.normal(
                scale=TRANSVERSE_NOISE_STD, size=NUM_PAIRS),
            1e-4,
            1.0 - 1e-4,
        )
        horizontal_radius = np.sqrt(1.0 - y ** 2)
        return np.column_stack((
            horizontal_radius * np.cos(angles),
            y,
            horizontal_radius * np.sin(angles),
        ))

    image_emb = _normalize_rows(make_arc(IMAGE_Y))
    audio_emb = _normalize_rows(make_arc(AUDIO_Y))

    if np.any(image_emb <= 0.0) or np.any(audio_emb <= 0.0):
        raise RuntimeError('Synthetic embeddings left the positive octant.')

    # Shuffle only correspondence, leaving both point clouds unchanged.
    audio_emb = audio_emb[rng.permutation(NUM_PAIRS)]
    return image_emb, audio_emb


def main():
    args = parse_args()
    image_emb, audio_emb = make_nearby_semicircle_embeddings(args.seed)
    metrics = calculate_metrics(image_emb, audio_emb)
    output_path = args.output_dir / 'toy_nearby_semicircles.png'
    plot_example(
        image_emb,
        audio_emb,
        metrics,
        output_path,
        args.dpi,
        title='Toy modality gap: nearby semicircles',
        view_elev=25,
        view_azim=40,
    )

    for name, value in metrics.items():
        print('{}: {:.6f}'.format(name, value))
    print('Saved toy example to {}'.format(output_path))


if __name__ == '__main__':
    main()
