"""Plot opposite-pole embeddings with globally optimal positive pairing."""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from opposite_pole_clouds import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    calculate_metrics,
    make_opposite_pole_embeddings,
    plot_example,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot optimally paired opposite-pole embeddings.')
    parser.add_argument(
        '--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
        help='Directory receiving the output image.')
    parser.add_argument(
        '--dpi', type=int, default=150,
        help='Output image resolution (default: 150).')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Synthetic-data seed; use the same seed as the original plot.')
    return parser.parse_args()


def minimum_distance_pairing(image_emb, audio_emb):
    """Reorder audio embeddings to minimize total paired Euclidean distance."""
    squared_distances = np.sum(
        (image_emb[:, None, :] - audio_emb[None, :, :]) ** 2,
        axis=2,
    )
    image_indices, audio_indices = linear_sum_assignment(squared_distances)
    if not np.array_equal(image_indices, np.arange(image_emb.shape[0])):
        raise RuntimeError('Unexpected non-canonical image assignment order.')
    return audio_emb[audio_indices]


def main():
    args = parse_args()
    image_emb, audio_emb = make_opposite_pole_embeddings(args.seed)
    matched_audio_emb = minimum_distance_pairing(image_emb, audio_emb)
    metrics = calculate_metrics(image_emb, matched_audio_emb)
    output_path = args.output_dir / 'toy_opposite_poles_matched.png'
    plot_example(
        image_emb,
        matched_audio_emb,
        metrics,
        output_path,
        args.dpi,
        title='Toy modality gap: optimally matched opposite poles',
    )

    total_distance = np.linalg.norm(
        image_emb - matched_audio_emb, axis=1).sum()
    for name, value in metrics.items():
        print('{}: {:.6f}'.format(name, value))
    print('Minimum total paired Euclidean distance: {:.6f}'.format(
        total_distance))
    print('Saved toy example to {}'.format(output_path))


if __name__ == '__main__':
    main()
