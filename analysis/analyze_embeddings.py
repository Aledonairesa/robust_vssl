import argparse
import csv
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    from embedding_metrics import METRICS
except ImportError:
    from analysis.embedding_metrics import METRICS


EPOCH_PATTERN = re.compile(r'epoch_(\d+)\.npz$')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze saved image/audio embedding files.')
    parser.add_argument('--output_dir', default='outputs', type=str,
                        help='Root directory for experiment outputs')
    parser.add_argument('--exp_name', default=None, type=str,
                        help='Experiment name under output_dir')
    parser.add_argument('--embeddings_dir', default=None, type=str,
                        help='Direct path to an embeddings directory')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'],
                        choices=['val', 'test'],
                        help='Embedding splits to analyze')
    parser.add_argument('--test_set', default=None, type=str,
                        help='Specific test set subdirectory to analyze')
    return parser.parse_args()


def resolve_embeddings_dir(args):
    if args.embeddings_dir is not None:
        return Path(args.embeddings_dir)
    if args.exp_name is None:
        raise ValueError('Either --exp_name or --embeddings_dir must be set.')
    return Path(args.output_dir) / args.exp_name / 'embeddings'


def parse_epoch(path):
    match = EPOCH_PATTERN.match(path.name)
    if match is None:
        raise ValueError('Could not parse epoch from {}'.format(path))
    return int(match.group(1))


def load_embedding_file(path):
    data = np.load(path)
    required_keys = ['names', 'image_emb', 'audio_emb']
    missing_keys = [key for key in required_keys if key not in data.files]
    if missing_keys:
        raise ValueError(
            '{} is missing required keys: {}'.format(path, missing_keys))

    names = data['names']
    image_emb = data['image_emb']
    audio_emb = data['audio_emb']

    if image_emb.shape != audio_emb.shape:
        raise ValueError(
            '{} has mismatched embedding shapes: {} vs {}'.format(
                path, image_emb.shape, audio_emb.shape))
    if image_emb.shape[0] != len(names):
        raise ValueError(
            '{} has {} names but {} embeddings'.format(
                path, len(names), image_emb.shape[0]))

    return names, image_emb, audio_emb


def compute_metrics(path):
    names, image_emb, audio_emb = load_embedding_file(path)
    row = {
        'epoch': parse_epoch(path),
        'num_samples': len(names),
        'embedding_file': str(path),
    }
    for metric_name, metric_fn in METRICS.items():
        row[metric_name] = metric_fn(image_emb, audio_emb)
    return row


def analyze_val(embeddings_dir, analysis_dir):
    val_dir = embeddings_dir / 'val'
    files = sorted(val_dir.glob('epoch_*.npz'))
    if not files:
        print('No validation embedding files found in {}'.format(val_dir))
        return None

    rows = sorted([compute_metrics(path) for path in files],
                  key=lambda row: row['epoch'])
    csv_path = analysis_dir / 'val_metrics.csv'
    write_csv(csv_path, rows)
    print('Saved validation metrics to {}'.format(csv_path))

    plot_val_metrics(rows, analysis_dir / 'plots')
    return rows


def iter_test_dirs(test_root, test_set):
    if test_set is not None:
        yield test_set, test_root / test_set
        return

    for path in sorted(test_root.iterdir()):
        if path.is_dir():
            yield path.name, path

    direct_files = sorted(test_root.glob('epoch_*.npz'))
    if direct_files:
        yield 'test', test_root


def analyze_test(embeddings_dir, analysis_dir, test_set):
    test_root = embeddings_dir / 'test'
    if not test_root.exists():
        print('No test embeddings directory found in {}'.format(test_root))
        return []

    outputs = []
    for current_test_set, test_dir in iter_test_dirs(test_root, test_set):
        files = sorted(test_dir.glob('epoch_*.npz'))
        if not files:
            print('No test embedding files found in {}'.format(test_dir))
            continue

        rows = []
        for path in files:
            row = compute_metrics(path)
            row['split'] = 'test'
            row['test_set'] = current_test_set
            rows.append(row)

        rows = sorted(rows, key=lambda row: row['epoch'])
        csv_path = analysis_dir / 'test_{}_metrics.csv'.format(
            current_test_set)
        write_csv(csv_path, rows)
        print('Saved test metrics to {}'.format(csv_path))
        outputs.append(rows)

    return outputs


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_val_metrics(rows, plots_dir):
    if not rows:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row['epoch'] for row in rows]

    for metric_name in METRICS.keys():
        values = [row[metric_name] for row in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, values, marker='o', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title('Validation {}'.format(metric_name))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = plots_dir / 'val_{}.png'.format(metric_name)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print('Saved validation plot to {}'.format(plot_path))


def main():
    args = parse_args()
    embeddings_dir = resolve_embeddings_dir(args)
    analysis_dir = embeddings_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if 'val' in args.splits:
        analyze_val(embeddings_dir, analysis_dir)
    if 'test' in args.splits:
        analyze_test(embeddings_dir, analysis_dir, args.test_set)


if __name__ == '__main__':
    main()
