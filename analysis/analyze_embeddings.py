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
    from broad_class_labels import (
        assign_broad_classes,
        print_broad_class_summary,
    )
    from class_embedding_metrics import (
        compute_class_metrics,
        compute_topk_retrieval_rows,
        plot_class_metric_outputs,
    )
    from embedding_metrics import METRICS
except ImportError:
    from analysis.broad_class_labels import (
        assign_broad_classes,
        print_broad_class_summary,
    )
    from analysis.class_embedding_metrics import (
        compute_class_metrics,
        compute_topk_retrieval_rows,
        plot_class_metric_outputs,
    )
    from analysis.embedding_metrics import METRICS


EPOCH_PATTERN = re.compile(r'epoch_(\d+)\.npz$')
IMAGE_EMBEDDING_KEYS = {
    'positive_mask_mean': 'image_emb_positive_mask_mean',
    'maxpool': 'image_emb',
}
IMAGE_EMBEDDING_CHOICES = tuple(IMAGE_EMBEDDING_KEYS)


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
    parser.add_argument('--broad_classes_dir',
                        default='analysis/broad_classes', type=str,
                        help='Directory containing inferred broad-class JSONs')
    parser.add_argument('--class_retrieval_top_k', default=5, type=int,
                        help='Top-k used for class retrieval distributions')
    parser.add_argument('--image_embedding', default='positive_mask_mean',
                        choices=IMAGE_EMBEDDING_CHOICES,
                        help='Image embedding representation to analyze')
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


def load_embedding_file(path, image_embedding='positive_mask_mean'):
    if image_embedding not in IMAGE_EMBEDDING_KEYS:
        raise ValueError(
            'Unknown image embedding representation: {}'.format(
                image_embedding))

    image_embedding_key = IMAGE_EMBEDDING_KEYS[image_embedding]
    with np.load(path) as data:
        required_keys = ['names', image_embedding_key, 'audio_emb']
        missing_keys = [
            key for key in required_keys if key not in data.files
        ]
        if missing_keys:
            message = '{} is missing required keys: {}'.format(
                path, missing_keys)
            if (
                image_embedding == 'positive_mask_mean'
                and 'image_emb' in data.files
            ):
                message += (
                    '. This appears to be a legacy embedding file; use '
                    '--image_embedding maxpool.')
            raise ValueError(message)

        names = data['names']
        image_emb = data[image_embedding_key]
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


def compute_metrics(path, names=None, image_emb=None, audio_emb=None,
                    image_embedding='positive_mask_mean'):
    if names is None or image_emb is None or audio_emb is None:
        names, image_emb, audio_emb = load_embedding_file(
            path, image_embedding=image_embedding)
    row = {
        'epoch': parse_epoch(path),
        'num_samples': len(names),
        'embedding_file': str(path),
        'image_embedding': image_embedding,
    }
    for metric_name, metric_fn in METRICS.items():
        row[metric_name] = metric_fn(image_emb, audio_emb)
    return row


def analyze_val(embeddings_dir, analysis_dir, image_embedding):
    val_dir = embeddings_dir / 'val'
    files = sorted(val_dir.glob('epoch_*.npz'))
    if not files:
        print('No validation embedding files found in {}'.format(val_dir))
        return None

    rows = sorted([
        compute_metrics(path, image_embedding=image_embedding)
        for path in files
    ],
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


def analyze_test(embeddings_dir, analysis_dir, test_set, broad_classes_dir,
                 class_retrieval_top_k, image_embedding):
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
        class_rows = []
        topk_retrieval_rows = []
        for path in files:
            names, image_emb, audio_emb = load_embedding_file(
                path, image_embedding=image_embedding)
            epoch = parse_epoch(path)

            row = compute_metrics(
                path, names, image_emb, audio_emb,
                image_embedding=image_embedding)
            row['split'] = 'test'
            row['test_set'] = current_test_set
            rows.append(row)

            try:
                labels, diagnostics = assign_broad_classes(
                    names, current_test_set, broad_classes_dir)
            except FileNotFoundError as exc:
                print(exc)
                continue

            print_broad_class_summary(current_test_set, epoch, diagnostics)
            current_class_rows, value_sets = compute_class_metrics(
                current_test_set,
                epoch,
                path,
                image_emb,
                audio_emb,
                labels,
            )
            current_topk_retrieval_rows = compute_topk_retrieval_rows(
                current_test_set,
                epoch,
                path,
                image_emb,
                audio_emb,
                labels,
                class_retrieval_top_k,
            )
            class_rows.extend(current_class_rows)
            topk_retrieval_rows.extend(current_topk_retrieval_rows)
            plot_class_metric_outputs(
                current_class_rows,
                value_sets,
                current_topk_retrieval_rows,
                image_emb,
                audio_emb,
                labels,
                current_test_set,
                epoch,
                analysis_dir / 'plots',
            )

        rows = sorted(rows, key=lambda row: row['epoch'])
        csv_path = analysis_dir / 'test_{}_metrics.csv'.format(
            current_test_set)
        write_csv(csv_path, rows)
        print('Saved test metrics to {}'.format(csv_path))

        class_csv_path = analysis_dir / 'test_{}_class_metrics.csv'.format(
            current_test_set)
        write_csv(class_csv_path, class_rows)
        if class_rows:
            print('Saved test class metrics to {}'.format(class_csv_path))

        topk_retrieval_csv_path = (
            analysis_dir / 'test_{}_class_topk_retrieval.csv'
            .format(current_test_set)
        )
        write_csv(topk_retrieval_csv_path, topk_retrieval_rows)
        if topk_retrieval_rows:
            print(
                'Saved test class top-k retrieval to {}'
                .format(topk_retrieval_csv_path))
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
    analysis_dir = embeddings_dir / 'analysis' / args.image_embedding
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if 'val' in args.splits:
        analyze_val(embeddings_dir, analysis_dir, args.image_embedding)
    if 'test' in args.splits:
        analyze_test(
            embeddings_dir,
            analysis_dir,
            args.test_set,
            args.broad_classes_dir,
            args.class_retrieval_top_k,
            args.image_embedding,
        )


if __name__ == '__main__':
    main()
