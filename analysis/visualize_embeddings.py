import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

try:
    from analyze_embeddings import load_embedding_file, parse_epoch
    from broad_class_labels import (
        UNKNOWN_LABEL,
        assign_broad_classes,
        print_broad_class_summary,
    )
except ImportError:
    from analysis.analyze_embeddings import load_embedding_file, parse_epoch
    from analysis.broad_class_labels import (
        UNKNOWN_LABEL,
        assign_broad_classes,
        print_broad_class_summary,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize saved image/audio embeddings.')
    parser.add_argument('--output_dir', default='outputs', type=str,
                        help='Root directory for experiment outputs')
    parser.add_argument('--exp_name', default=None, type=str,
                        help='Experiment name under output_dir')
    parser.add_argument('--embeddings_dir', default=None, type=str,
                        help='Direct path to an embeddings directory')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'],
                        choices=['val', 'test'],
                        help='Embedding splits to visualize')
    parser.add_argument('--visualizations', nargs='+',
                        default=['umap', 'cosine_similarity_hist'],
                        choices=['umap', 'class_umap',
                                 'cosine_similarity_hist'],
                        help='Visualization types to generate')
    parser.add_argument('--test_set', default=None, type=str,
                        help='Specific test set subdirectory to visualize')
    parser.add_argument('--val_epochs', nargs='+', type=int, default=None,
                        help='Validation epochs to visualize. Defaults to all.')
    parser.add_argument('--test_epoch', type=int, default=None,
                        help='Test epoch to visualize. Defaults to latest.')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='UMAP n_neighbors')
    parser.add_argument('--min_dist', type=float, default=0.1,
                        help='UMAP min_dist')
    parser.add_argument('--random_state', type=int, default=0,
                        help='UMAP random seed')
    parser.add_argument('--point_size', type=float, default=14,
                        help='Scatter point size')
    parser.add_argument('--draw_pairs', action='store_true',
                        help='Draw faint lines between matching pairs')
    parser.add_argument('--broad_classes_dir',
                        default='analysis/broad_classes', type=str,
                        help='Directory containing inferred broad-class JSONs')
    parser.add_argument('--gif_duration', type=int, default=600,
                        help='Validation GIF frame duration in milliseconds')
    return parser.parse_args()


def resolve_embeddings_dir(args):
    if args.embeddings_dir is not None:
        return Path(args.embeddings_dir)
    if args.exp_name is None:
        raise ValueError('Either --exp_name or --embeddings_dir must be set.')
    return Path(args.output_dir) / args.exp_name / 'embeddings'


def make_umap(args):
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            'UMAP visualizations require umap-learn. Install it in the '
            'active environment, e.g. `pip install umap-learn`.') from exc

    return UMAP(
        n_components=2,
        metric='cosine',
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
    )


def stack_modalities(image_emb, audio_emb):
    return np.concatenate([image_emb, audio_emb], axis=0)


def axis_limits(coords):
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_pad = max((x_max - x_min) * 0.05, 1e-3)
    y_pad = max((y_max - y_min) * 0.05, 1e-3)
    return (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)


def plot_modality_umap(coords, num_samples, title, output_path, point_size,
                       limits=None, draw_pairs=False):
    image_coords = coords[:num_samples]
    audio_coords = coords[num_samples:]

    fig, ax = plt.subplots(figsize=(7, 6))
    if draw_pairs:
        for idx in range(num_samples):
            ax.plot(
                [image_coords[idx, 0], audio_coords[idx, 0]],
                [image_coords[idx, 1], audio_coords[idx, 1]],
                color='0.75',
                linewidth=0.5,
                alpha=0.35,
                zorder=1,
            )

    ax.scatter(
        image_coords[:, 0],
        image_coords[:, 1],
        s=point_size,
        c='#1f77b4',
        alpha=0.8,
        label='image',
        edgecolors='none',
        zorder=2,
    )
    ax.scatter(
        audio_coords[:, 0],
        audio_coords[:, 1],
        s=point_size,
        c='#ff7f0e',
        alpha=0.8,
        label='audio',
        edgecolors='none',
        zorder=3,
    )

    if limits is not None:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
    ax.set_title(title)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print('Saved UMAP plot to {}'.format(output_path))


def broad_class_colors(labels):
    known_labels = sorted(set(labels) - {UNKNOWN_LABEL})
    if len(known_labels) <= 20:
        palette = list(plt.get_cmap('tab20').colors)
        colors = {
            label: palette[idx % len(palette)]
            for idx, label in enumerate(known_labels)
        }
    else:
        cmap = plt.get_cmap('hsv')
        colors = {
            label: cmap(idx / len(known_labels))
            for idx, label in enumerate(known_labels)
        }

    if UNKNOWN_LABEL in labels:
        colors[UNKNOWN_LABEL] = '#9e9e9e'
    return colors


def ordered_broad_class_labels(labels):
    labels = sorted(set(labels))
    if UNKNOWN_LABEL in labels:
        labels.remove(UNKNOWN_LABEL)
        labels.append(UNKNOWN_LABEL)
    return labels


def plot_broad_class_umap(coords, labels, title, output_path, point_size,
                          limits=None, draw_pairs=False):
    labels = np.asarray(labels)
    num_samples = len(labels)
    image_coords = coords[:num_samples]
    audio_coords = coords[num_samples:]
    colors = broad_class_colors(labels)
    class_point_size = point_size * 1.35

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    if draw_pairs:
        for idx in range(num_samples):
            ax.plot(
                [image_coords[idx, 0], audio_coords[idx, 0]],
                [image_coords[idx, 1], audio_coords[idx, 1]],
                color='0.75',
                linewidth=0.5,
                alpha=0.3,
                zorder=1,
            )

    for label in ordered_broad_class_labels(labels):
        mask = labels == label
        color = colors[label]
        ax.scatter(
            image_coords[mask, 0],
            image_coords[mask, 1],
            s=class_point_size,
            c=[color],
            marker='o',
            alpha=0.82,
            edgecolors='none',
            zorder=2,
        )
        ax.scatter(
            audio_coords[mask, 0],
            audio_coords[mask, 1],
            s=class_point_size,
            c=[color],
            marker='^',
            alpha=0.82,
            edgecolors='none',
            zorder=3,
        )

    if limits is not None:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
    ax.set_title(title)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.grid(True, alpha=0.2)

    class_handles = [
        Line2D(
            [0],
            [0],
            marker='o',
            color='none',
            markerfacecolor=colors[label],
            markeredgecolor='none',
            markersize=7,
            label=label,
        )
        for label in ordered_broad_class_labels(labels)
    ]
    modality_handles = [
        Line2D(
            [0], [0], marker='o', color='0.25', linestyle='none',
            markersize=7, label='image'),
        Line2D(
            [0], [0], marker='^', color='0.25', linestyle='none',
            markersize=7, label='audio'),
    ]

    class_legend = ax.legend(
        handles=class_handles,
        title='Class color',
        loc='upper left',
        bbox_to_anchor=(1.02, 0.78),
        fontsize=8,
    )
    ax.add_artist(class_legend)
    modality_legend = ax.legend(
        handles=modality_handles,
        title='Marker',
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches='tight',
        bbox_extra_artists=(class_legend, modality_legend),
    )
    plt.close(fig)
    print('Saved broad-class UMAP plot to {}'.format(output_path))


def select_epoch_files(files, requested_epochs=None):
    if requested_epochs is None:
        return files
    requested_epochs = set(requested_epochs)
    return [path for path in files if parse_epoch(path) in requested_epochs]


def select_single_epoch_file(files, requested_epoch=None):
    if not files:
        return None
    if requested_epoch is not None:
        for path in files:
            if parse_epoch(path) == requested_epoch:
                return path
        raise ValueError('Could not find test epoch {}'.format(requested_epoch))
    return max(files, key=parse_epoch)


def create_gif(frame_paths, output_path, duration):
    if not frame_paths:
        return
    frames = [Image.open(path).convert('RGB') for path in frame_paths]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    for frame in frames:
        frame.close()
    print('Saved GIF to {}'.format(output_path))


def paired_unpaired_cosine_similarity(image_emb, audio_emb):
    similarities = image_emb @ audio_emb.T
    paired = np.diag(similarities)
    mask = ~np.eye(similarities.shape[0], dtype=bool)
    unpaired = similarities[mask]
    return paired, unpaired


def histogram_y_limit(value_sets, bins):
    max_freq = 0
    for values in value_sets:
        counts, _ = np.histogram(values, bins=bins)
        if values.size > 0:
            counts = counts / values.size
        max_freq = max(max_freq, counts.max())
    return max_freq * 1.1 if max_freq > 0 else 1


def plot_cosine_similarity_histogram(paired, unpaired, title, output_path,
                                     bins, y_max=None):
    paired_mean = paired.mean()
    unpaired_mean = unpaired.mean()
    paired_q1 = np.percentile(paired, 25)
    unpaired_q3 = np.percentile(unpaired, 75)
    separability = paired_q1 - unpaired_q3

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(
        paired,
        bins=bins,
        weights=np.ones_like(paired) / paired.size,
        color='#1f77b4',
        alpha=0.62,
        edgecolor='white',
        linewidth=0.6,
        label='paired',
    )
    ax.hist(
        unpaired,
        bins=bins,
        weights=np.ones_like(unpaired) / unpaired.size,
        color='#ff7f0e',
        alpha=0.5,
        edgecolor='white',
        linewidth=0.6,
        label='unpaired',
    )
    ax.axvline(
        paired_mean,
        color='#1f77b4',
        linestyle='--',
        linewidth=1.6,
        label='paired mean {:.4f}'.format(paired_mean),
    )
    ax.axvline(
        unpaired_mean,
        color='#ff7f0e',
        linestyle='--',
        linewidth=1.6,
        label='unpaired mean {:.4f}'.format(unpaired_mean),
    )
    ax.axvline(
        paired_q1,
        color='#1f77b4',
        linestyle=':',
        linewidth=2.0,
        label='paired Q1 {:.4f}'.format(paired_q1),
    )
    ax.axvline(
        unpaired_q3,
        color='#ff7f0e',
        linestyle=':',
        linewidth=2.0,
        label='unpaired Q3 {:.4f}'.format(unpaired_q3),
    )
    ax.set_xlim(bins[0], bins[-1])
    if y_max is not None:
        ax.set_ylim(0, y_max)
    ax.set_xlabel('Cosine similarity')
    ax.set_ylabel('Relative frequency')
    ax.set_title('{} | sep={:.4f}'.format(title, separability))
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print('Saved cosine similarity histogram to {}'.format(output_path))


def load_val_files(embeddings_dir, args):
    val_dir = embeddings_dir / 'val'
    files = sorted(val_dir.glob('epoch_*.npz'), key=parse_epoch)
    files = select_epoch_files(files, args.val_epochs)
    if not files:
        print('No validation embedding files found in {}'.format(val_dir))
        return []

    loaded = []
    for path in files:
        names, image_emb, audio_emb = load_embedding_file(path)
        loaded.append((path, names, image_emb, audio_emb))
    return loaded


def visualize_val_umap(loaded, plots_dir, args):
    all_embeddings = []
    for path, names, image_emb, audio_emb in loaded:
        embeddings = stack_modalities(image_emb, audio_emb)
        all_embeddings.append(embeddings)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    reducer = make_umap(args)
    all_coords = reducer.fit_transform(all_embeddings)
    limits = axis_limits(all_coords)

    val_plot_dir = plots_dir / 'umap' / 'val'
    val_plot_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    offset = 0
    for path, names, image_emb, audio_emb in loaded:
        embeddings = stack_modalities(image_emb, audio_emb)
        epoch = parse_epoch(path)
        coords = all_coords[offset:offset + len(embeddings)]
        offset += len(embeddings)

        frame_path = val_plot_dir / 'epoch_{:04d}.png'.format(epoch)
        plot_modality_umap(
            coords,
            len(names),
            'Validation UMAP epoch {:04d}'.format(epoch),
            frame_path,
            args.point_size,
            limits=limits,
            draw_pairs=args.draw_pairs,
        )
        frame_paths.append(frame_path)

    gif_path = plots_dir / 'umap' / 'val_evolution.gif'
    create_gif(frame_paths, gif_path, args.gif_duration)


def visualize_val_cosine_similarity_hist(loaded, plots_dir, args):
    hist_plot_dir = plots_dir / 'cosine_similarity_hist' / 'val'
    hist_plot_dir.mkdir(parents=True, exist_ok=True)

    bins = np.linspace(-1, 1, 51)
    epoch_values = [
        (path, *paired_unpaired_cosine_similarity(image_emb, audio_emb))
        for path, names, image_emb, audio_emb in loaded
    ]
    y_max = histogram_y_limit(
        [values for path, paired, unpaired in epoch_values
         for values in (paired, unpaired)], bins)
    frame_paths = []
    for path, paired, unpaired in epoch_values:
        epoch = parse_epoch(path)
        frame_path = hist_plot_dir / 'epoch_{:04d}.png'.format(epoch)
        plot_cosine_similarity_histogram(
            paired,
            unpaired,
            'Validation cosine similarity epoch {:04d}'.format(epoch),
            frame_path,
            bins,
            y_max=y_max,
        )
        frame_paths.append(frame_path)

    gif_path = plots_dir / 'cosine_similarity_hist' / 'val_evolution.gif'
    create_gif(frame_paths, gif_path, args.gif_duration)


def visualize_val(embeddings_dir, plots_dir, args):
    loaded = load_val_files(embeddings_dir, args)
    if not loaded:
        return

    if 'umap' in args.visualizations:
        visualize_val_umap(loaded, plots_dir, args)
    if 'cosine_similarity_hist' in args.visualizations:
        visualize_val_cosine_similarity_hist(loaded, plots_dir, args)


def iter_test_dirs(test_root, test_set):
    if test_set is not None:
        yield test_set, test_root / test_set
        return

    for path in sorted(test_root.iterdir()):
        if path.is_dir():
            yield path.name, path

    direct_files = sorted(test_root.glob('epoch_*.npz'), key=parse_epoch)
    if direct_files:
        yield 'test', test_root


def visualize_test_umap(test_set, path, names, image_emb, audio_emb, plots_dir,
                        args):
    embeddings = stack_modalities(image_emb, audio_emb)
    reducer = make_umap(args)
    coords = reducer.fit_transform(embeddings)

    epoch = parse_epoch(path)
    plot_path = plots_dir / 'umap' / 'test_{}_epoch_{:04d}.png'.format(
        test_set, epoch)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_modality_umap(
        coords,
        len(names),
        'Test {} UMAP epoch {:04d}'.format(test_set, epoch),
        plot_path,
        args.point_size,
        limits=axis_limits(coords),
        draw_pairs=args.draw_pairs,
    )


def visualize_test_class_umap(test_set, path, names, image_emb, audio_emb,
                              plots_dir, args):
    epoch = parse_epoch(path)
    try:
        labels, diagnostics = assign_broad_classes(
            names,
            test_set,
            args.broad_classes_dir,
        )
    except FileNotFoundError as exc:
        print(exc)
        return

    print_broad_class_summary(test_set, epoch, diagnostics)

    embeddings = stack_modalities(image_emb, audio_emb)
    reducer = make_umap(args)
    coords = reducer.fit_transform(embeddings)

    plot_path = plots_dir / 'class_umap' / 'test_{}_epoch_{:04d}.png'.format(
        test_set, epoch)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_broad_class_umap(
        coords,
        labels,
        'Test {} broad-class UMAP epoch {:04d}'.format(test_set, epoch),
        plot_path,
        args.point_size,
        limits=axis_limits(coords),
        draw_pairs=args.draw_pairs,
    )


def visualize_test_cosine_similarity_hist(test_set, path, image_emb, audio_emb,
                                          plots_dir):
    epoch = parse_epoch(path)
    paired, unpaired = paired_unpaired_cosine_similarity(image_emb, audio_emb)
    bins = np.linspace(-1, 1, 51)
    plot_path = (
        plots_dir / 'cosine_similarity_hist' /
        'test_{}_epoch_{:04d}.png'.format(test_set, epoch)
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_cosine_similarity_histogram(
        paired,
        unpaired,
        'Test {} cosine similarity epoch {:04d}'.format(test_set, epoch),
        plot_path,
        bins,
    )


def visualize_test(embeddings_dir, plots_dir, args):
    test_root = embeddings_dir / 'test'
    if not test_root.exists():
        print('No test embeddings directory found in {}'.format(test_root))
        return

    for test_set, test_dir in iter_test_dirs(test_root, args.test_set):
        files = sorted(test_dir.glob('epoch_*.npz'), key=parse_epoch)
        path = select_single_epoch_file(files, args.test_epoch)
        if path is None:
            print('No test embedding files found in {}'.format(test_dir))
            continue

        names, image_emb, audio_emb = load_embedding_file(path)
        if 'umap' in args.visualizations:
            visualize_test_umap(
                test_set, path, names, image_emb, audio_emb, plots_dir, args)
        if 'class_umap' in args.visualizations:
            visualize_test_class_umap(
                test_set, path, names, image_emb, audio_emb, plots_dir, args)
        if 'cosine_similarity_hist' in args.visualizations:
            visualize_test_cosine_similarity_hist(
                test_set, path, image_emb, audio_emb, plots_dir)


def main():
    args = parse_args()
    embeddings_dir = resolve_embeddings_dir(args)
    plots_dir = embeddings_dir / 'analysis' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    if 'val' in args.splits:
        visualize_val(embeddings_dir, plots_dir, args)
    if 'test' in args.splits:
        visualize_test(embeddings_dir, plots_dir, args)


if __name__ == '__main__':
    main()
