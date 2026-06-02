from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    from broad_class_labels import UNKNOWN_LABEL
except ImportError:
    from analysis.broad_class_labels import UNKNOWN_LABEL


def _safe_mean(values):
    if values.size == 0:
        return float('nan')
    return float(values.mean())


def _safe_percentile(values, percentile):
    if values.size == 0:
        return float('nan')
    return float(np.percentile(values, percentile))


def _safe_cosine(x, y):
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return float('nan')
    return float(np.dot(x, y) / denom)


def _off_diagonal_values(matrix):
    if matrix.shape[0] < 2:
        return np.asarray([], dtype=matrix.dtype)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def _cosine_distance_values(similarities):
    return (1 - similarities) / 2


def _cosine_distance_matrix(left_emb, right_emb):
    return _cosine_distance_values(left_emb @ right_emb.T)


def _normalize_rows(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = np.full(embeddings.shape, np.nan, dtype=float)
    return np.divide(embeddings, norms, out=normalized, where=norms != 0)


def _class_names(labels):
    return sorted(set(labels) - {UNKNOWN_LABEL})


def _class_centroid_cosine_similarity_matrix(image_emb, audio_emb, labels):
    labels = np.asarray(labels)
    classes = _class_names(labels)
    if not classes:
        return classes, np.empty((0, 0), dtype=float)

    image_centroids = np.stack([
        image_emb[labels == class_name].mean(axis=0)
        for class_name in classes
    ])
    audio_centroids = np.stack([
        audio_emb[labels == class_name].mean(axis=0)
        for class_name in classes
    ])
    image_centroids = _normalize_rows(image_centroids)
    audio_centroids = _normalize_rows(audio_centroids)
    similarities = image_centroids @ audio_centroids.T
    return classes, np.clip(similarities, -1, 1)


def _relative_modality_gap(paired_distances, image_distances,
                           audio_distances):
    if image_distances.size == 0 or audio_distances.size == 0:
        return float('nan')

    paired_gap = paired_distances.mean()
    intra_spread = 0.5 * (image_distances.mean() + audio_distances.mean())
    denominator = intra_spread + paired_gap
    if denominator == 0:
        return float('nan')
    return float(paired_gap / denominator)


def compute_class_metrics(test_set, epoch, embedding_file, image_emb,
                          audio_emb, labels):
    labels = np.asarray(labels)
    rows = []
    value_sets = {}

    for class_name in _class_names(labels):
        mask = labels == class_name
        class_image_emb = image_emb[mask]
        class_audio_emb = audio_emb[mask]

        cross_similarity = class_image_emb @ class_audio_emb.T
        paired_similarities = np.diag(cross_similarity)
        unpaired_similarities = _off_diagonal_values(cross_similarity)
        paired_distances = _cosine_distance_values(paired_similarities)
        unpaired_distances = _cosine_distance_values(unpaired_similarities)
        image_distances = _off_diagonal_values(
            _cosine_distance_matrix(class_image_emb, class_image_emb))
        audio_distances = _off_diagonal_values(
            _cosine_distance_matrix(class_audio_emb, class_audio_emb))

        image_centroid = class_image_emb.mean(axis=0)
        audio_centroid = class_audio_emb.mean(axis=0)
        paired_q1 = _safe_percentile(paired_similarities, 25)
        unpaired_q3 = _safe_percentile(unpaired_similarities, 75)

        row = {
            'epoch': epoch,
            'split': 'test',
            'test_set': test_set,
            'broad_class': class_name,
            'num_samples': int(mask.sum()),
            'embedding_file': str(embedding_file),
            'mean_paired_cosine_similarity': _safe_mean(
                paired_similarities),
            'paired_cosine_similarity_q1': paired_q1,
            'paired_cosine_similarity_q3': _safe_percentile(
                paired_similarities, 75),
            'mean_paired_cosine_distance': _safe_mean(paired_distances),
            'paired_cosine_distance_q1': _safe_percentile(
                paired_distances, 25),
            'paired_cosine_distance_q3': _safe_percentile(
                paired_distances, 75),
            'mean_same_class_unpaired_cosine_similarity': _safe_mean(
                unpaired_similarities),
            'same_class_unpaired_cosine_similarity_q1': _safe_percentile(
                unpaired_similarities, 25),
            'same_class_unpaired_cosine_similarity_q3': unpaired_q3,
            'mean_same_class_unpaired_cosine_distance': _safe_mean(
                unpaired_distances),
            'same_class_unpaired_cosine_distance_q1': _safe_percentile(
                unpaired_distances, 25),
            'same_class_unpaired_cosine_distance_q3': _safe_percentile(
                unpaired_distances, 75),
            'global_separability': (
                paired_q1 - unpaired_q3
                if np.isfinite(paired_q1) and np.isfinite(unpaired_q3)
                else float('nan')),
            'image_intra_spread': _safe_mean(image_distances),
            'audio_intra_spread': _safe_mean(audio_distances),
            'centroid_distance': float(
                np.linalg.norm(image_centroid - audio_centroid)),
            'centroid_cosine_similarity': _safe_cosine(
                image_centroid, audio_centroid),
            'relative_modality_gap': _relative_modality_gap(
                paired_distances, image_distances, audio_distances),
        }
        rows.append(row)
        value_sets[class_name] = {
            'paired_distances': paired_distances,
            'unpaired_distances': unpaired_distances,
        }

    return rows, value_sets


def _topk_distribution_matrix(query_emb, target_emb, query_labels,
                              target_labels, classes, top_k):
    similarities = query_emb @ target_emb.T
    ranking = np.argsort(-similarities, axis=1)
    effective_top_k = min(top_k, target_emb.shape[0])
    ranked_target_labels = target_labels[ranking[:, :effective_top_k]]
    matrix = np.zeros((len(classes), len(classes)), dtype=float)
    mean_counts = np.zeros((len(classes), len(classes)), dtype=float)
    query_counts = np.zeros(len(classes), dtype=int)

    for query_class_idx, query_class in enumerate(classes):
        query_indices = np.where(query_labels == query_class)[0]
        if query_indices.size == 0:
            matrix[query_class_idx, :] = np.nan
            mean_counts[query_class_idx, :] = np.nan
            continue
        query_counts[query_class_idx] = query_indices.size

        for target_class_idx, target_class in enumerate(classes):
            count = np.sum(
                ranked_target_labels[query_indices] == target_class)
            mean_counts[query_class_idx, target_class_idx] = (
                count / query_indices.size)
            matrix[query_class_idx, target_class_idx] = (
                count / (query_indices.size * effective_top_k))

    return matrix, mean_counts, query_counts, effective_top_k


def compute_topk_retrieval_rows(test_set, epoch, embedding_file, image_emb,
                                audio_emb, labels, top_k):
    labels = np.asarray(labels)
    known_mask = labels != UNKNOWN_LABEL
    known_labels = labels[known_mask]
    classes = _class_names(known_labels)
    if not classes:
        return []

    image_known = image_emb[known_mask]
    audio_known = audio_emb[known_mask]
    i2a, i2a_counts, i2a_query_counts, i2a_effective_top_k = (
        _topk_distribution_matrix(
            image_known, audio_known, known_labels, known_labels, classes,
            top_k))
    a2i, a2i_counts, a2i_query_counts, a2i_effective_top_k = (
        _topk_distribution_matrix(
            audio_known, image_known, known_labels, known_labels, classes,
            top_k))

    rows = []
    direction_values = [
        ('image_to_audio', i2a, i2a_counts, i2a_query_counts,
         i2a_effective_top_k),
        ('audio_to_image', a2i, a2i_counts, a2i_query_counts,
         a2i_effective_top_k),
    ]
    for direction, matrix, mean_counts, query_counts, effective_top_k in (
            direction_values):
        for query_idx, query_class in enumerate(classes):
            for target_idx, target_class in enumerate(classes):
                rows.append({
                    'epoch': epoch,
                    'split': 'test',
                    'test_set': test_set,
                    'direction': direction,
                    'query_class': query_class,
                    'retrieved_class': target_class,
                    'top_k': int(top_k),
                    'effective_top_k': int(effective_top_k),
                    'retrieved_fraction': float(
                        matrix[query_idx, target_idx]),
                    'mean_retrieved_count': float(
                        mean_counts[query_idx, target_idx]),
                    'num_query_samples': int(query_counts[query_idx]),
                    'num_known_samples': int(known_mask.sum()),
                    'embedding_file': str(embedding_file),
                })

    return rows


def _plot_width(num_classes):
    return max(8, min(18, 0.85 * num_classes + 4))


def _sorted_rows(rows, metric_name, descending=False):
    finite_rows = [
        row for row in rows if np.isfinite(float(row[metric_name]))
    ]
    return sorted(
        finite_rows,
        key=lambda row: float(row[metric_name]),
        reverse=descending,
    )


def _save_figure(fig, output_path, message):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('{} {}'.format(message, output_path))


def _sample_values(values, max_points=400):
    if values.size <= max_points:
        return np.sort(values)
    indices = np.linspace(0, values.size - 1, max_points).astype(int)
    return np.sort(values)[indices]


def plot_class_distance_lollipop(rows, value_sets, test_set, epoch, plots_dir):
    rows = _sorted_rows(rows, 'mean_paired_cosine_distance',
                        descending=False)
    if not rows:
        return

    classes = [row['broad_class'] for row in rows]
    y_positions = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(_plot_width(len(classes)), 5.5))

    for y_pos, row in zip(y_positions, rows):
        class_name = row['broad_class']
        paired_mean = float(row['mean_paired_cosine_distance'])
        unpaired_mean = float(
            row['mean_same_class_unpaired_cosine_distance'])

        paired_values = _sample_values(
            value_sets[class_name]['paired_distances'])
        unpaired_values = _sample_values(
            value_sets[class_name]['unpaired_distances'])

        if paired_values.size:
            ax.scatter(
                paired_values,
                np.full(paired_values.shape, y_pos - 0.15),
                s=10,
                color='#1f77b4',
                alpha=0.24,
                edgecolors='none',
            )
        if unpaired_values.size:
            ax.scatter(
                unpaired_values,
                np.full(unpaired_values.shape, y_pos + 0.15),
                s=10,
                color='#ff7f0e',
                alpha=0.18,
                edgecolors='none',
            )

        paired_q1 = row['paired_cosine_distance_q1']
        paired_q3 = row['paired_cosine_distance_q3']
        unpaired_q1 = row['same_class_unpaired_cosine_distance_q1']
        unpaired_q3 = row['same_class_unpaired_cosine_distance_q3']

        if np.isfinite(paired_q1) and np.isfinite(paired_q3):
            ax.hlines(
                y_pos - 0.15,
                paired_q1,
                paired_q3,
                color='#1f77b4',
                linewidth=3,
            )
        if np.isfinite(unpaired_q1) and np.isfinite(unpaired_q3):
            ax.hlines(
                y_pos + 0.15,
                unpaired_q1,
                unpaired_q3,
                color='#ff7f0e',
                linewidth=3,
            )

        if np.isfinite(paired_mean):
            ax.scatter(
                paired_mean,
                y_pos - 0.15,
                s=44,
                color='#1f77b4',
                label='paired mean' if y_pos == 0 else None,
                zorder=3,
            )
        if np.isfinite(unpaired_mean):
            ax.scatter(
                unpaired_mean,
                y_pos + 0.15,
                s=52,
                marker='^',
                color='#ff7f0e',
                label='same-class unpaired mean' if y_pos == 0 else None,
                zorder=3,
            )
        if np.isfinite(paired_mean) and np.isfinite(unpaired_mean):
            ax.plot(
                [paired_mean, unpaired_mean],
                [y_pos, y_pos],
                color='0.5',
                linewidth=1.0,
                alpha=0.55,
                zorder=1,
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Cosine distance ((1 - cosine) / 2)')
    ax.set_title(
        'Test {} paired vs same-class unpaired distances epoch {:04d}'
        .format(test_set, epoch))
    ax.grid(True, axis='x', alpha=0.25)
    ax.legend(loc='best', fontsize=8)

    output_path = (
        Path(plots_dir) / 'class_distance_lollipop' /
        'test_{}_epoch_{:04d}.png'.format(test_set, epoch)
    )
    _save_figure(fig, output_path, 'Saved class distance lollipop plot to')


def plot_class_separability(rows, test_set, epoch, plots_dir):
    rows = _sorted_rows(rows, 'global_separability', descending=True)
    if not rows:
        return

    classes = [row['broad_class'] for row in rows]
    values = [float(row['global_separability']) for row in rows]
    fig, ax = plt.subplots(figsize=(_plot_width(len(classes)), 4.8))
    ax.bar(classes, values, color='#4c78a8')
    ax.axhline(0, color='0.25', linewidth=0.8)
    ax.set_ylabel('Paired Q1 - same-class unpaired Q3')
    ax.set_title(
        'Test {} class separability epoch {:04d}'.format(test_set, epoch))
    ax.tick_params(axis='x', rotation=35)
    ax.grid(True, axis='y', alpha=0.25)

    output_path = (
        Path(plots_dir) / 'class_separability' /
        'test_{}_epoch_{:04d}.png'.format(test_set, epoch)
    )
    _save_figure(fig, output_path, 'Saved class separability plot to')


def plot_class_intra_spread(rows, test_set, epoch, plots_dir):
    finite_rows = [
        row for row in rows
        if (
            np.isfinite(float(row['image_intra_spread'])) or
            np.isfinite(float(row['audio_intra_spread']))
        )
    ]
    if not finite_rows:
        return

    def sort_key(row):
        values = [
            float(row['image_intra_spread']),
            float(row['audio_intra_spread']),
        ]
        values = [value for value in values if np.isfinite(value)]
        return np.mean(values) if values else float('inf')

    rows = sorted(finite_rows, key=sort_key)
    classes = [row['broad_class'] for row in rows]
    image_values = [float(row['image_intra_spread']) for row in rows]
    audio_values = [float(row['audio_intra_spread']) for row in rows]
    x = np.arange(len(classes))
    width = 0.38

    fig, ax = plt.subplots(figsize=(_plot_width(len(classes)), 4.8))
    ax.bar(x - width / 2, image_values, width, label='image',
           color='#1f77b4')
    ax.bar(x + width / 2, audio_values, width, label='audio',
           color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=35, ha='right')
    ax.set_ylabel('Mean within-class cosine distance')
    ax.set_title(
        'Test {} image/audio intra spread epoch {:04d}'.format(
            test_set, epoch))
    ax.legend(loc='best')
    ax.grid(True, axis='y', alpha=0.25)

    output_path = (
        Path(plots_dir) / 'class_intra_spread' /
        'test_{}_epoch_{:04d}.png'.format(test_set, epoch)
    )
    _save_figure(fig, output_path, 'Saved class intra-spread plot to')


def plot_class_centroid_alignment(rows, test_set, epoch, plots_dir):
    rows = _sorted_rows(rows, 'centroid_cosine_similarity', descending=True)
    if not rows:
        return

    classes = [row['broad_class'] for row in rows]
    values = [float(row['centroid_cosine_similarity']) for row in rows]
    fig, ax = plt.subplots(figsize=(_plot_width(len(classes)), 4.8))
    ax.bar(classes, values, color='#59a14f')
    ax.axhline(0, color='0.25', linewidth=0.8)
    ax.set_ylabel('Image/audio centroid cosine similarity')
    ax.set_title(
        'Test {} class centroid alignment epoch {:04d}'.format(
            test_set, epoch))
    ax.tick_params(axis='x', rotation=35)
    ax.grid(True, axis='y', alpha=0.25)

    output_path = (
        Path(plots_dir) / 'class_centroid_alignment' /
        'test_{}_epoch_{:04d}.png'.format(test_set, epoch)
    )
    _save_figure(fig, output_path, 'Saved class centroid alignment plot to')


def plot_class_centroid_cosine_similarity_heatmap(image_emb, audio_emb, labels,
                                                  test_set, epoch, plots_dir):
    classes, matrix = _class_centroid_cosine_similarity_matrix(
        image_emb, audio_emb, labels)
    if not classes:
        return

    fig, ax = plt.subplots(
        figsize=(_plot_width(len(classes)), max(5.5, 0.6 * len(classes) + 2)))
    im = ax.imshow(matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=40, ha='right')
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel('Audio centroid class')
    ax.set_ylabel('Image centroid class')
    ax.set_title(
        'Test {} image/audio class centroid cosine similarity epoch {:04d}'
        .format(test_set, epoch))

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(
                    col_idx,
                    row_idx,
                    '{:.2f}'.format(value),
                    ha='center',
                    va='center',
                    fontsize=7,
                    color='white' if abs(value) > 0.6 else 'black',
                )

    fig.colorbar(im, ax=ax, label='Cosine similarity')
    output_path = (
        Path(plots_dir) / 'class_centroid_cosine_similarity_heatmap' /
        'test_{}_epoch_{:04d}.png'.format(test_set, epoch)
    )
    _save_figure(
        fig, output_path, 'Saved class centroid cosine-similarity heatmap to')


def _retrieval_matrix(rows, direction, classes):
    matrix = np.full((len(classes), len(classes)), np.nan, dtype=float)
    for row in rows:
        if row['direction'] != direction:
            continue
        query_idx = classes.index(row['query_class'])
        target_idx = classes.index(row['retrieved_class'])
        matrix[query_idx, target_idx] = float(row['retrieved_fraction'])
    return matrix


def _draw_topk_heatmap(ax, matrix, classes, title):
    im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=40, ha='right')
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel('Retrieved target class')
    ax.set_ylabel('Query class')

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(
                    col_idx,
                    row_idx,
                    '{:.0%}'.format(value),
                    ha='center',
                    va='center',
                    fontsize=7,
                    color='white' if value > 0.45 else 'black',
                )
    return im


def plot_class_topk_retrieval_distribution(retrieval_rows, test_set, epoch,
                                           plots_dir):
    classes = sorted(set(row['query_class'] for row in retrieval_rows))
    if not classes:
        return
    top_k = retrieval_rows[0]['effective_top_k']

    i2a = _retrieval_matrix(retrieval_rows, 'image_to_audio', classes)
    a2i = _retrieval_matrix(retrieval_rows, 'audio_to_image', classes)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(12, _plot_width(len(classes)) * 1.5), 5.6),
        constrained_layout=True,
    )
    im = _draw_topk_heatmap(
        axes[0],
        i2a,
        classes,
        'Image -> audio top-{} class distribution'.format(top_k),
    )
    _draw_topk_heatmap(
        axes[1],
        a2i,
        classes,
        'Audio -> image top-{} class distribution'.format(top_k),
    )
    fig.suptitle(
        'Test {} top-{} retrieval distribution by broad class epoch {:04d}'
        .format(test_set, top_k, epoch))
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82,
                 label='Fraction of top-{} retrieved items'.format(top_k))

    output_path = (
        Path(plots_dir) / 'class_topk_retrieval_distribution' /
        'test_{}_epoch_{:04d}.png'.format(test_set, epoch)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved class top-k retrieval plot to {}'.format(output_path))


def plot_class_relative_modality_gap(rows, test_set, epoch, plots_dir):
    rows = _sorted_rows(rows, 'relative_modality_gap', descending=False)
    if not rows:
        return

    classes = [row['broad_class'] for row in rows]
    values = [float(row['relative_modality_gap']) for row in rows]
    fig, ax = plt.subplots(figsize=(_plot_width(len(classes)), 4.8))
    ax.bar(classes, values, color='#b279a2')
    ax.set_ylabel('Relative modality gap')
    ax.set_title(
        'Test {} relative modality gap epoch {:04d}'.format(
            test_set, epoch))
    ax.tick_params(axis='x', rotation=35)
    ax.grid(True, axis='y', alpha=0.25)

    output_path = (
        Path(plots_dir) / 'class_relative_modality_gap' /
        'test_{}_epoch_{:04d}.png'.format(test_set, epoch)
    )
    _save_figure(fig, output_path, 'Saved class modality-gap plot to')


def plot_class_metric_outputs(rows, value_sets, retrieval_rows, image_emb,
                              audio_emb, labels, test_set, epoch, plots_dir):
    plot_class_distance_lollipop(rows, value_sets, test_set, epoch, plots_dir)
    plot_class_separability(rows, test_set, epoch, plots_dir)
    plot_class_intra_spread(rows, test_set, epoch, plots_dir)
    plot_class_centroid_alignment(rows, test_set, epoch, plots_dir)
    plot_class_centroid_cosine_similarity_heatmap(
        image_emb, audio_emb, labels, test_set, epoch, plots_dir)
    plot_class_topk_retrieval_distribution(
        retrieval_rows, test_set, epoch, plots_dir)
    plot_class_relative_modality_gap(rows, test_set, epoch, plots_dir)
