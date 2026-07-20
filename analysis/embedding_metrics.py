import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit


def mean_joint_angular_displacement_deg(previous_image_emb,
                                        previous_audio_emb,
                                        current_image_emb,
                                        current_audio_emb,
                                        epoch_delta=1):
    """Return mean image/audio angular displacement in degrees per epoch."""
    arrays = {
        'previous image': np.asarray(previous_image_emb, dtype=np.float64),
        'previous audio': np.asarray(previous_audio_emb, dtype=np.float64),
        'current image': np.asarray(current_image_emb, dtype=np.float64),
        'current audio': np.asarray(current_audio_emb, dtype=np.float64),
    }
    shapes = {name: value.shape for name, value in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(
            'Angular displacement requires matching shapes, got {}'.format(
                shapes))
    if arrays['current image'].ndim != 2:
        raise ValueError(
            'Angular displacement expects 2D embedding arrays, got {}'.format(
                arrays['current image'].shape))
    if arrays['current image'].shape[0] == 0:
        raise ValueError('Angular displacement requires at least one sample.')
    if epoch_delta <= 0:
        raise ValueError(
            'epoch_delta must be positive, got {}'.format(epoch_delta))

    normalized = {}
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(
                '{} embeddings contain non-finite values.'.format(name))
        norms = np.linalg.norm(value, axis=1, keepdims=True)
        zero_indices = np.flatnonzero(norms[:, 0] == 0)
        if zero_indices.size:
            raise ValueError(
                '{} embeddings contain {} zero-norm rows; first index: {}'
                .format(name, zero_indices.size, zero_indices[0]))
        normalized[name] = value / norms

    image_cosines = np.sum(
        normalized['previous image'] * normalized['current image'], axis=1)
    audio_cosines = np.sum(
        normalized['previous audio'] * normalized['current audio'], axis=1)
    image_angles = np.arccos(np.clip(image_cosines, -1.0, 1.0))
    audio_angles = np.arccos(np.clip(audio_cosines, -1.0, 1.0))
    joint_mean_radians = 0.5 * (
        image_angles.mean() + audio_angles.mean())
    return float(np.degrees(joint_mean_radians) / epoch_delta)


def linear_separability_accuracy(image_emb, audio_emb, group_ids=None,
                                 test_size=0.2, random_state=0,
                                 regularization_c=1.0):
    """Return held-out accuracy of a linear modality classifier.

    The split is performed over paired-sample groups before constructing the
    binary image/audio classification dataset. This keeps both modalities of
    a pair, and any repeated samples sharing a group ID, entirely in either
    the training or test partition.
    """
    image_emb = np.asarray(image_emb)
    audio_emb = np.asarray(audio_emb)
    if image_emb.shape != audio_emb.shape:
        raise ValueError(
            'Linear separability requires matching embedding shapes, got '
            '{} and {}.'.format(image_emb.shape, audio_emb.shape))
    if image_emb.ndim != 2:
        raise ValueError(
            'Linear separability expects 2D embedding arrays, got {}.'
            .format(image_emb.shape))
    if image_emb.shape[0] < 2:
        raise ValueError(
            'Linear separability requires at least two paired samples.')
    if not np.isfinite(image_emb).all() or not np.isfinite(audio_emb).all():
        raise ValueError(
            'Linear separability embeddings contain non-finite values.')
    if not 0 < test_size < 1:
        raise ValueError(
            'test_size must be between 0 and 1, got {}.'.format(test_size))
    if regularization_c <= 0:
        raise ValueError(
            'regularization_c must be positive, got {}.'
            .format(regularization_c))

    num_pairs = image_emb.shape[0]
    if group_ids is None:
        groups = np.arange(num_pairs)
    else:
        groups = np.asarray(group_ids)
        if groups.ndim != 1 or len(groups) != num_pairs:
            raise ValueError(
                'Linear separability requires one group ID per pair, got '
                '{} IDs for {} pairs.'.format(groups.size, num_pairs))
    if len(np.unique(groups)) < 2:
        raise ValueError(
            'Linear separability requires at least two sample groups.')

    pair_indices = np.arange(num_pairs)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_indices, test_indices = next(
        splitter.split(pair_indices, groups=groups))

    train_embeddings = np.concatenate([
        image_emb[train_indices],
        audio_emb[train_indices],
    ], axis=0)
    train_labels = np.concatenate([
        np.zeros(len(train_indices), dtype=np.int64),
        np.ones(len(train_indices), dtype=np.int64),
    ])
    test_embeddings = np.concatenate([
        image_emb[test_indices],
        audio_emb[test_indices],
    ], axis=0)
    test_labels = np.concatenate([
        np.zeros(len(test_indices), dtype=np.int64),
        np.ones(len(test_indices), dtype=np.int64),
    ])

    classifier = LogisticRegression(
        C=regularization_c,
        solver='liblinear',
        max_iter=2000,
        random_state=random_state,
    )
    classifier.fit(train_embeddings, train_labels)
    return float(classifier.score(test_embeddings, test_labels))


def _centroids(image_emb, audio_emb):
    return image_emb.mean(axis=0), audio_emb.mean(axis=0)


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


def _cosine_dissimilarity_matrix(left_emb, right_emb):
    return (1 - (left_emb @ right_emb.T)) / 2


def _nearest_neighbor_overlap(image_emb, audio_emb, k):
    image_emb = np.asarray(image_emb)
    audio_emb = np.asarray(audio_emb)
    num_images = image_emb.shape[0]
    num_audio = audio_emb.shape[0]
    if num_images == 0 or num_audio == 0:
        return float('nan')

    combined = np.concatenate([image_emb, audio_emb], axis=0)
    num_total = combined.shape[0]
    if num_total < 2:
        return float('nan')

    effective_k = min(k, num_total - 1)
    if effective_k <= 0:
        return float('nan')

    modality_labels = np.concatenate([
        np.zeros(num_images, dtype=int),
        np.ones(num_audio, dtype=int),
    ])
    similarities = combined @ combined.T
    np.fill_diagonal(similarities, -np.inf)

    neighbor_indices = np.argpartition(
        -similarities,
        kth=effective_k - 1,
        axis=1,
    )[:, :effective_k]
    neighbor_labels = modality_labels[neighbor_indices]

    image_neighbor_overlap = np.mean(
        neighbor_labels[:num_images] == 1,
        axis=1,
    )
    audio_neighbor_overlap = np.mean(
        neighbor_labels[num_images:] == 0,
        axis=1,
    )
    return float(
        0.5 * (
            image_neighbor_overlap.mean()
            + audio_neighbor_overlap.mean()
        )
    )


def centroid_distance(image_emb, audio_emb):
    image_centroid, audio_centroid = _centroids(image_emb, audio_emb)
    return float(np.linalg.norm(image_centroid - audio_centroid))


def centroid_cosine_similarity(image_emb, audio_emb):
    image_centroid, audio_centroid = _centroids(image_emb, audio_emb)
    return _safe_cosine(image_centroid, audio_centroid)


def mean_paired_cosine_similarity(image_emb, audio_emb):
    return float(np.sum(image_emb * audio_emb, axis=1).mean())


def mean_unpaired_cosine_similarity(image_emb, audio_emb):
    similarities = image_emb @ audio_emb.T
    unpaired = _off_diagonal_values(similarities)
    if unpaired.size == 0:
        return float('nan')
    return float(unpaired.mean())


def global_separability(image_emb, audio_emb):
    similarities = image_emb @ audio_emb.T
    positive = np.diag(similarities)
    negative = _off_diagonal_values(similarities)
    if negative.size == 0:
        return float('nan')
    return float(np.percentile(positive, 25) - np.percentile(negative, 75))


def image_intra_spread(image_emb, audio_emb):
    distances = _off_diagonal_values(
        _cosine_dissimilarity_matrix(image_emb, image_emb))
    if distances.size == 0:
        return float('nan')
    return float(distances.mean())


def audio_intra_spread(image_emb, audio_emb):
    distances = _off_diagonal_values(
        _cosine_dissimilarity_matrix(audio_emb, audio_emb))
    if distances.size == 0:
        return float('nan')
    return float(distances.mean())


def relative_modality_gap(image_emb, audio_emb):
    cross_distances = _cosine_dissimilarity_matrix(image_emb, audio_emb)
    paired_gap = np.diag(cross_distances).mean()

    image_distances = _off_diagonal_values(
        _cosine_dissimilarity_matrix(image_emb, image_emb))
    audio_distances = _off_diagonal_values(
        _cosine_dissimilarity_matrix(audio_emb, audio_emb))
    if image_distances.size == 0 or audio_distances.size == 0:
        return float('nan')

    intra_spread = 0.5 * (image_distances.mean() + audio_distances.mean())
    denominator = intra_spread + paired_gap
    if denominator == 0:
        return float('nan')
    return float(paired_gap / denominator)


def nn_overlap_at_1(image_emb, audio_emb):
    return _nearest_neighbor_overlap(image_emb, audio_emb, 1)


def nn_overlap_at_5(image_emb, audio_emb):
    return _nearest_neighbor_overlap(image_emb, audio_emb, 5)


def nn_overlap_at_10(image_emb, audio_emb):
    return _nearest_neighbor_overlap(image_emb, audio_emb, 10)


def nn_overlap_at_50(image_emb, audio_emb):
    return _nearest_neighbor_overlap(image_emb, audio_emb, 50)


METRICS = {
    'centroid_distance': centroid_distance,
    'centroid_cosine_similarity': centroid_cosine_similarity,
    'mean_paired_cosine_similarity': mean_paired_cosine_similarity,
    'mean_unpaired_cosine_similarity': mean_unpaired_cosine_similarity,
    'global_separability': global_separability,
    'image_intra_spread': image_intra_spread,
    'audio_intra_spread': audio_intra_spread,
    'relative_modality_gap': relative_modality_gap,
    'nn_overlap_at_1': nn_overlap_at_1,
    'nn_overlap_at_5': nn_overlap_at_5,
    'nn_overlap_at_10': nn_overlap_at_10,
    'nn_overlap_at_50': nn_overlap_at_50,
}
