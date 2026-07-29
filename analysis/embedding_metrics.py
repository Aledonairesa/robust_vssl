import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupShuffleSplit


CLASS_GAP_COHERENCE = 'class_gap_coherence'


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


def minimum_cosine_distance(image_emb, audio_emb, block_size=1024):
    """Return symmetric cross-modal nearest-neighbor cosine distance.

    For every image embedding, this computes the cosine distance to its
    nearest audio embedding, and vice versa. The returned scalar gives equal
    weight to the two directional means:

        0.5 * (mean_i(1 - max_j cos(image_i, audio_j))
               + mean_j(1 - max_i cos(image_i, audio_j)))

    Inputs are assumed to be L2-normalized; this function deliberately does
    not normalize them. Similarities are computed in two-dimensional blocks
    so every sample is used without materializing the full cross-modal
    similarity matrix.
    """
    image_emb = np.asarray(image_emb, dtype=np.float64)
    audio_emb = np.asarray(audio_emb, dtype=np.float64)
    if image_emb.ndim != 2 or audio_emb.ndim != 2:
        raise ValueError(
            'Minimum cosine distance expects 2D embedding arrays, got '
            '{} and {}.'.format(image_emb.shape, audio_emb.shape))
    if image_emb.shape[1] != audio_emb.shape[1]:
        raise ValueError(
            'Minimum cosine distance requires matching embedding '
            'dimensions, got {} and {}.'.format(
                image_emb.shape[1], audio_emb.shape[1]))
    if (
            image_emb.shape[0] == 0
            or audio_emb.shape[0] == 0
            or image_emb.shape[1] == 0):
        raise ValueError(
            'Minimum cosine distance requires non-empty embeddings.')
    if not np.isfinite(image_emb).all() or not np.isfinite(audio_emb).all():
        raise ValueError(
            'Minimum cosine distance embeddings contain non-finite values.')
    if (
            isinstance(block_size, (bool, np.bool_))
            or not isinstance(block_size, (int, np.integer))
            or block_size <= 0):
        raise ValueError(
            'block_size must be a positive integer, got {!r}.'.format(
                block_size))

    num_images = image_emb.shape[0]
    num_audio = audio_emb.shape[0]
    image_max_similarities = np.full(num_images, -np.inf)
    audio_max_similarities = np.full(num_audio, -np.inf)

    for image_start in range(0, num_images, block_size):
        image_end = min(image_start + block_size, num_images)
        image_block = image_emb[image_start:image_end]
        block_image_maxima = np.full(image_end - image_start, -np.inf)

        for audio_start in range(0, num_audio, block_size):
            audio_end = min(audio_start + block_size, num_audio)
            similarities = (
                image_block @ audio_emb[audio_start:audio_end].T)
            block_image_maxima = np.maximum(
                block_image_maxima,
                similarities.max(axis=1),
            )
            audio_max_similarities[audio_start:audio_end] = np.maximum(
                audio_max_similarities[audio_start:audio_end],
                similarities.max(axis=0),
            )

        image_max_similarities[image_start:image_end] = block_image_maxima

    # Unit-normalized vectors have cosine similarities in [-1, 1]. Clipping
    # removes only floating-point excursions outside that theoretical range.
    image_max_similarities = np.clip(
        image_max_similarities, -1.0, 1.0)
    audio_max_similarities = np.clip(
        audio_max_similarities, -1.0, 1.0)
    image_to_audio = 1.0 - image_max_similarities
    audio_to_image = 1.0 - audio_max_similarities
    return float(
        0.5 * (image_to_audio.mean() + audio_to_image.mean()))


def modality_silhouette_score(image_emb, audio_emb):
    """Return the Euclidean silhouette score of the modality labels.

    Image and audio embeddings are stacked into one sample set and assigned
    their respective modality labels. A high score means the modalities form
    compact, separated clusters; a score near zero means that modality does
    not explain the embedding geometry well. Negative scores are retained
    because they indicate that samples are closer, on average, to the other
    modality than to their own.

    Inputs are assumed to be L2-normalized; this function deliberately does
    not normalize or subsample them.
    """
    image_emb = np.asarray(image_emb, dtype=np.float64)
    audio_emb = np.asarray(audio_emb, dtype=np.float64)
    if image_emb.shape != audio_emb.shape:
        raise ValueError(
            'Modality silhouette score requires matching embedding shapes, '
            'got {} and {}.'.format(image_emb.shape, audio_emb.shape))
    if image_emb.ndim != 2:
        raise ValueError(
            'Modality silhouette score expects 2D embedding arrays, got {}.'
            .format(image_emb.shape))
    if image_emb.shape[0] < 2 or image_emb.shape[1] == 0:
        raise ValueError(
            'Modality silhouette score requires at least two paired samples '
            'and one embedding dimension.')
    if not np.isfinite(image_emb).all() or not np.isfinite(audio_emb).all():
        raise ValueError(
            'Modality silhouette score embeddings contain non-finite values.')

    joint_emb = np.concatenate([image_emb, audio_emb], axis=0)
    modality_labels = np.concatenate([
        np.zeros(image_emb.shape[0], dtype=np.int64),
        np.ones(audio_emb.shape[0], dtype=np.int64),
    ])
    return float(silhouette_score(
        joint_emb,
        modality_labels,
        metric='euclidean',
    ))


def gaussian_wasserstein_uniformity(image_emb, audio_emb):
    """Return joint Gaussian Wasserstein uniformity (higher is better).

    The image and audio embeddings are treated as samples from one shared
    embedding distribution. Inputs are assumed to be L2-normalized; this
    function deliberately does not normalize them.

    The metric is the negative quadratic Wasserstein distance between the
    Gaussian fitted to the joint empirical distribution and N(0, I / d).
    Its maximum is zero.
    """
    image_emb = np.asarray(image_emb, dtype=np.float64)
    audio_emb = np.asarray(audio_emb, dtype=np.float64)
    if image_emb.shape != audio_emb.shape:
        raise ValueError(
            'Gaussian Wasserstein uniformity requires matching embedding '
            'shapes, got {} and {}.'.format(
                image_emb.shape, audio_emb.shape))
    if image_emb.ndim != 2:
        raise ValueError(
            'Gaussian Wasserstein uniformity expects 2D embedding arrays, '
            'got {}.'.format(image_emb.shape))
    if image_emb.shape[0] == 0 or image_emb.shape[1] == 0:
        raise ValueError(
            'Gaussian Wasserstein uniformity requires non-empty embeddings.')
    if not np.isfinite(image_emb).all() or not np.isfinite(audio_emb).all():
        raise ValueError(
            'Gaussian Wasserstein uniformity embeddings contain non-finite '
            'values.')

    joint_emb = np.concatenate([image_emb, audio_emb], axis=0)
    sample_mean = joint_emb.mean(axis=0)
    centered = joint_emb - sample_mean

    # This is the covariance of the empirical distribution, whose samples
    # each have mass 1 / N. The 1 / N divisor also makes the metric invariant
    # to cloning the complete sample set.
    covariance = centered.T @ centered / joint_emb.shape[0]
    covariance = 0.5 * (covariance + covariance.T)

    # Covariance is positive semidefinite. Clip negative eigenvalues caused
    # only by floating-point roundoff before taking their square roots.
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    # This is algebraically equivalent to
    #
    # ||mean||^2 + 1 + tr(covariance)
    #     - (2 / sqrt(d)) * tr(sqrt(covariance)),
    #
    # but avoids cancellation near the ideal covariance I / d.
    target_standard_deviation = 1.0 / np.sqrt(joint_emb.shape[1])
    wasserstein_squared = (
        np.dot(sample_mean, sample_mean)
        + np.square(
            np.sqrt(eigenvalues) - target_standard_deviation).sum()
    )
    return -float(np.sqrt(wasserstein_squared))


def class_gap_coherence(image_emb, audio_emb, labels,
                        unknown_label='unknown'):
    """Return the coherence of broad-class modality-gap vectors.

    Each known broad class receives equal weight. For every class, image and
    audio embeddings are averaged separately and the two centroid directions
    are L2-normalized. If ``g_c`` is the audio-minus-image displacement between
    those directions for class ``c``, the score is

        ||mean_c(g_c)||^2 / mean_c(||g_c||^2).

    The score is in [0, 1] up to floating-point roundoff. A value near one
    means that class gaps share a common displacement, whereas a value near
    zero means that their directions cancel or vary strongly by class.

    Sample embeddings are not normalized internally. Unknown samples are
    excluded. The metric is undefined and returns NaN when fewer than two
    known classes remain, a class centroid has zero norm, or every class gap
    vanishes.
    """
    image_emb = np.asarray(image_emb, dtype=np.float64)
    audio_emb = np.asarray(audio_emb, dtype=np.float64)
    labels = np.asarray(labels)
    if image_emb.shape != audio_emb.shape:
        raise ValueError(
            'Class-gap coherence requires matching embedding shapes, got '
            '{} and {}.'.format(image_emb.shape, audio_emb.shape))
    if image_emb.ndim != 2:
        raise ValueError(
            'Class-gap coherence expects 2D embedding arrays, got {}.'
            .format(image_emb.shape))
    if labels.ndim != 1 or len(labels) != image_emb.shape[0]:
        raise ValueError(
            'Class-gap coherence requires one label per embedding pair, got '
            '{} labels for {} pairs.'.format(labels.size, image_emb.shape[0]))

    known_mask = labels != unknown_label
    if (
            not np.isfinite(image_emb[known_mask]).all()
            or not np.isfinite(audio_emb[known_mask]).all()):
        raise ValueError(
            'Class-gap coherence embeddings contain non-finite values.')

    known_labels = labels[known_mask]
    classes = np.unique(known_labels)
    if len(classes) < 2:
        return float('nan')

    known_image = image_emb[known_mask]
    known_audio = audio_emb[known_mask]
    image_centroids = np.stack([
        known_image[known_labels == class_name].mean(axis=0)
        for class_name in classes
    ])
    audio_centroids = np.stack([
        known_audio[known_labels == class_name].mean(axis=0)
        for class_name in classes
    ])
    image_norms = np.linalg.norm(
        image_centroids, axis=1, keepdims=True)
    audio_norms = np.linalg.norm(
        audio_centroids, axis=1, keepdims=True)
    if (
            np.any(image_norms[:, 0] == 0.0)
            or np.any(audio_norms[:, 0] == 0.0)):
        return float('nan')

    gap_vectors = (
        audio_centroids / audio_norms
        - image_centroids / image_norms
    )
    mean_squared_gap = float(
        np.mean(np.sum(gap_vectors * gap_vectors, axis=1)))
    if mean_squared_gap == 0.0:
        return float('nan')

    mean_gap = gap_vectors.mean(axis=0)
    coherence = float(np.dot(mean_gap, mean_gap) / mean_squared_gap)
    return float(np.clip(coherence, 0.0, 1.0))


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
    'minimum_cosine_distance': minimum_cosine_distance,
    'modality_silhouette_score': modality_silhouette_score,
    'gaussian_wasserstein_uniformity': gaussian_wasserstein_uniformity,
    'global_separability': global_separability,
    'image_intra_spread': image_intra_spread,
    'audio_intra_spread': audio_intra_spread,
    'relative_modality_gap': relative_modality_gap,
    'nn_overlap_at_1': nn_overlap_at_1,
    'nn_overlap_at_5': nn_overlap_at_5,
    'nn_overlap_at_10': nn_overlap_at_10,
    'nn_overlap_at_50': nn_overlap_at_50,
}
