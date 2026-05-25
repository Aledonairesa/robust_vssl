import numpy as np


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


METRICS = {
    'centroid_distance': centroid_distance,
    'centroid_cosine_similarity': centroid_cosine_similarity,
    'mean_paired_cosine_similarity': mean_paired_cosine_similarity,
    'mean_unpaired_cosine_similarity': mean_unpaired_cosine_similarity,
    'global_separability': global_separability,
    'image_intra_spread': image_intra_spread,
    'audio_intra_spread': audio_intra_spread,
    'relative_modality_gap': relative_modality_gap,
}
