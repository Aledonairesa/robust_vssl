"""Unsupervised semantic clustering metrics for multimodal embeddings."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure


JOINT_V_MEASURE = 'joint_clustering_v_measure'
JOINT_HOMOGENEITY = 'joint_clustering_homogeneity'
JOINT_COMPLETENESS = 'joint_clustering_completeness'
JOINT_NUM_CLASSES = 'joint_clustering_num_classes'
JOINT_NUM_SAMPLES = 'joint_clustering_num_samples'
IMAGE_V_MEASURE = 'image_clustering_v_measure'
AUDIO_V_MEASURE = 'audio_clustering_v_measure'

CLUSTERING_METRIC_NAMES = (
    JOINT_V_MEASURE,
    JOINT_HOMOGENEITY,
    JOINT_COMPLETENESS,
    JOINT_NUM_CLASSES,
    JOINT_NUM_SAMPLES,
    IMAGE_V_MEASURE,
    AUDIO_V_MEASURE,
)


def _validate_inputs(image_emb, audio_emb, labels):
    image_emb = np.asarray(image_emb, dtype=np.float64)
    audio_emb = np.asarray(audio_emb, dtype=np.float64)
    labels = np.asarray(labels)

    if image_emb.shape != audio_emb.shape:
        raise ValueError(
            'Semantic clustering requires matching embedding shapes, got '
            '{} and {}.'.format(image_emb.shape, audio_emb.shape))
    if image_emb.ndim != 2:
        raise ValueError(
            'Semantic clustering expects 2D embedding arrays, got {}.'
            .format(image_emb.shape))
    if labels.ndim != 1 or len(labels) != image_emb.shape[0]:
        raise ValueError(
            'Semantic clustering requires one label per embedding pair, got '
            '{} labels for {} pairs.'.format(labels.size, image_emb.shape[0]))
    return image_emb, audio_emb, labels


def _l2_normalize(embeddings, modality):
    if not np.isfinite(embeddings).all():
        raise ValueError(
            '{} embeddings contain non-finite values.'.format(modality))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    zero_indices = np.flatnonzero(norms[:, 0] == 0)
    if zero_indices.size:
        raise ValueError(
            '{} embeddings contain {} zero-norm rows; first index: {}.'
            .format(modality, zero_indices.size, zero_indices[0]))
    return embeddings / norms


def _fit_kmeans(embeddings, labels, num_clusters, n_init, random_state):
    assignments = KMeans(
        n_clusters=num_clusters,
        init='k-means++',
        n_init=n_init,
        random_state=random_state,
        algorithm='lloyd',
    ).fit_predict(embeddings)
    return homogeneity_completeness_v_measure(labels, assignments)


def semantic_clustering_metrics(image_emb, audio_emb, labels,
                                unknown_label='unknown', n_init=20,
                                random_state=0):
    """Cluster image/audio embeddings and compare clusters with semantics.

    Image and audio embeddings are treated as separate points in one shared
    space. Unknown semantic labels are excluded before L2 normalization and
    clustering. ``joint_clustering_num_samples`` is the number of individual
    embedding points used by joint K-means, so it is twice the number of
    retained image/audio pairs.
    """
    image_emb, audio_emb, labels = _validate_inputs(
        image_emb, audio_emb, labels)
    if not isinstance(n_init, (int, np.integer)) or n_init <= 0:
        raise ValueError('n_init must be a positive integer, got {}.'.format(
            n_init))

    known_mask = labels != unknown_label
    known_labels = labels[known_mask]
    num_classes = int(len(np.unique(known_labels)))
    num_joint_samples = int(2 * len(known_labels))
    if num_classes == 0:
        return {
            JOINT_V_MEASURE: float('nan'),
            JOINT_HOMOGENEITY: float('nan'),
            JOINT_COMPLETENESS: float('nan'),
            JOINT_NUM_CLASSES: 0,
            JOINT_NUM_SAMPLES: 0,
            IMAGE_V_MEASURE: float('nan'),
            AUDIO_V_MEASURE: float('nan'),
        }

    known_image = _l2_normalize(image_emb[known_mask], 'Image')
    known_audio = _l2_normalize(audio_emb[known_mask], 'Audio')
    joint_embeddings = np.concatenate([known_image, known_audio], axis=0)
    joint_labels = np.concatenate([known_labels, known_labels], axis=0)

    joint_homogeneity, joint_completeness, joint_v_measure = _fit_kmeans(
        joint_embeddings,
        joint_labels,
        num_classes,
        n_init,
        random_state,
    )
    _, _, image_v_measure = _fit_kmeans(
        known_image,
        known_labels,
        num_classes,
        n_init,
        random_state,
    )
    _, _, audio_v_measure = _fit_kmeans(
        known_audio,
        known_labels,
        num_classes,
        n_init,
        random_state,
    )

    return {
        JOINT_V_MEASURE: float(joint_v_measure),
        JOINT_HOMOGENEITY: float(joint_homogeneity),
        JOINT_COMPLETENESS: float(joint_completeness),
        JOINT_NUM_CLASSES: num_classes,
        JOINT_NUM_SAMPLES: num_joint_samples,
        IMAGE_V_MEASURE: float(image_v_measure),
        AUDIO_V_MEASURE: float(audio_v_measure),
    }
