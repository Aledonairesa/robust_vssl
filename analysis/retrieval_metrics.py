import numpy as np

try:
    from broad_class_labels import UNKNOWN_LABEL
except ImportError:
    from analysis.broad_class_labels import UNKNOWN_LABEL


PRECISION_K = (5, 10)
ACCURACY_K = (1, 5, 10)


def _empty_direction_metrics(direction, precision_k, accuracy_k):
    metrics = {}
    for k in precision_k:
        metrics['semantic_{}_precision_at_{}'.format(direction, k)] = (
            float('nan'))
    for k in accuracy_k:
        metrics['semantic_{}_accuracy_at_{}'.format(direction, k)] = (
            float('nan'))
    return metrics


def _direction_metrics(query_emb, target_emb, query_labels, target_labels,
                       direction, precision_k, accuracy_k,
                       exclude_self=False):
    query_emb = np.asarray(query_emb)
    target_emb = np.asarray(target_emb)
    query_labels = np.asarray(query_labels)
    target_labels = np.asarray(target_labels)

    metric_values = {
        'precision': {k: [] for k in precision_k},
        'accuracy': {k: [] for k in accuracy_k},
    }
    k_values = sorted(set(precision_k) | set(accuracy_k))

    query_indices = np.where(query_labels != UNKNOWN_LABEL)[0]
    target_indices = np.where(target_labels != UNKNOWN_LABEL)[0]
    if query_indices.size == 0 or target_indices.size == 0:
        return _empty_direction_metrics(direction, precision_k, accuracy_k)

    similarities = query_emb @ target_emb.T
    for query_idx in query_indices:
        candidate_indices = target_indices
        if exclude_self:
            candidate_indices = candidate_indices[candidate_indices != query_idx]
        if candidate_indices.size == 0:
            continue

        query_class = query_labels[query_idx]
        candidate_labels = target_labels[candidate_indices]
        if not np.any(candidate_labels == query_class):
            continue

        scores = similarities[query_idx, candidate_indices]
        ranking = np.argsort(-scores)
        ranked_labels = candidate_labels[ranking]

        for k in k_values:
            effective_k = min(k, ranked_labels.size)
            top_labels = ranked_labels[:effective_k]
            same_class_count = np.sum(top_labels == query_class)
            if k in metric_values['precision']:
                metric_values['precision'][k].append(
                    same_class_count / effective_k)
            if k in metric_values['accuracy']:
                metric_values['accuracy'][k].append(
                    float(same_class_count > 0))

    metrics = {}
    for k in precision_k:
        values = metric_values['precision'][k]
        metrics['semantic_{}_precision_at_{}'.format(direction, k)] = (
            float(np.mean(values)) if values else float('nan'))
    for k in accuracy_k:
        values = metric_values['accuracy'][k]
        metrics['semantic_{}_accuracy_at_{}'.format(direction, k)] = (
            float(np.mean(values)) if values else float('nan'))
    return metrics


def compute_semantic_retrieval_metrics(image_emb, audio_emb, labels,
                                       precision_k=PRECISION_K,
                                       accuracy_k=ACCURACY_K):
    labels = np.asarray(labels)
    metrics = {}
    metrics.update(_direction_metrics(
        image_emb,
        audio_emb,
        labels,
        labels,
        'i2a',
        precision_k,
        accuracy_k,
    ))
    metrics.update(_direction_metrics(
        audio_emb,
        image_emb,
        labels,
        labels,
        'a2i',
        precision_k,
        accuracy_k,
    ))
    metrics.update(_direction_metrics(
        image_emb,
        image_emb,
        labels,
        labels,
        'i2i',
        precision_k,
        accuracy_k,
        exclude_self=True,
    ))
    metrics.update(_direction_metrics(
        audio_emb,
        audio_emb,
        labels,
        labels,
        'a2a',
        precision_k,
        accuracy_k,
        exclude_self=True,
    ))
    return metrics
