import csv
import math
import os
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


FIELDNAMES = [
    'epoch',
    'iteration',
    'learning_rate',
    'train_loss',
    'train_loss_cl',
    'train_loss_cl_ts',
    'train_loss_ts',
    'train_loss_atp',
    'train_loss_atp_ts',
    'train_loss_cu',
    'train_loss_cu_ts',
    'train_loss_cl_weighted',
    'train_loss_cl_ts_weighted',
    'train_loss_ts_weighted',
    'train_loss_atp_weighted',
    'train_loss_cu_weighted',
    'train_loss_cl_weight',
    'train_loss_cl_ts_weight',
    'train_loss_ts_weight',
    'train_loss_atp_weight',
    'train_loss_cu_weight',
    'train_top1_i2a',
    'train_top5_i2a',
    'train_top1_a2i',
    'train_top5_a2i',
    'train_top1_ts_i2a',
    'train_top5_ts_i2a',
    'train_top1_ts_a2i',
    'train_top5_ts_a2i',
    'train_temperature_v',
    'train_temperature',
    'train_inverse_temperature',
    'train_temperature_learning_rate',
    'train_sigmoid_t',
    'train_sigmoid_scale',
    'train_sigmoid_b',
    'train_epoch_seconds',
    'validation_ran',
    'val_loss',
    'val_loss_weighted',
    'val_loss_weight',
    'val_top1_i2a',
    'val_top5_i2a',
    'val_top1_a2i',
    'val_top5_a2i',
    'val_mean_ciou',
    'val_mean_auc',
    'val_has_annotations',
    'val_epoch_seconds',
    'checkpoint_metric',
    'checkpoint_score',
    'best_metric_score',
    'is_best',
    'early_stop_wait',
]


TEST_FIELDNAMES = [
    'epoch',
    'test_set',
    'test_set_scale',
    'checkpoint_path',
    'num_samples',
    'loss',
    'top1_i2a',
    'top5_i2a',
    'top1_a2i',
    'top5_a2i',
    'mean_ciou',
    'auc',
    'has_annotations',
    'test_seconds',
]


DASHBOARD_PLOTS = [
    (
        'Validation Retrieval Accuracy',
        [
            ('val_top1_i2a', 'Top-1 image to audio'),
            ('val_top5_i2a', 'Top-5 image to audio'),
            ('val_top1_a2i', 'Top-1 audio to image'),
            ('val_top5_a2i', 'Top-5 audio to image'),
        ],
    ),
    (
        'Train Retrieval Accuracy',
        [
            ('train_top1_i2a', 'Top-1 image to audio'),
            ('train_top5_i2a', 'Top-5 image to audio'),
            ('train_top1_a2i', 'Top-1 audio to image'),
            ('train_top5_a2i', 'Top-5 audio to image'),
        ],
    ),
    (
        'Transformed Train Retrieval Accuracy',
        [
            ('train_top1_ts_i2a', 'Top-1 image to audio'),
            ('train_top5_ts_i2a', 'Top-5 image to audio'),
            ('train_top1_ts_a2i', 'Top-1 audio to image'),
            ('train_top5_ts_a2i', 'Top-5 audio to image'),
        ],
    ),
    (
        'Validation Localization',
        [
            ('val_mean_ciou', 'Mean cIoU'),
            ('val_mean_auc', 'Mean AUC'),
        ],
    ),
    (
        'Epoch Duration',
        [
            ('train_epoch_seconds', 'Train'),
            ('val_epoch_seconds', 'Validation'),
        ],
    ),
]


LOSS_DASHBOARD_PLOTS = [
    {
        'title': 'Train Total Loss',
        'series': [
            {'field': 'train_loss', 'label': 'Train total'},
        ],
    },
    {
        'title': 'Contrastive Loss',
        'series': [
            {
                'field': 'train_loss_cl_weighted',
                'fallback_field': 'train_loss_cl',
                'weight_field': 'train_loss_cl_weight',
                'default_weight': 0.5,
                'label': 'Train contrastive',
            },
            {
                'field': 'val_loss_weighted',
                'fallback_field': 'val_loss',
                'weight_field': 'val_loss_weight',
                'default_weight': 0.5,
                'label': 'Validation contrastive',
            },
        ],
    },
    {
        'title': 'Train Transformed Contrastive Loss',
        'series': [
            {
                'field': 'train_loss_cl_ts_weighted',
                'fallback_field': 'train_loss_cl_ts',
                'weight_field': 'train_loss_cl_ts_weight',
                'default_weight': 0.5,
                'label': 'Train transformed contrastive',
            },
        ],
    },
    {
        'title': 'Train Equivariance Loss',
        'series': [
            {
                'field': 'train_loss_ts_weighted',
                'fallback_field': 'train_loss_ts',
                'weight_field': 'train_loss_ts_weight',
                'default_weight': 1.0,
                'label': 'Train equivariance',
            },
        ],
    },
    {
        'title': 'Train ATP Loss',
        'optional': True,
        'activation_weight_field': 'train_loss_atp_weight',
        'series': [
            {
                'field': 'train_loss_atp_weighted',
                'weight_field': 'train_loss_atp_weight',
                'default_weight': 0.0,
                'label': 'Train ATP',
            },
        ],
    },
    {
        'title': 'Train CU Loss',
        'optional': True,
        'activation_weight_field': 'train_loss_cu_weight',
        'series': [
            {
                'field': 'train_loss_cu_weighted',
                'weight_field': 'train_loss_cu_weight',
                'default_weight': 0.0,
                'label': 'Train CU',
            },
        ],
    },
]


def record_epoch(metrics_dir, row):
    os.makedirs(metrics_dir, exist_ok=True)
    csv_path = os.path.join(metrics_dir, 'training_epochs.csv')
    rows = _read_rows(csv_path, FIELDNAMES)
    normalized_row = {
        field: _serialize_value(row.get(field))
        for field in FIELDNAMES
    }
    normalized_row['epoch'] = str(int(row['epoch']))
    rows_by_epoch = {int(existing['epoch']): existing for existing in rows}
    rows_by_epoch[int(normalized_row['epoch'])] = normalized_row
    rows = [rows_by_epoch[epoch] for epoch in sorted(rows_by_epoch)]

    _write_csv_atomic(csv_path, rows, FIELDNAMES, '.training-epochs-')
    _write_dashboard_atomic(
        os.path.join(metrics_dir, 'training_dashboard.png'), rows)
    _write_latest_metrics_atomic(
        os.path.join(metrics_dir, 'latest_metrics.txt'), rows[-1])


def record_test_result(metrics_dir, row):
    os.makedirs(metrics_dir, exist_ok=True)
    csv_path = os.path.join(metrics_dir, 'test_metrics.csv')
    rows = _read_rows(csv_path, TEST_FIELDNAMES)
    normalized_row = {
        field: _serialize_value(row.get(field))
        for field in TEST_FIELDNAMES
    }
    normalized_row['epoch'] = str(int(row['epoch']))

    def row_key(current_row):
        return (
            current_row['test_set'],
            current_row['test_set_scale'],
            int(current_row['epoch']),
            current_row['checkpoint_path'],
        )

    rows_by_key = {row_key(existing): existing for existing in rows}
    rows_by_key[row_key(normalized_row)] = normalized_row
    rows = sorted(
        rows_by_key.values(),
        key=lambda current_row: (
            int(current_row['epoch']),
            current_row['test_set'],
            current_row['test_set_scale'],
            current_row['checkpoint_path'],
        ),
    )
    _write_csv_atomic(
        csv_path, rows, TEST_FIELDNAMES, '.test-metrics-')


def _read_rows(csv_path, fieldnames):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, 'r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        return [
            {field: row.get(field, '') for field in fieldnames}
            for row in reader
            if row.get('epoch')
        ]


def _serialize_value(value):
    if value is None:
        return ''
    if isinstance(value, float) and not math.isfinite(value):
        return ''
    return str(value)


def _write_csv_atomic(csv_path, rows, fieldnames, prefix):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                'w', newline='', delete=False, dir=os.path.dirname(csv_path),
                prefix=prefix, suffix='.tmp') as csv_file:
            temp_path = csv_file.name
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, csv_path)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)


def _write_dashboard_atomic(dashboard_path, rows):
    plot_specs = _dashboard_plot_specs(rows)
    num_cols = 2
    num_rows = int(math.ceil(len(plot_specs) / num_cols))
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(14, max(4, 4 * num_rows)))
    if hasattr(axes, 'flat'):
        axes_flat = list(axes.flat)
    else:
        axes_flat = [axes]
    epochs = [int(row['epoch']) for row in rows]

    for ax, plot_spec in zip(axes_flat, plot_specs):
        title = plot_spec['title']
        series = plot_spec['series']
        plotted = False
        for series_spec in series:
            points = _series_points(rows, epochs, series_spec)
            points = [(epoch, value) for epoch, value in points
                      if value is not None]
            if points:
                ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker='o',
                    linewidth=1.5,
                    markersize=3,
                    label=_series_label(rows, series_spec),
                )
                plotted = True
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize='small')
        else:
            ax.text(
                0.5, 0.5, 'No data yet', ha='center', va='center',
                transform=ax.transAxes)

    for ax in axes_flat[len(plot_specs):]:
        ax.axis('off')

    fig.tight_layout()
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                delete=False, dir=os.path.dirname(dashboard_path),
                prefix='.dashboard-', suffix='.tmp') as temp_file:
            temp_path = temp_file.name
        fig.savefig(temp_path, format='png', dpi=150)
        os.replace(temp_path, dashboard_path)
    finally:
        plt.close(fig)
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)


def _write_latest_metrics_atomic(latest_path, row):
    lines = ['Epoch {}'.format(row['epoch'])]
    for field in FIELDNAMES:
        value = row.get(field, '')
        if field != 'epoch' and value != '':
            lines.append('{}: {}'.format(field, value))

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                'w', delete=False, dir=os.path.dirname(latest_path),
                prefix='.latest-', suffix='.tmp') as latest_file:
            temp_path = latest_file.name
            latest_file.write('\n'.join(lines) + '\n')
        os.replace(temp_path, latest_path)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)


def _parse_float(value):
    if value in (None, ''):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _dashboard_plot_specs(rows):
    loss_plots = []
    for plot_spec in LOSS_DASHBOARD_PLOTS:
        if plot_spec.get('optional') and not _optional_plot_active(
                rows, plot_spec):
            continue
        loss_plots.append(plot_spec)

    base_plots = [
        {
            'title': title,
            'series': [
                {'field': field, 'label': label}
                for field, label in series
            ],
        }
        for title, series in DASHBOARD_PLOTS
    ]
    return loss_plots + base_plots


def _optional_plot_active(rows, plot_spec):
    weight_field = plot_spec.get('activation_weight_field')
    if weight_field is not None:
        for row in rows:
            weight = _parse_float(row.get(weight_field))
            if weight is not None and weight > 0:
                return True

    for series_spec in plot_spec['series']:
        for row in rows:
            value = _series_value(row, series_spec)
            if value is not None and abs(value) > 0:
                return True
    return False


def _series_points(rows, epochs, series_spec):
    return [
        (epoch, _series_value(row, series_spec))
        for epoch, row in zip(epochs, rows)
    ]


def _series_value(row, series_spec):
    value = _parse_float(row.get(series_spec['field']))
    if value is not None:
        return value

    fallback_field = series_spec.get('fallback_field')
    if fallback_field is None:
        return None

    fallback_value = _parse_float(row.get(fallback_field))
    if fallback_value is None:
        return None
    weight = _series_weight(row, series_spec)
    return fallback_value * weight


def _series_label(rows, series_spec):
    label = series_spec['label']
    if 'weight_field' not in series_spec:
        return label

    weight = None
    for row in rows:
        weight = _series_weight(row, series_spec)
        if weight is not None:
            break
    if weight is None:
        return label
    return '{} ({:g})'.format(label, weight)


def _series_weight(row, series_spec):
    weight = _parse_float(row.get(series_spec.get('weight_field')))
    if weight is not None:
        return weight
    return series_spec.get('default_weight')
