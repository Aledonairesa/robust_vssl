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
    'train_top1_i2a',
    'train_top5_i2a',
    'train_top1_a2i',
    'train_top5_a2i',
    'train_top1_ts_i2a',
    'train_top5_ts_i2a',
    'train_top1_ts_a2i',
    'train_top5_ts_a2i',
    'train_sigmoid_t',
    'train_sigmoid_scale',
    'train_sigmoid_b',
    'train_epoch_seconds',
    'validation_ran',
    'val_loss',
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


DASHBOARD_PLOTS = [
    (
        'Losses',
        [
            ('train_loss', 'Train total'),
            ('train_loss_cl', 'Train contrastive'),
            ('train_loss_cl_ts', 'Train transformed contrastive'),
            ('train_loss_ts', 'Train equivariance'),
            ('val_loss', 'Validation'),
        ],
    ),
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


def record_epoch(metrics_dir, row):
    os.makedirs(metrics_dir, exist_ok=True)
    csv_path = os.path.join(metrics_dir, 'epochs.csv')
    rows = _read_rows(csv_path)
    normalized_row = {
        field: _serialize_value(row.get(field))
        for field in FIELDNAMES
    }
    normalized_row['epoch'] = str(int(row['epoch']))
    rows_by_epoch = {int(existing['epoch']): existing for existing in rows}
    rows_by_epoch[int(normalized_row['epoch'])] = normalized_row
    rows = [rows_by_epoch[epoch] for epoch in sorted(rows_by_epoch)]

    _write_csv_atomic(csv_path, rows)
    _write_dashboard_atomic(
        os.path.join(metrics_dir, 'training_dashboard.png'), rows)
    _write_latest_metrics_atomic(
        os.path.join(metrics_dir, 'latest_metrics.txt'), rows[-1])


def _read_rows(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, 'r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        return [
            {field: row.get(field, '') for field in FIELDNAMES}
            for row in reader
            if row.get('epoch')
        ]


def _serialize_value(value):
    if value is None:
        return ''
    if isinstance(value, float) and not math.isfinite(value):
        return ''
    return str(value)


def _write_csv_atomic(csv_path, rows):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                'w', newline='', delete=False, dir=os.path.dirname(csv_path),
                prefix='.epochs-', suffix='.tmp') as csv_file:
            temp_path = csv_file.name
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, csv_path)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)


def _write_dashboard_atomic(dashboard_path, rows):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    epochs = [int(row['epoch']) for row in rows]

    for ax, (title, series) in zip(axes.flat, DASHBOARD_PLOTS):
        plotted = False
        for field, label in series:
            points = [
                (epoch, _parse_float(row.get(field)))
                for epoch, row in zip(epochs, rows)
            ]
            points = [(epoch, value) for epoch, value in points
                      if value is not None]
            if points:
                ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker='o',
                    linewidth=1.5,
                    markersize=3,
                    label=label,
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
