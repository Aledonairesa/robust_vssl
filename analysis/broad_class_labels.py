import json
import re
from collections import Counter
from pathlib import Path


UNKNOWN_LABEL = 'unknown'
TIMESTAMP_SUFFIX_PATTERN = re.compile(r'_\d{6}$')


def infer_broad_classes_path(test_set, broad_classes_dir):
    filename = '{}_broad_classes.json'.format(test_set.lower())
    return Path(broad_classes_dir) / filename


def load_broad_class_lookup(path):
    with open(path, 'r') as json_file:
        classes_to_names = json.load(json_file)

    lookup = {}
    conflicts = []
    for class_name, names in classes_to_names.items():
        for name in names:
            name = str(name)
            previous = lookup.get(name)
            if previous is not None and previous != class_name:
                conflicts.append((name, previous, class_name))
                continue
            lookup[name] = class_name

    return lookup, conflicts


def name_candidates(name):
    stem = Path(str(name)).stem
    candidates = [stem]

    without_timestamp = TIMESTAMP_SUFFIX_PATTERN.sub('', stem)
    if without_timestamp != stem:
        candidates.append(without_timestamp)

    if len(stem) >= 11:
        candidates.append(stem[:11])

    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def assign_broad_classes(names, test_set, broad_classes_dir):
    path = infer_broad_classes_path(test_set, broad_classes_dir)
    if not path.exists():
        raise FileNotFoundError(
            'Broad-class file not found for test set {}: {}'.format(
                test_set, path))

    lookup, conflicts = load_broad_class_lookup(path)

    labels = []
    matched = 0
    for name in names:
        label = UNKNOWN_LABEL
        for candidate in name_candidates(name):
            if candidate in lookup:
                label = lookup[candidate]
                matched += 1
                break
        labels.append(label)

    counts = Counter(labels)
    diagnostics = {
        'path': path,
        'matched': matched,
        'unmatched': len(labels) - matched,
        'total': len(labels),
        'counts': counts,
        'conflicts': conflicts,
    }
    return labels, diagnostics


def print_broad_class_summary(test_set, epoch, diagnostics):
    print(
        'Broad-class coverage for {} epoch {:04d}: {}/{} matched, {} unknown'
        .format(
            test_set,
            epoch,
            diagnostics['matched'],
            diagnostics['total'],
            diagnostics['unmatched'],
        ))

    for label, count in diagnostics['counts'].most_common():
        print('  {}: {}'.format(label, count))

    conflicts = diagnostics['conflicts']
    if conflicts:
        print(
            '  warning: {} class-name conflicts found in {}'
            .format(len(conflicts), diagnostics['path']))
