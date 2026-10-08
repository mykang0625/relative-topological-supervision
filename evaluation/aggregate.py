"""Aggregate independent evaluation records into JSON, CSV and Markdown (sample SD)."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_records(paths):
    files = []
    for path in paths:
        path = Path(path)
        files.extend(sorted(path.rglob('evaluation_*.json')) if path.is_dir() else [path])
    records = []
    for path in files:
        if path.name == 'evaluation_manifest.json':
            continue
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('format') != 'rts-evaluation-v1':
            raise ValueError(f'Not an independent per-seed evaluation: {path}')
        manifest_path = path.parent / 'evaluation_manifest.json'
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest.get('status') != 'complete':
                raise ValueError(f'Incomplete evaluation directory: {path.parent}')
            hashes = {r['path']: r['sha256'] for r in manifest['results']}
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(path.name):
                raise ValueError(f'Evaluation file is absent from or differs from its manifest: {path}')
        record['_input_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(record)
    if not records:
        raise ValueError('No independent evaluation records')
    return records


def aggregate(records, expected_seeds=None):
    if expected_seeds is not None and (not expected_seeds or len(set(expected_seeds)) != len(expected_seeds)
                                      or any(type(s) is not int or s < 0 for s in expected_seeds)):
        raise ValueError('Expected seeds must be non-empty and unique')
    groups = {}
    checkpoints = set()
    for record in records:
        if record.get('format') != 'rts-evaluation-v1':
            raise ValueError('Unsupported evaluation format')
        group = record['group']
        if group['selection'] != record['selection'] or group['evaluation']['split'] != 'test':
            raise ValueError('Mixed or inconsistent selection/split metadata')
        identity = json.dumps(group, sort_keys=True, separators=(',', ':'))
        # best and final may have identical weights; the payload hash still differs.
        duplicate = (record['checkpoint_sha256'], identity)
        if duplicate in checkpoints:
            raise ValueError('Duplicate checkpoint evaluation')
        checkpoints.add(duplicate)
        members = groups.setdefault(identity, [])
        if any(m['seed'] == record['seed'] for m in members):
            raise ValueError('Duplicate seed within one model/protocol/selection group')
        if type(record['seed']) is not int or record['seed'] < 0:
            raise ValueError('Invalid seed')
        if type(record['num_test_units']) is not int or record['num_test_units'] < 1:
            raise ValueError('Invalid test-unit count')
        metrics = record['metrics']
        if not metrics or any(type(v) not in (int, float) or not math.isfinite(v) for v in metrics.values()):
            raise ValueError('Missing or non-finite metric')
        if members and (set(metrics) != set(members[0]['metrics']) or
                        record['num_test_units'] != members[0]['num_test_units'] or
                        record['parameter_count'] != members[0]['parameter_count'] or
                        record['units'] != members[0]['units'] or
                        set(record.get('per_image') or {}) != set(members[0].get('per_image') or {})):
            raise ValueError('Metric schema, test units, parameter count or image membership differs across seeds')
        members.append(record)
    rows = []
    for identity, members in sorted(groups.items()):
        members.sort(key=lambda r: r['seed'])
        seeds = [r['seed'] for r in members]
        if expected_seeds is not None and set(seeds) != set(expected_seeds):
            raise ValueError(f'Incomplete/unexpected seed set: got {seeds}, expected {expected_seeds}')
        values = {}
        for metric in sorted(members[0]['metrics']):
            per_seed = [r['metrics'][metric] for r in members]
            values[metric] = dict(mean=statistics.mean(per_seed),
                                  sample_std=statistics.stdev(per_seed) if len(per_seed) > 1 else None,
                                  per_seed=per_seed)
        rows.append(dict(group_id=hashlib.sha256(identity.encode()).hexdigest()[:16],
                         group=json.loads(identity), seeds=seeds, num_seeds=len(seeds),
                         status='single_seed_preliminary' if len(seeds) == 1 else 'multi_seed',
                         num_test_units=members[0]['num_test_units'],
                         parameter_count=members[0]['parameter_count'],
                         units=members[0]['units'], metrics=values,
                         per_seed_records=members))
    return dict(format='rts-aggregate-v1', aggregation='mean and sample standard deviation across independent training seeds (ddof=1)',
                single_seed_std=None, expected_seeds=expected_seeds, groups=rows)


def write_outputs(summary, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh aggregate output directory')
    output.mkdir(parents=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    fields = ['group_id', 'model', 'condition', 'train_split', 'selection', 'num_seeds', 'seeds',
              'num_test_units', 'parameter_count', 'metric', 'unit', 'mean', 'sample_std']
    markdown = ['# Test results', '', 'Mean ± sample SD across seeds. Single-seed SD is unavailable, not zero.',
                'DRIVE metrics are clean-image per-image means, not pooled query-pair scores.', '',
                '| Group | Model / condition | Split / selection | Seeds | Metric | Mean ± SD |',
                '| --- | --- | --- | --- | --- | --- |']
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary['groups']:
            group = row['group']
            model = group['model']['name']
            condition = (group['config'].get('condition') or group['options'].get('condition') or
                         ('label_only' if group['runner'] == 'run_hgru_datasize_ablation' or
                          group['config'].get('aux_weight') == 0 else 'relative'))
            for metric, value in row['metrics'].items():
                writer.writerow(dict(group_id=row['group_id'], model=model, condition=condition,
                                train_split=group['train_split'], selection=group['selection'],
                                num_seeds=row['num_seeds'], seeds=','.join(map(str, row['seeds'])),
                                num_test_units=row['num_test_units'], parameter_count=row['parameter_count'],
                                metric=metric, unit=row['units'].get(metric, 'native'),
                                mean=value['mean'], sample_std=value['sample_std']))
                scale = 100 if metric in ('accuracy', 'balanced_accuracy') else 1
                mean = f"{value['mean'] * scale:.4f}"
                std = f"{value['sample_std'] * scale:.4f}" if value['sample_std'] is not None else 'N/A'
                unit = ' (%)' if scale == 100 else ''
                markdown.append(f"| {row['group_id']} | {model} / {condition} | {group['train_split']} / {group['selection']} | {row['seeds']} | {metric}{unit} | {mean} ± {std} |")
    (output / 'summary.md').write_text('\n'.join(markdown) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-seeds', nargs='+', type=int)
    args = parser.parse_args()
    try:
        output = args.output_dir.resolve()
        if output == ROOT or (output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / 'outputs')):
            raise ValueError('Within the package, put aggregates beneath outputs/')
        if any(p.resolve().is_relative_to(output) for p in args.inputs):
            raise ValueError('Aggregate output must not contain input results')
        result = aggregate(read_records(args.inputs), args.expected_seeds)
        write_outputs(result, output)
        print(f"Wrote {len(result['groups'])} groups to {output}")
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
