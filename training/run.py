"""Plan or run explicit experiment presets. Default: dry-run, with no writes."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = {
    'run_plain_ssl_3k', 'run_datasize_ablation', 'run_model_size_ablation_16k',
    'run_swin_capacity_ablation', 'run_hgru_datasize_ablation',
    'run_pretrained_backbones_4k', 'pretrain_plain_spt', 'run_drive_experiment',
}
TARGET_VERSIONS = {
    'pathfinder': 'analytic_capsule_pixel_centres_r1p5_v2',
    'dense_clutter': 'legacy_antialiased_bridge_prequantisation_v1',
}


def read_presets(path):
    with Path(path).open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get('schema_version') != 1:
        raise ValueError('Expected experiment schema_version: 1')
    for name, preset in config['experiments'].items():
        names = [name] + preset.get('backbones', []) + preset.get('conditions', [])
        if any(not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value) for value in names):
            raise ValueError('Experiment/model/condition names must be simple path-safe identifiers')
        if preset['runner'] not in RUNNERS or preset['dataset'] not in {*TARGET_VERSIONS, 'drive'}:
            raise ValueError(f'Unsupported runner or dataset in {name}')
    return config


def subset(requested, available, field):
    result = list(available if requested is None else requested)
    if not result or len(result) != len(set(result)) or any(x not in available for x in result):
        raise ValueError(f'{field} must be a non-empty, unique subset of {available}')
    return result


def build_plan(config, experiment, dataset_root, output_dir, *, seeds=None, models=None,
               splits=None, conditions=None, spt_dir=None, epochs=None):
    if experiment not in config['experiments']:
        raise ValueError(f'Unknown experiment: {experiment}')
    preset = copy.deepcopy(config['experiments'][experiment])
    seeds = list(config['seeds'] if seeds is None else seeds)
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Seeds must be unique non-negative integers')
    if epochs is not None and epochs < 1:
        raise ValueError('Epochs must be positive')
    dataset_root, output_dir = Path(dataset_root), Path(output_dir)
    jobs = []
    if preset['dataset'] == 'drive':
        if splits or spt_dir:
            raise ValueError('DRIVE does not accept split/SPT overrides')
        for backbone in subset(models, preset['backbones'], 'models'):
            for condition in subset(conditions, preset['conditions'], 'conditions'):
                name = f'drive_{backbone}_{condition}'
                cfg = copy.deepcopy(preset['configuration'])
                cfg.update(experiment=name, condition=condition,
                           model=dict(backbone=backbone, pretrained=backbone.startswith('resnet')))
                cfg['data']['dataset_root'] = str(dataset_root.resolve())
                cfg['training']['seeds'] = seeds
                cfg['training']['aux_weight'] = 0.0 if condition == 'seg_only' else 0.05
                if epochs is not None:
                    cfg['training']['epochs'] = epochs
                jobs.append(dict(name=name, configuration=cfg, options={'save-checkpoint': True}))
    else:
        if conditions:
            raise ValueError('--conditions is only a DRIVE filter; select the named Pathfinder condition')
        opts = preset['options']
        if models:
            if 'models' not in opts:
                raise ValueError('This preset has no model-list filter')
            opts['models'] = subset(models, opts['models'], 'models')
        if splits:
            if 'splits' not in opts:
                raise ValueError('This preset has one fixed training split')
            opts['splits'] = subset(splits, opts['splits'], 'splits')
        if 'pretrained-backbone-path' in opts:
            if spt_dir is None:
                raise ValueError('SPT fine-tuning requires --spt-dir with matching seed checkpoints')
            opts['pretrained-backbone-path'] = opts['pretrained-backbone-path'].replace('{spt_dir}', str(Path(spt_dir).resolve()))
        elif spt_dir is not None:
            raise ValueError('--spt-dir is only for SPT fine-tuning')
        opts['seeds'] = seeds
        if epochs is not None:
            opts['epochs'] = epochs
        root_flag = 'dataset-root' if preset['runner'] == 'run_hgru_datasize_ablation' else 'root'
        opts[root_flag] = str(dataset_root.resolve())
        jobs.append(dict(name=experiment, options=opts))
    for job in jobs:
        job.update(runner=preset['runner'], dataset=preset['dataset'], protocol=preset['protocol'],
                   seeds=seeds, nonstandard_epoch_override=epochs, output_dir=str((output_dir / job['name']).resolve()))
        job['options']['out-dir'] = job['output_dir']
        if 'configuration' in job:
            job['options']['config'] = str(Path(job['output_dir']) / 'resolved.yaml')
    return jobs


def command(job):
    args = [sys.executable, '-I', '-B', str(ROOT / 'training/_runners' / (job['runner'] + '.py'))]
    for key, value in job['options'].items():
        if not isinstance(key, str) or not key or not all(c.isalnum() or c == '-' for c in key):
            raise ValueError('Invalid argument name')
        if value is False or value is None:
            continue
        args.append('--' + key)
        if value is not True:
            args.extend(str(v) for v in (value if isinstance(value, list) else [value]))
    return args


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(jobs, dataset_root):
    """Read metadata and checkpoint availability only; loaders validate image contents."""
    root = Path(dataset_root).resolve()
    dataset = jobs[0]['dataset']
    metadata = root / ('metadata_qati_pairs_v1' if dataset == 'drive' else 'metadata_dashed_with_points')
    if not metadata.is_dir() and dataset != 'drive':
        metadata = root / 'metadata'
    summary = metadata / 'summary.json' if dataset == 'drive' else root / 'summary.json'
    payload = json.loads(summary.read_text(encoding='utf-8'))
    if dataset in TARGET_VERSIONS:
        version = payload.get('qati_target_version', payload.get('target_version'))
        if version != TARGET_VERSIONS[dataset]:
            raise ValueError(f'Wrong target version: expected {TARGET_VERSIONS[dataset]}, got {version}')
    elif payload.get('target_version') != 'drive_native_analytic_capsule_pixel_centres_r6_v1':
        raise ValueError('Wrong or missing DRIVE native target version')
    selected = set()
    for job in jobs:
        opts = job['options']
        selected.update(['train', 'val', 'test'] if dataset == 'drive' else opts.get('splits', [opts.get('train-split', 'train')]))
        if job['runner'] != 'pretrain_plain_spt':
            selected.update(['val', 'test'])
        if 'pretrained-backbone-path' in opts:
            for seed in job['seeds']:
                file = Path(opts['pretrained-backbone-path'].format(seed=seed))
                if not file.is_file():
                    raise FileNotFoundError(file)
    provenance = {'summary_sha256': sha256(summary), 'metadata': {}}
    for split in sorted(selected):
        file = metadata / (split + ('_pairs.csv' if dataset == 'drive' else '.csv'))
        with file.open(encoding='utf-8-sig', newline='') as stream:
            count = sum(1 for _ in csv.DictReader(stream))
        expected = int(split.split('_')[1]) if split.startswith('train_') else None
        if count < 1 or (expected is not None and count != expected):
            raise ValueError(f'Invalid split size: {split}: {count}')
        provenance['metadata'][split] = dict(rows=count, sha256=sha256(file))
    if dataset == 'drive':
        file = metadata / 'test_images.csv'
        with file.open(encoding='utf-8-sig', newline='') as stream:
            count = sum(1 for _ in csv.DictReader(stream))
        if count < 1:
            raise ValueError('Empty DRIVE test-image manifest')
        provenance['metadata']['test_images'] = dict(rows=count, sha256=sha256(file))
    return provenance


def validate_output(output, dataset):
    output, dataset = Path(output).resolve(), Path(dataset).resolve()
    if output == ROOT or output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError('Output must be separate from the dataset and package root')
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Within this package, write run outputs only beneath outputs/')
    if output.exists():
        raise FileExistsError('Output directory already exists; choose a fresh name (no implicit overwrite/resume)')
    return output


def execute(jobs, dataset_root, output_dir, config_path, device='auto', allow_downloads=False):
    output = validate_output(output_dir, dataset_root)
    dataset_record = preflight(jobs, dataset_root)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, status='running', jobs=jobs, device_request=device,
                    allow_weight_downloads=allow_downloads, config_sha256=sha256(config_path),
                    dataset=dataset_record, runner_hashes={j['runner']: sha256(ROOT / 'training/_runners' / (j['runner'] + '.py')) for j in jobs},
                    note='New execution, not proof of historical result reproduction. Run logs can contain local paths; anonymise before sharing.')
    manifest_path = output / 'run_manifest.json'
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    save()
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(RTS_DEVICE=device, RTS_ALLOW_DOWNLOADS='1' if allow_downloads else '0',
               PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    if device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    started = time.perf_counter()
    try:
        for job in jobs:
            directory = Path(job['output_dir'])
            directory.mkdir()
            env['RTS_RUNTIME_REPORT'] = str(directory / 'runtime.json')
            env['RTS_CHECKPOINT_ROOT'] = str(directory / 'checkpoints')
            env['RTS_JOB_SPEC'] = str(directory / 'resolved.json')
            job['dataset_provenance'] = dataset_record
            if 'configuration' in job:
                (directory / 'resolved.yaml').write_text(yaml.safe_dump(job['configuration'], sort_keys=False), encoding='utf-8')
            (directory / 'resolved.json').write_text(json.dumps(job, indent=2) + '\n', encoding='utf-8')
            print('Running ' + job['name'], flush=True)
            with (directory / 'console.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(command(job), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            job['exit_code'] = result.returncode
            save()
            if result.returncode:
                raise RuntimeError(f"{job['name']} failed; see {directory / 'console.log'}")
        manifest['status'] = 'complete'
    except BaseException:
        manifest['status'] = 'failed'
        raise
    finally:
        manifest['launcher_elapsed_seconds'] = time.perf_counter() - started
        save()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments.yaml')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--experiment')
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--models', nargs='+', help='Subset of preset models; for DRIVE, backbone names')
    parser.add_argument('--splits', nargs='+', help='Subset of a data-scaling preset')
    parser.add_argument('--conditions', nargs='+', help='Subset of DRIVE conditions')
    parser.add_argument('--spt-dir', type=Path)
    parser.add_argument('--epochs', type=int, help='Explicit nonstandard override, recorded in the plan')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--allow-weight-downloads', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Start training; otherwise only print the plan')
    mode.add_argument('--dry-run', action='store_true', help='Explicit spelling of the no-write default')
    args = parser.parse_args()
    try:
        config = read_presets(args.config)
        if args.list:
            print('\n'.join(config['experiments']))
            return
        if not args.experiment or args.dataset_root is None or args.output_dir is None:
            parser.error('--experiment, --dataset-root and --output-dir are required')
        jobs = build_plan(config, args.experiment, args.dataset_root, args.output_dir,
                          seeds=args.seeds, models=args.models, splits=args.splits,
                          conditions=args.conditions, spt_dir=args.spt_dir, epochs=args.epochs)
        if args.execute:
            execute(jobs, args.dataset_root, args.output_dir, args.config, args.device, args.allow_weight_downloads)
        else:
            print(json.dumps({'dry_run': True, 'input_files_validated': False, 'jobs': jobs,
                              'commands': [command(j) for j in jobs]}, indent=2))
    except (ValueError, OSError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
