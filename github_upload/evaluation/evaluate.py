"""Evaluate release checkpoints on held-out test data; never select by test score."""
from __future__ import annotations
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_dataset(payload, root):
    from training.checkpoints import sha256
    job = payload['job']
    provenance = job.get('dataset_provenance')
    if not provenance:
        raise ValueError('Checkpoint has no dataset provenance; use the public training launcher')
    drive = job['dataset'] == 'drive'
    view = job['options'].get('view', 'dashed_with_points')
    metadata = root / ('metadata_qati_pairs_v1' if drive else 'metadata_' + view)
    if not metadata.is_dir() and not drive:
        metadata = root / 'metadata'
    summary = metadata / 'summary.json' if drive else root / 'summary.json'
    if sha256(summary) != provenance['summary_sha256']:
        raise ValueError('Dataset summary differs from the training dataset')
    keys = ['test'] + (['test_images'] if drive else [])
    for key in keys:
        filename = key + ('_pairs.csv' if drive and key == 'test' else '.csv')
        expected = provenance['metadata'].get(key, {}).get('sha256')
        if not expected or sha256(metadata / filename) != expected:
            raise ValueError(f'Test metadata differs from training provenance: {filename}')
    if drive:
        with (metadata / 'test_images.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len({r['image_id'] for r in rows}) != len(rows):
            raise ValueError('Repeated DRIVE image IDs are not independent test units')
        for row in rows:
            for key in ('image_path', 'vessel_mask_path', 'fov_mask_path'):
                path = (root / row[key]).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError('DRIVE image path escapes the dataset or is missing')
    return dict(summary_sha256=provenance['summary_sha256'],
                metadata={k: provenance['metadata'][k] for k in keys})


def grouping(payload):
    """Only seed/path fields are removed; actual model/protocol/split remain distinct."""
    cfg = {k: v for k, v in payload['config'].items() if k != 'seed'}
    job = payload['job']
    options = {k: v for k, v in job.get('options', {}).items() if k not in
               ('root', 'dataset-root', 'out-dir', 'seeds', 'models', 'splits', 'config', 'pretrained-backbone-path')}
    # A scaling job can contain many splits; each checkpoint belongs to just one.
    provenance = job['dataset_provenance']
    split = cfg.get('split_name', options.get('train-split', 'train'))
    return dict(runner=payload['runner'], model=payload['model_spec'], config=cfg,
                options=options, dataset=job['dataset'], train_split=split,
                dataset_summary_sha256=provenance['summary_sha256'],
                metadata={k: v for k, v in provenance['metadata'].items()
                          if k in (split, 'train' if job['dataset'] == 'drive' else split, 'val', 'test', 'test_images')},
                source_hashes=payload['source_hashes'], selection=payload['selection'])


def evaluate_checkpoint(checkpoint, dataset_root, device, *, allow_source_mismatch=False,
                        allow_device_change=False):
    import numpy as np
    import torch
    from training.checkpoints import build_model, restore_rng, sha256
    from training.run import RUNNERS
    checkpoint, dataset_root = Path(checkpoint), Path(dataset_root).resolve()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload.get('format') != 'rts-evaluation-checkpoint-v1':
        raise ValueError('Expected a new release checkpoint, not a legacy bare state dict')
    if payload['runner'] not in RUNNERS - {'pretrain_plain_spt'}:
        raise ValueError('Unsupported evaluation runner')
    index_path = checkpoint.parent / 'checkpoints.json'
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding='utf-8'))
        entry = index[payload['selection']]
        if entry['path'] != checkpoint.name or entry['sha256'] != sha256(checkpoint):
            raise ValueError('Checkpoint differs from its integrity index')
    provenance = check_dataset(payload, dataset_root)
    changed = []
    for relative, expected in payload['source_hashes'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Checkpoint source path escapes the package')
        if not path.is_file() or sha256(path) != expected:
            changed.append(relative)
    if changed and not allow_source_mismatch:
        raise ValueError('Source changed since training; explicit --allow-source-mismatch required: ' + ', '.join(changed))
    is_hgru = payload['runner'] == 'run_hgru_datasize_ablation'
    if is_hgru and device.type != payload['device_type'] and not allow_device_change:
        raise ValueError('hGRU RNG replay requires the original device type; use --allow-device-change for a new, separately labelled evaluation')
    torch.manual_seed(payload['seed'])
    np.random.seed(payload['seed'])
    model = build_model(payload['model_spec']).to(device)
    model.load_state_dict(payload['model_state'], strict=True)
    model.eval()
    cfg, job = payload['config'], payload['job']
    runner = importlib.import_module('training._runners.' + payload['runner'])
    runner.DEV = device
    started = time.perf_counter()
    details = None
    if job['dataset'] == 'drive':
        from evaluation.drive import evaluate_test_image_segmentation
        # Clean image inference; no query markers or annotation-derived inputs.
        details = evaluate_test_image_segmentation(model, device, dataset_root, image_size=cfg['image_size'])
        metrics = {k.removesuffix('_mean'): v for k, v in details.items() if k.endswith('_mean')}
        semantics = 'clean_image_per_image_mean_inside_fov'
        count = len(details['per_image'])
        precision = 'fp32'
    else:
        from data_loading.pathfinder import load_split_data, load_label_split
        view = job['options'].get('view', 'dashed_with_points')
        if is_hgru:
            x, y = load_label_split(dataset_root, 'test', view=view)
            d = beta = None
        else:
            x, y, d, beta, _ = load_split_data(dataset_root, 'test', view=view)
        if cfg.get('condition') == 'betti0':
            d = beta
        if payload['test_rng'] is not None:
            restore_rng(payload['test_rng'], device)
        family = payload['runner']
        precision = ('fp16' if family == 'run_plain_ssl_3k' else cfg.get('precision', 'bf16')) if device.type == 'cuda' else 'fp32'
        if family == 'run_plain_ssl_3k':
            metrics = runner.evaluate(model, x, y, d, is_multitask=cfg['model_type'].startswith('multitask_'), aux_weight=cfg['aux_weight'])
        elif family == 'run_datasize_ablation':
            metrics = runner.evaluate(model, cfg['model_name'], x, y, d, aux_weight=cfg['aux_weight'], precision=cfg['precision'])
        elif family == 'run_swin_capacity_ablation':
            metrics = runner.evaluate(model, x, y, d, precision=cfg['precision'])
        elif is_hgru:
            metrics = runner.evaluate(model, x, y, batch_size=64, device=device)
        else:
            metrics = runner.evaluate(model, x, y, d, aux_weight=cfg['aux_weight'], precision=cfg['precision'])
        if cfg.get('condition') == 'betti0':
            metrics['global_betti0_mae'] = metrics.pop('delta_mae')
            metrics['loss_global_betti0'] = metrics.pop('loss_delta')
        if cfg.get('model_type') == 'plain_conn_only':
            metrics.pop('delta_mae', None)
            metrics.pop('loss_delta', None)
        semantics, count = 'scene_level_classification', len(y)
    if count < 1 or not all(np.isfinite(v) for v in metrics.values()):
        raise ValueError('Empty or non-finite test result')
    group = grouping(payload)
    group['evaluation'] = dict(split='test', semantics=semantics, precision=precision,
                               device_type=device.type, batch_size=1 if details is not None else 64,
                               source_mismatches=changed,
                               current_source_hashes={p: sha256(ROOT / p) for p in payload['source_hashes'] if (ROOT / p).is_file()})
    group['evaluation']['evaluator_sha256'] = sha256(Path(__file__))
    return dict(format='rts-evaluation-v1', seed=payload['seed'], epoch=payload['epoch'],
                selection=payload['selection'], selection_metric=payload['selection_metric'],
                group=group, checkpoint_sha256=sha256(checkpoint), dataset=provenance,
                parameter_count=payload['parameter_count'], num_test_units=count,
                metrics=metrics, units={'accuracy': 'fraction', 'balanced_accuracy': 'fraction'},
                per_image=details['per_image'] if details is not None else None,
                evaluation_seconds=time.perf_counter() - started,
                environment=dict(python=sys.version.split()[0], torch=str(torch.__version__),
                                 cuda=torch.version.cuda, device_type=device.type,
                                 hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'),
                recorded_test_metrics=payload['recorded_test_metrics'],
                rng_replay='saved_state_same_device_type' if is_hgru and device.type == payload['device_type'] else
                           'seeded_cross_device_not_exact_replay' if is_hgru else 'not_required',
                note='Independent test evaluation; no test-driven model selection. Metadata identity is checked, not historical pixel-byte identity.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path, help='Public training output; recursively finds release checkpoints')
    source.add_argument('--checkpoint', type=Path)
    parser.add_argument('--selection', choices=['best_val', 'final', 'both'], default='best_val')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--allow-source-mismatch', action='store_true')
    parser.add_argument('--allow-device-change', action='store_true')
    args = parser.parse_args()
    os.environ.update(RTS_DEVICE=args.device, RTS_ALLOW_DOWNLOADS='0')
    os.environ.pop('RTS_RUNTIME_REPORT', None)
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from training.run import validate_output
    from training.runtime import select_device
    from training.checkpoints import sha256
    try:
        output = validate_output(args.output_dir, args.dataset_root)
        selected = ('best_val', 'final') if args.selection == 'both' else (args.selection,)
        if args.run_dir:
            manifest_path = args.run_dir / 'run_manifest.json'
            if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding='utf-8'))['status'] != 'complete':
                raise ValueError('Training run is incomplete; use an explicit checkpoint only if partial evaluation is intended')
        files = [args.checkpoint] if args.checkpoint else sorted(p for s in selected for p in args.run_dir.rglob(s + '.pt'))
        if not files or any(not p.is_file() for p in files):
            raise ValueError('No release checkpoints found')
        output.mkdir(parents=True)
        device = select_device(allow_mps=False)
        manifest = dict(format='rts-evaluation-manifest-v1', status='running', results=[])
        manifest_path = output / 'evaluation_manifest.json'
        def save():
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        save()
        try:
            for i, path in enumerate(files):
                result = evaluate_checkpoint(path, args.dataset_root, device,
                          allow_source_mismatch=args.allow_source_mismatch, allow_device_change=args.allow_device_change)
                if result['selection'] not in selected:
                    raise ValueError('Checkpoint selection disagrees with --selection')
                name = f"evaluation_{i:04d}_s{result['seed']}_{result['selection']}.json"
                result_path = output / name
                result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
                manifest['results'].append(dict(path=name, sha256=sha256(result_path)))
                print(name + ': ' + json.dumps(result['metrics']), flush=True)
                save()
            manifest['status'] = 'complete'
        except BaseException:
            manifest['status'] = 'failed'
            raise
        finally:
            save()
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
