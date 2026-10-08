"""CPU fixture checks for checkpoint -> independent evaluation -> aggregation."""
from __future__ import annotations
import argparse
import copy
import gc
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

os.environ.update(CUDA_VISIBLE_DEVICES='-1', RTS_DEVICE='cpu', RTS_ALLOW_DOWNLOADS='0',
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from data_loading.test_loading import pathfinder_fixture, drive_fixture, write_rows
from data_loading.pathfinder import load_split_data
from evaluation.evaluate import evaluate_checkpoint
from evaluation.aggregate import aggregate, read_records
from training import run
from training.checkpoints import CheckpointRecorder, build_model, model_spec, sha256


def rejected(fn, exception=(ValueError, RuntimeError, KeyError)):
    try:
        fn()
    except exception:
        return
    raise AssertionError('Invalid input was accepted')


def cli(script, *arguments):
    result = subprocess.run([sys.executable, '-I', '-B', str(ROOT / script), *map(str, arguments)],
                            cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=240)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return result


def pf_fixture(root):
    rows = pathfinder_fixture(root)
    for split in ('val', 'test'):
        updated = []
        for original in rows:
            row = dict(original, split=split, sample_id=split + original['sample_id'])
            row['image_path'] = row['dashed_with_points_path'] = original['image_path'].replace('train/', split + '/')
            target = root / row['image_path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / original['image_path'], target)
            updated.append(row)
        write_rows(root / 'metadata_dashed_with_points' / (split + '.csv'), updated)


def set_job(work, name, runner, dataset, family='pathfinder'):
    directory = work / name
    directory.mkdir()
    job = dict(runner=runner, dataset=family, options={'train-split': 'train_4', 'view': 'dashed_with_points'}, seeds=[23])
    job['dataset_provenance'] = run.preflight([job], dataset)
    path = directory / 'resolved.json'
    path.write_text(json.dumps(job), encoding='utf-8')
    os.environ.update(RTS_CHECKPOINT_ROOT=str(directory / 'checkpoints'), RTS_JOB_SPEC=str(path))
    return directory


def compare_saved(directory, dataset):
    results = []
    for path in sorted(directory.rglob('*.pt')):
        result = evaluate_checkpoint(path, dataset, torch.device('cpu'))
        expected = result['recorded_test_metrics']
        if expected:
            for key, value in result['metrics'].items():
                original = {'global_betti0_mae': 'delta_mae', 'loss_global_betti0': 'loss_delta'}.get(key, key)
                assert value == expected[original], (path, key, value, expected[original])
        results.append(result)
    assert len(results) == 2, directory
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    work = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    dataset = work / 'pathfinder'
    pf_fixture(dataset)
    config = run.read_presets(ROOT / 'configs/experiments.yaml')
    config['experiments'] = {'plain_label_only': config['experiments']['plain_label_only']}
    config['experiments']['plain_label_only']['options'].update({'train-split': 'train_4', 'width': 6, 'epochs': 2, 'batch-size': 2})
    cfg_path = work / 'smoke.yaml'
    cfg_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    cli('training/run.py', '--config', cfg_path, '--experiment', 'plain_label_only', '--dataset-root', dataset,
        '--output-dir', work / 'training', '--device', 'cpu', '--execute')
    cli('evaluation/evaluate.py', '--run-dir', work / 'training', '--dataset-root', dataset,
        '--output-dir', work / 'evaluated', '--selection', 'both', '--device', 'cpu')
    cli('evaluation/aggregate.py', '--inputs', work / 'evaluated', '--output-dir', work / 'aggregated', '--expected-seeds', 23, 7, 42)
    records = read_records([work / 'evaluated'])
    result = aggregate(records, [23, 7, 42])
    assert len(result['groups']) == 2 and all(g['num_seeds'] == 3 for g in result['groups'])
    for record in records:
        for key, value in record['metrics'].items():
            assert value == record['recorded_test_metrics'][key]
    rejected(lambda: aggregate(records + [records[0]]))
    rejected(lambda: aggregate(records[:1], [23, 7, 42]))
    assert aggregate(records[:1])['groups'][0]['metrics']['accuracy']['sample_std'] is None
    known = [copy.deepcopy(records[0]) for _ in range(3)]
    for i, r in enumerate(known):
        r.update(seed=i, checkpoint_sha256=str(i), metrics={'accuracy': [0.1, 0.2, 0.3][i]})
    assert abs(aggregate(known)['groups'][0]['metrics']['accuracy']['sample_std'] - 0.1) < 1e-12
    bad = copy.deepcopy(known)
    bad[0]['metrics']['accuracy'] = float('nan')
    rejected(lambda: aggregate(bad))

    # Exact re-evaluation of both selections for every Pathfinder runner family.
    train = load_split_data(dataset, 'train_4')
    val, test = load_split_data(dataset, 'val'), load_split_data(dataset, 'test')
    data = {f'{k}_{split}': values[i] for split, values in [('train', train), ('val', val), ('test', test)]
            for i, k in enumerate(('X', 'Y', 'D'))}
    short = dict(seed=23, epochs=2, batch_size=2)
    cases = []
    for condition in ('qati_delta', 'shuffled_qati_delta', 'betti0'):
        name = 'plain_' + condition
        directory = set_job(work, name, 'run_plain_ssl_3k', dataset)
        module = importlib.import_module('training._runners.run_plain_ssl_3k')
        values = dict(data)
        if condition == 'betti0':
            values.update(D_train=train[3], D_val=val[3], D_test=test[3])
        module.train_single_run(**values, **short, model_type='multitask_plain', width=6, bridge_thickness=3.,
                                condition=condition, shuffle_delta=condition == 'shuffled_qati_delta')
        compare_saved(directory, dataset)
        cases.append(name)
    from models.swin import SWIN_CONFIGS
    for runner, kwargs in [
        ('run_datasize_ablation', dict(model_name='plain', split_name='train_4')),
        ('run_datasize_ablation', dict(model_name='cnn_s', split_name='train_4')),
        ('run_datasize_ablation', dict(model_name='swin_t', split_name='train_4')),
        ('run_model_size_ablation_16k', dict(model_name='cnn_s', width=3)),
        ('run_swin_capacity_ablation', dict(config_name='swin_12_273k', cfg=SWIN_CONFIGS['swin_12_273k'])),
        ('run_pretrained_backbones_4k', dict(model_name='resnet18', split_name='train_4')),
    ]:
        name = runner + '_' + kwargs.get('model_name', 'small')
        directory = set_job(work, name, runner, dataset)
        module = importlib.import_module('training._runners.' + runner)
        from models.resnet import MultiTaskResNet
        if runner == 'run_pretrained_backbones_4k':
            with patch.object(module, 'build_pretrained_model', side_effect=lambda _: MultiTaskResNet(variant='resnet18', pretrained=False)):
                module.train_single_run(**data, **short, **kwargs, precision='fp32')
        else:
            module.train_single_run(**data, **short, **kwargs, precision='fp32')
        compare_saved(directory, dataset)
        cases.append(name)
        gc.collect()

    # hGRU forward-only RNG replay at full image resolution (no long BPTT run).
    from models.hgru import HGRU
    hgru = importlib.import_module('training._runners.run_hgru_datasize_ablation')
    directory = set_job(work, 'hgru_rng', 'run_hgru_datasize_ablation', dataset)
    model = HGRU()
    recorder = CheckpointRecorder(model, 'run_hgru_datasize_ablation', dict(seed=23, split_name='train_4', epochs=1, batch_size=32, lr=0.001))
    recorder.before_test()
    metrics = hgru.evaluate(model, test[0], test[1], device=torch.device('cpu'))
    recorder.consider(1, {'accuracy': 0.0}, metrics)
    recorder.finish(1, {'accuracy': 0.0}, metrics)
    compare_saved(directory, dataset)
    cases.append('hgru_rng_replay_forward_only')

    # Real clean-image topology evaluator, not the training smoke's mock.
    drive_root = work / 'drive'
    drive_fixture(drive_root)
    write_rows(drive_root / 'metadata_qati_pairs_v1/test_images.csv', [dict(image_id='test', image_path='test_image.png',
               vessel_mask_path='test_vessel.png', fov_mask_path='test_fov.png')])
    drive = importlib.import_module('training._runners.run_drive_experiment')
    for condition in ('seg_only', 'global_betti', 'shuffled_control', 'qati_only'):
        directory = set_job(work, 'drive_' + condition, 'run_drive_experiment', drive_root, 'drive')
        result = drive.train_single_seed(condition=condition, **short, lr=0.001, weight_decay=0.0001,
                       seg_weight=1., conn_weight=0., aux_weight=0 if condition == 'seg_only' else 0.05,
                       backbone='plain', pretrained=False, image_size=64, max_shift=2, precision='fp32',
                       device=torch.device('cpu'), dataset_root=drive_root)
        evaluated = compare_saved(directory, drive_root)
        best = next(r for r in evaluated if r['selection'] == 'best_val')
        assert best['per_image'] == result['test_seg_topology']['per_image']
        assert best['num_test_units'] == 1
        cases.append('drive_' + condition)
        gc.collect()

    # Strict model state round-trip for the other three DRIVE backbones, offline.
    from models.drive_unet import RETAMultiTaskUNet
    for backbone in ('cnn_s', 'resnet18', 'resnet152'):
        model = RETAMultiTaskUNet(backbone=backbone, pretrained=False)
        rebuilt = build_model(model_spec(model, {}))
        rebuilt.load_state_dict(model.state_dict(), strict=True)
        del model, rebuilt
        gc.collect()
    checkpoint = next((work / 'training').rglob('best_val.pt'))
    # A later parameter update must not mutate the earlier validation selection.
    from models.plain import PlainNet
    directory = set_job(work, 'snapshot_regression', 'run_plain_ssl_3k', dataset)
    model = PlainNet(width=6)
    recorder = CheckpointRecorder(model, 'run_plain_ssl_3k', dict(seed=23, model_type='plain_conn_only', width=6, aux_weight=0.5))
    recorder.before_test()
    recorder.consider(1, {'accuracy': 0.9})
    first_state = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    recorder.consider(2, {'accuracy': 0.1})
    recorder.finish(2, {'accuracy': 0.1})
    best = torch.load(next(directory.rglob('best_val.pt')), weights_only=True)
    final = torch.load(next(directory.rglob('final.pt')), weights_only=True)
    assert best['epoch'] == 1 and final['epoch'] == 2
    assert all(torch.equal(value, best['model_state'][key]) for key, value in first_state.items())
    assert any(not torch.equal(value, final['model_state'][key]) for key, value in first_state.items())
    # Same-source check, source path containment and selection are explicit contracts.
    changed = torch.load(checkpoint, weights_only=True)
    changed['source_hashes'][next(iter(changed['source_hashes']))] = 'wrong'
    changed_path = work / 'changed_source.pt'
    torch.save(changed, changed_path)
    rejected(lambda: evaluate_checkpoint(changed_path, dataset, torch.device('cpu')))
    payload = torch.load(checkpoint, weights_only=True)
    payload['model_state'].pop(next(iter(payload['model_state'])))
    broken = work / 'missing_weight.pt'
    torch.save(payload, broken)
    rejected(lambda: evaluate_checkpoint(broken, dataset, torch.device('cpu')))
    summary_path = dataset / 'summary.json'
    original = summary_path.read_text()
    summary_path.write_text('{}')
    rejected(lambda: evaluate_checkpoint(checkpoint, dataset, torch.device('cpu')))
    summary_path.write_text(original)
    report = dict(ok=True, public_cli_three_seed_two_epoch_roundtrip=True, independent_evaluations=6,
                  sample_std_and_single_seed_checks=True, duplicate_missing_nonfinite_rejections=True,
                  runner_roundtrips=cases, other_drive_backbone_strict_roundtrips=3,
                  validation_only_snapshot_clone_and_source_guard=True,
                  wrong_dataset_and_incomplete_checkpoint_rejected=True,
                  limitations=['Synthetic CPU fixtures; no paper accuracy reproduction',
                               'hGRU full-resolution test is forward-only RNG replay',
                               'Pretrained runner uses scratch weights offline; DRIVE images are synthetic'])
    (work / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    with patch('torch.hub.download_url_to_file', side_effect=RuntimeError('No downloads during checks')):
        main()
