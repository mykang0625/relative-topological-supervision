"""Bounded CPU training checks on synthetic fixtures, not accuracy reproduction."""
from __future__ import annotations
import argparse
import ast
import copy
import gc
import importlib
import json
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['RTS_DEVICE'] = 'cpu'
os.environ['RTS_ALLOW_DOWNLOADS'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from training import run


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    if isinstance(value, float):
        return np.isfinite(value)
    if isinstance(value, torch.Tensor):
        return torch.isfinite(value).all().item()
    return True


def check_plans(work):
    config = run.read_presets(run.ROOT / 'configs/experiments.yaml')
    checked = 0
    for name in config['experiments']:
        jobs = run.build_plan(config, name, work / 'absent_data', work / 'absent_output',
                              spt_dir=work / 'absent_spt' if name in ('spt_label_only', 'spt_relative') else None)
        for job in jobs:
            source = (run.ROOT / 'training/_runners' / (job['runner'] + '.py')).read_text(encoding='utf-8')
            options = set()
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                    options.update(a.value[2:] for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith('--'))
            assert set(job['options']) <= options, (name, set(job['options']) - options)
            run.command(job)
            checked += 1
    assert not (work / 'absent_output').exists()
    for kwargs in (dict(seeds=[23, 23]), dict(epochs=0), dict(models=['nonexistent'])):
        try:
            run.build_plan(config, 'data_scaling', work, work / 'output', **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid override accepted')
    for output, dataset in [(work, work / 'data'), (work / 'data/out', work / 'data'), (run.ROOT / 'models/out', work / 'data')]:
        try:
            run.validate_output(output, dataset)
        except (ValueError, FileExistsError):
            pass
        else:
            raise AssertionError('Unsafe output accepted')
    return dict(presets=len(config['experiments']), planned_jobs=checked, dry_run_no_writes=True, invalid_overrides_and_outputs_rejected=True)


def check_training(work):
    torch.set_num_threads(1)
    modules = {name: importlib.import_module('training._runners.' + name) for name in run.RUNNERS}
    torch.manual_seed(42)
    # The recorded batch augmentation explicitly crops to 128x128.
    x = torch.randint(0, 256, (4, 128, 128), dtype=torch.uint8)
    y = torch.tensor([0, 1, 0, 1])
    d = torch.tensor([0., 1., 2., 3.])
    data = dict(X_train=x, Y_train=y, D_train=d, X_val=x, Y_val=y, D_val=d,
                X_test=x, Y_test=y, D_test=d)
    short = dict(seed=23, epochs=2, batch_size=2)
    results = []
    def record(name, result):
        assert finite(result), name
        if 'best_val_epoch' in result:
            assert result['best_val_epoch'] in (1, 2), name
        results.append(dict(name=name, finite=True, epochs=2))
        gc.collect()
    plain = modules['run_plain_ssl_3k']
    for condition in ('connectivity', 'qati_delta', 'shuffled_qati_delta', 'betti0'):
        args = dict(data)
        if condition == 'betti0':
            args.update(D_train=d + 40, D_val=d + 40, D_test=d + 40)
        before = args['D_train'].clone()
        result = plain.train_single_run(**args, **short, model_type='plain_conn_only' if condition == 'connectivity' else 'multitask_plain',
                                        width=6, bridge_thickness=3., condition=condition, shuffle_delta=condition == 'shuffled_qati_delta')
        assert torch.equal(before, args['D_train'])
        record('plain_' + condition, result)
    for model in ('plain', 'cnn_s', 'swin_t'):
        record('scaling_' + model, modules['run_datasize_ablation'].train_single_run(**data, **short, model_name=model, split_name='fixture', precision='fp32'))
    for model, width in [('plain', 6), ('cnn_s', 3)]:
        record('capacity_' + model, modules['run_model_size_ablation_16k'].train_single_run(**data, **short, model_name=model, width=width, precision='fp32'))
    swin = modules['run_swin_capacity_ablation']
    record('capacity_swin', swin.train_single_run(**data, **short, config_name='swin_12_273k', cfg=swin.SWIN_CONFIGS['swin_12_273k'], precision='fp32'))
    # Exercise this runner offline with scratch initialisation; cached weight parity
    # is covered separately by the model relocation audit, not this CPU smoke.
    pre = modules['run_pretrained_backbones_4k']
    from models.resnet import MultiTaskResNet
    with patch.object(pre, 'build_pretrained_model', side_effect=lambda _: MultiTaskResNet(variant='resnet18', pretrained=False)):
        record('pretrained_runner_scratch_smoke', pre.train_single_run(**data, **short, model_name='resnet18', split_name='fixture', precision='fp32'))
    hgru = modules['run_hgru_datasize_ablation']
    with patch.object(hgru, 'load_split_data', return_value=(x[:, :16, :16], y)):
        record('hgru_label_only', hgru.train_single_run(**short, split_name='train_4', dataset_root=work, out_dir=work / 'hgru'))
    spt = modules['pretrain_plain_spt'].pretrain_single_seed(x, **short, width=6)
    record('spt_pretrain', spt)
    checkpoint = work / 'spt_encoder.pt'
    torch.save(spt['encoder_state_dict'], checkpoint)
    record('spt_strict_transfer', plain.train_single_run(**data, **short, model_type='plain_conn_only', width=6,
            bridge_thickness=3., pretrained_backbone_path=checkpoint))
    bad = work / 'incomplete_encoder.pt'
    torch.save({'not_a_weight': torch.zeros(1)}, bad)
    try:
        plain.train_single_run(**data, **short, model_type='plain_conn_only', width=6, bridge_thickness=3., pretrained_backbone_path=bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Incomplete SPT transfer was silently accepted')
    # DRIVE real loader/model/loss/backward/checkpoint with small synthetic rasters.
    # Full retinal topology evaluation is intentionally mocked: it is a separate release stage.
    from data_loading.test_loading import drive_fixture
    drive_root = work / 'drive_fixture'
    drive_root.mkdir()
    drive_fixture(drive_root)
    drive = modules['run_drive_experiment']
    for condition in ('seg_only', 'global_betti', 'shuffled_control', 'qati_only'):
        with patch.object(drive, 'evaluate_test_image_segmentation', return_value={'dice_mean': 0., 'cldice_mean': 0.}):
            result = drive.train_single_seed(condition=condition, **short, lr=.001, weight_decay=.0001,
                      seg_weight=1., conn_weight=0., aux_weight=0. if condition == 'seg_only' else .05,
                      backbone='plain', pretrained=False, image_size=64, max_shift=2, precision='fp32',
                      device=torch.device('cpu'), dataset_root=drive_root, save_checkpoint=True, out_dir=work / 'drive')
        state = torch.load(work / 'drive' / f'{condition}_s23_best.pt', weights_only=True)
        assert state['condition'] == condition
        record('drive_' + condition, result)
    # Force the first validation epoch to be best, then verify that saved tensors
    # are that epoch's tensors, not aliases changed by the second CPU update.
    original_evaluate = drive.evaluate
    snapshots = []
    def select_first(model, loader, condition, device, precision='bf16'):
        metrics = original_evaluate(model, loader, condition, device, precision)
        snapshots.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        metrics['dice'] = .9 if len(snapshots) == 1 else .1
        return metrics
    with patch.object(drive, 'evaluate', side_effect=select_first), patch.object(drive, 'evaluate_test_image_segmentation', return_value={}):
        drive.train_single_seed(condition='seg_only', **short, lr=.001, weight_decay=.0001,
                  seg_weight=1., conn_weight=0., aux_weight=0., backbone='plain', pretrained=False,
                  image_size=64, max_shift=2, precision='fp32', device=torch.device('cpu'),
                  dataset_root=drive_root, save_checkpoint=True, out_dir=work / 'best_snapshot')
    saved = torch.load(work / 'best_snapshot/seg_only_s23_best.pt', weights_only=True)['model_state']
    assert all(torch.equal(saved[k], snapshots[0][k]) for k in saved)
    assert any(not torch.equal(snapshots[0][k], snapshots[1][k]) for k in saved)
    # Regression test for CPU best-state aliasing: later model updates cannot alter a snapshot.
    model = torch.nn.Linear(2, 1)
    snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    old = snapshot['weight'].clone()
    with torch.no_grad():
        model.weight.add_(1)
    assert torch.equal(snapshot['weight'], old)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path, help='New directory for synthetic fixtures/checkpoints; no existing data is used')
    args = parser.parse_args()
    work = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    plans = check_plans(work)
    with patch('torch.hub.download_url_to_file', side_effect=RuntimeError('No downloads in smoke checks')):
        cases = check_training(work)
    report = dict(ok=True, plans=plans, training_cases=cases, device='cpu',
                  scope='Two-epoch synthetic smoke only. No paper accuracy or GPU equivalence claim. DRIVE topology evaluation mocked; pretrained runner uses scratch weights.')
    (work / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
