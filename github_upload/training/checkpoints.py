"""Evaluation-only checkpoints. No optimiser/resume state and no training RNG draws."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEYS = ('seed', 'model_type', 'model_name', 'width', 'config_name', 'cfg',
               'split_name', 'epochs', 'batch_size', 'lr', 'weight_decay',
               'warmup_epochs', 'min_lr', 'aux_weight', 'augmentation', 'precision',
               'condition', 'shuffle_delta', 'optimizer_type', 'scheduler_type',
               'grad_clip', 'backbone', 'pretrained', 'image_size', 'max_shift',
               'seg_weight', 'conn_weight', 'bridge_thickness')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_spec(model, config):
    name = type(model).__name__
    if name in ('PlainNet', 'MultiTaskPlain', 'CNNSmall', 'MultiTaskCNNSmall'):
        width = next(m.out_channels for m in model.modules() if isinstance(m, torch.nn.Conv2d))
        kwargs = dict(width=width, in_channels=1)
    elif name in ('MultiTaskResNet', 'ResNet'):
        kwargs = dict(variant=model.variant, pretrained=False, in_channels=1)
    elif name in ('MultiTaskSwinT', 'SwinT'):
        kwargs = dict(pretrained=False, in_channels=1, image_size=model.image_size)
    elif name == 'SwinCapacityMultiTask':
        kwargs = {key: config['cfg'][key] for key in ('embed_dim', 'depths', 'num_heads', 'window_size')}
        kwargs.update(in_channels=1, image_size=model.image_size, stochastic_depth_prob=0.0)
    elif name == 'HGRU':
        kwargs = dict(channels=25, recurrent_kernel_size=15, timesteps=8, num_classes=2)
    elif name == 'MultiTaskUNet':
        kwargs = dict(backbone=model.backbone_name, pretrained=False, in_channels=3,
                      aux_topology=model.aux_topology, pixel_aligned=model.pixel_aligned,
                      roi_source=model.roi_source, roi_head_type=model.roi_head_type)
    else:
        raise ValueError(f'No release checkpoint constructor for {name}')
    return dict(name=name, kwargs=kwargs)


def build_model(spec):
    # Never dynamically import a module named by a checkpoint.
    from models.plain import PlainNet, MultiTaskPlain
    from models.cnn_s import CNNSmall, MultiTaskCNNSmall
    from models.resnet import ResNet, MultiTaskResNet
    from models.swin import SwinT, MultiTaskSwinT, SwinCapacityMultiTask
    from models.hgru import HGRU
    from models.drive_unet import MultiTaskUNet
    constructors = {c.__name__: c for c in (PlainNet, MultiTaskPlain, CNNSmall, MultiTaskCNNSmall,
                    ResNet, MultiTaskResNet, SwinT, MultiTaskSwinT, SwinCapacityMultiTask, HGRU, MultiTaskUNet)}
    kwargs = dict(spec['kwargs'])
    if 'pretrained' in kwargs:
        kwargs['pretrained'] = False  # Full trained state follows; no weight download.
    if spec['name'] not in constructors:
        raise ValueError('Unsupported checkpoint model')
    return constructors[spec['name']](**kwargs)


def rng_state(device):
    state = np.random.get_state()
    return dict(torch=torch.get_rng_state().clone(), python=random.getstate(),
                numpy=[state[0], torch.tensor(state[1].astype(np.int64)), int(state[2]), int(state[3]), float(state[4])],
                cuda=torch.cuda.get_rng_state(device).cpu() if device.type == 'cuda' else None,
                device_type=device.type)


def restore_rng(state, device):
    torch.set_rng_state(state['torch'].cpu())
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], n[1].numpy().astype(np.uint32), n[2], n[3], n[4]))
    if device.type == 'cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


class CheckpointRecorder:
    """Enabled by the public launcher; direct historical calls remain unchanged."""
    def __init__(self, model, runner, arguments):
        directory = os.environ.get('RTS_CHECKPOINT_ROOT')
        self.enabled = bool(directory)
        self.best = None
        self.test_rng = None
        if not self.enabled:
            return
        self.model = model
        self.device = next(model.parameters()).device
        self.config = {k: arguments[k] for k in CONFIG_KEYS if k in arguments}
        self.config = json.loads(json.dumps(self.config))
        self.runner = runner.rsplit('.', 1)[-1]
        self.spec = model_spec(model, self.config)
        job_path = os.environ.get('RTS_JOB_SPEC')
        self.job = json.loads(Path(job_path).read_text(encoding='utf-8')) if job_path else {}
        # Evaluation checkpoints may be shared: do not embed workstation paths.
        self.job = {k: self.job[k] for k in ('dataset', 'protocol', 'dataset_provenance', 'options') if k in self.job}
        self.job['options'] = {k: v for k, v in self.job.get('options', {}).items() if k not in
                               ('root', 'dataset-root', 'out-dir', 'config', 'pretrained-backbone-path')}
        identity = dict(runner=self.runner, config=self.config, model=self.spec)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        self.directory = Path(directory) / f"s{self.config['seed']}_{digest}"
        if self.directory.exists():
            raise FileExistsError(f'Checkpoint run already exists: {self.directory}')
        self.directory.mkdir(parents=True)
        self.source_hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for folder in
                              ('models', 'data_loading', 'training') for p in (ROOT / folder).rglob('*.py')}
        if self.runner == 'run_drive_experiment':
            for relative in ('evaluation/drive.py', 'evaluation/topology.py'):
                self.source_hashes[relative] = sha256(ROOT / relative)

    def before_test(self):
        if self.enabled:
            self.test_rng = rng_state(self.device)

    def _snapshot(self, epoch, val_metrics, test_metrics, selection):
        return dict(format='rts-evaluation-checkpoint-v1', runner=self.runner,
                    model_spec=self.spec, config=self.config, job=self.job,
                    seed=self.config['seed'], epoch=epoch, selection=selection,
                    selection_metric='validation_dice' if self.runner == 'run_drive_experiment' else 'validation_accuracy',
                    val_metrics=val_metrics, recorded_test_metrics=test_metrics,
                    model_state={k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                    test_rng=self.test_rng, device_type=self.device.type,
                    source_hashes=self.source_hashes,
                    parameter_count=sum(p.numel() for p in self.model.parameters()))

    def consider(self, epoch, val_metrics, test_metrics=None):
        if not self.enabled:
            return
        key = 'dice' if self.runner == 'run_drive_experiment' else 'accuracy'
        score = float(val_metrics[key])
        if not math.isfinite(score):
            raise ValueError('Non-finite validation selection metric')
        if self.best is None or score > self.best['val_metrics'][key]:
            self.best = self._snapshot(epoch, val_metrics, test_metrics, 'best_val')

    def finish(self, epoch, val_metrics, test_metrics=None):
        if not self.enabled:
            return {}
        if self.best is None:
            raise ValueError('No validation-selected checkpoint')
        final = self._snapshot(epoch, val_metrics, test_metrics, 'final')
        files = {}
        for selection, payload in [('best_val', self.best), ('final', final)]:
            path = self.directory / (selection + '.pt')
            with path.open('xb') as stream:
                torch.save(payload, stream)
            files[selection] = dict(path=path.name, sha256=sha256(path), epoch=payload['epoch'])
        (self.directory / 'checkpoints.json').write_text(json.dumps(files, indent=2) + '\n', encoding='utf-8')
        self.best = None
        return files
