"""Small runtime helpers; never change an experiment's optimiser or loss."""
from __future__ import annotations
import os


def select_device(gpu_id=0, allow_mps=True):
    import torch
    if os.environ.get('RTS_ALLOW_DOWNLOADS') == '0':
        def disabled_download(*args, **kwargs):
            raise RuntimeError('Pretrained weights are not cached. Use --allow-weight-downloads explicitly.')
        torch.hub.download_url_to_file = disabled_download
    requested = os.environ.get('RTS_DEVICE', 'auto')
    if requested == 'cpu':
        device = torch.device('cpu')
    elif requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
        device = torch.device(f'cuda:{gpu_id}')
    elif requested != 'auto':
        raise ValueError('RTS_DEVICE must be auto, cpu or cuda')
    elif torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    elif allow_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    if os.environ.get('RTS_RUNTIME_REPORT'):
        import json
        import platform
        from pathlib import Path
        from importlib.metadata import version
        record = dict(device=str(device), hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else device.type,
                      python=platform.python_version(), torch=torch.__version__, torchvision=version('torchvision'),
                      numpy=version('numpy'), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                      cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic)
        Path(os.environ['RTS_RUNTIME_REPORT']).write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return device
