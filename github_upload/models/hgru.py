"""Paper-faithful PyTorch hGRU for Pathfinder binary connectivity classification.

Ported from the reference implementation released by Linsley et al. (NeurIPS 2018).

The previous revision of this file diverged from that reference in four ways. It was still
trainable -- it could memorise a 1,000-image subset -- but it was unstable and far weaker
than the published circuit: one recorded run diverged (validation loss 0.69 -> 5.35 within
four epochs), and a full 100-epoch run reached only 60.4% test accuracy on length-6
contours. For the record, and so the mistakes are not reintroduced:

1. It used a single shared horizontal kernel for both inhibition and excitation; the
   circuit needs separate ``w_gate_inh`` and ``w_gate_exc``, orthogonally initialised,
   each constrained towards channel symmetry by a gradient hook.
2. It applied no normalisation inside the recurrence. The reference applies four
   BatchNorm2d layers *per timestep* -- this is the "recurrent batchnorm" the paper
   describes, and an 8-step recurrence does not train without it.
3. It misapplied Chronos initialisation. Chronos initialises a *gate bias* that is then
   passed through a sigmoid, yielding a gate in (0, 1). The old code instead multiplied
   the hidden state by a raw scalar ``eta_t = -log(t+1)``, which is exactly 0 at t=0 and
   negative for every t >= 1: the state was annihilated on the first timestep and then
   sign-flipped and amplified on every subsequent one.
4. Its output update replaced the reference's convex gate combination
   ``(1 - g2) * h2 + g2 * h_cand`` with that same spurious ``eta_t`` factor.

One deliberate deviation from the vendored reference: the reference writes
``self.u1_gate.bias.data.log()``, which is not an in-place operation, so the Chronos log
is silently discarded there and the bias remains uniform on [1, T-1]. We apply the log
(``log_()``), matching the intent described in the paper.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

DEFAULT_GABOR_PATH = Path(__file__).resolve().parent / "assets" / "gabors_for_contours_7.npy"
DEFAULT_GABOR_SHA256 = "4f0482e5c032d0c52fea89ca24cbe5d059bddf94ca7de9ad2dd826a968ecd0f4"
UPSTREAM_GABOR_URL = (
    "https://raw.githubusercontent.com/serre-lab/hgru_share/"
    "4ac92bd12b3c91092415ee78530f1e4a81c2f2c0/weights/gabors_for_contours_7.npy"
)



class HGRUCell(nn.Module):
    """Horizontal Gated Recurrent Unit (Linsley et al., NeurIPS 2018).

    One recurrent step comprises an inhibitory stage (suppressing the feedforward drive
    via gated horizontal convolution) and an excitatory stage (facilitating it), combined
    through a learned update gate. With ``batchnorm=True`` the cell keeps
    ``4 * timesteps`` independent BatchNorm2d layers, one per normalisation site per
    timestep, exactly as in the reference.
    """

    def __init__(
        self,
        channels: int = 25,
        kernel_size: int = 15,
        timesteps: int = 8,
        batchnorm: bool = True,
        batchnorm_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve spatial dimensions.")
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2 for Chronos initialization.")

        self.channels = channels
        self.kernel_size = kernel_size
        self.timesteps = timesteps
        self.padding = kernel_size // 2
        self.batchnorm = batchnorm

        # Gain (inhibitory) and mix (excitatory/update) gates.
        self.u1_gate = nn.Conv2d(channels, channels, kernel_size=1)
        self.u2_gate = nn.Conv2d(channels, channels, kernel_size=1)

        # Separate inhibitory and excitatory horizontal kernels.
        self.w_gate_inh = nn.Parameter(torch.empty(channels, channels, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(torch.empty(channels, channels, kernel_size, kernel_size))

        # Per-channel interaction coefficients.
        self.alpha = nn.Parameter(torch.empty(channels, 1, 1))
        self.gamma = nn.Parameter(torch.empty(channels, 1, 1))
        self.kappa = nn.Parameter(torch.empty(channels, 1, 1))
        self.omega = nn.Parameter(torch.empty(channels, 1, 1))
        self.mu = nn.Parameter(torch.empty(channels, 1, 1))

        if batchnorm:
            self.bn = nn.ModuleList(
                [nn.BatchNorm2d(channels, eps=batchnorm_eps) for _ in range(4 * timesteps)]
            )
            self.n = None
        else:
            self.bn = None
            # Learned per-timestep gain, used only in the normalisation-free variant.
            self.n = nn.Parameter(torch.randn(timesteps, 1, 1))

        self.reset_parameters()

        # Constrain the horizontal kernels towards symmetry in the channel dimensions.
        self.w_gate_inh.register_hook(lambda grad: (grad + grad.transpose(0, 1)) * 0.5)
        self.w_gate_exc.register_hook(lambda grad: (grad + grad.transpose(0, 1)) * 0.5)

    def reset_parameters(self) -> None:
        init.orthogonal_(self.w_gate_inh)
        init.orthogonal_(self.w_gate_exc)
        init.orthogonal_(self.u1_gate.weight)
        init.orthogonal_(self.u2_gate.weight)

        init.constant_(self.alpha, 0.1)
        init.constant_(self.gamma, 1.0)
        init.constant_(self.kappa, 0.5)
        init.constant_(self.omega, 0.5)
        init.constant_(self.mu, 1.0)

        if self.bn is not None:
            for bn in self.bn:
                init.constant_(bn.weight, 0.1)
                init.zeros_(bn.bias)

        # Chronos initialisation: bias_1 ~ log(U(1, T-1)) pre-sigmoid, bias_2 = -bias_1,
        # spreading the gates' initial time constants across the unrolled window.
        with torch.no_grad():
            init.uniform_(self.u1_gate.bias, 1.0, float(self.timesteps) - 1.0)
            self.u1_gate.bias.log_()
            self.u2_gate.bias.copy_(-self.u1_gate.bias)

    def _initial_state(self, external_drive: torch.Tensor) -> torch.Tensor:
        state = torch.empty_like(external_drive)
        init.xavier_normal_(state)
        return state

    def forward_states(
        self,
        external_drive: torch.Tensor,
        capture_timesteps: tuple[int, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Return recurrent states after selected one-based timesteps."""

        if not capture_timesteps:
            raise ValueError("capture_timesteps must not be empty.")
        if len(set(capture_timesteps)) != len(capture_timesteps):
            raise ValueError("capture_timesteps must be unique.")
        if any(step < 1 or step > self.timesteps for step in capture_timesteps):
            raise ValueError(
                f"capture_timesteps must lie in [1, {self.timesteps}]."
            )
        requested = set(capture_timesteps)
        captured: dict[int, torch.Tensor] = {}
        h2 = self._initial_state(external_drive)

        for t in range(self.timesteps):
            if self.batchnorm:
                # Inhibitory stage.
                g1 = torch.sigmoid(self.bn[t * 4 + 0](self.u1_gate(h2)))
                c1 = self.bn[t * 4 + 1](
                    F.conv2d(h2 * g1, self.w_gate_inh, padding=self.padding)
                )
                h1 = F.relu(external_drive - F.relu(c1 * (self.alpha * h2 + self.mu)))

                # Excitatory stage.
                g2 = torch.sigmoid(self.bn[t * 4 + 2](self.u2_gate(h1)))
                c2 = self.bn[t * 4 + 3](
                    F.conv2d(h1, self.w_gate_exc, padding=self.padding)
                )
                candidate = F.relu(
                    self.kappa * h1 + self.gamma * c2 + self.omega * (h1 * c2)
                )
                h2 = (1.0 - g2) * h2 + g2 * candidate
            else:
                g1 = torch.sigmoid(self.u1_gate(h2))
                c1 = F.conv2d(h2 * g1, self.w_gate_inh, padding=self.padding)
                h1 = torch.tanh(external_drive - c1 * (self.alpha * h2 + self.mu))

                g2 = torch.sigmoid(self.u2_gate(h1))
                c2 = F.conv2d(h1, self.w_gate_exc, padding=self.padding)
                candidate = torch.tanh(
                    self.kappa * (h1 + self.gamma * c2)
                    + self.omega * (h1 * (self.gamma * c2))
                )
                h2 = self.n[t] * ((1.0 - g2) * h2 + g2 * candidate)

            step = t + 1
            if step in requested:
                captured[step] = h2

        return tuple(captured[step] for step in capture_timesteps)

    def forward(self, external_drive: torch.Tensor) -> torch.Tensor:
        return self.forward_states(external_drive, (self.timesteps,))[0]


class HGRU(nn.Module):
    """Single-layer hGRU classifier for Pathfinder connectivity."""

    def __init__(
        self,
        channels: int = 25,
        recurrent_kernel_size: int = 15,
        timesteps: int = 8,
        num_classes: int = 2,
        gabor_path: str | Path = DEFAULT_GABOR_PATH,
        gabor_trainable: bool = True,
        batchnorm: bool = True,
        batchnorm_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes

        self.gabor = nn.Conv2d(1, channels, kernel_size=7, padding=3, bias=True)
        self._init_gabor_weights(Path(gabor_path).resolve())
        self.gabor.weight.requires_grad_(gabor_trainable)
        self.gabor.bias.requires_grad_(gabor_trainable)

        self.recurrent_cell = HGRUCell(
            channels=channels,
            kernel_size=recurrent_kernel_size,
            timesteps=timesteps,
            batchnorm=batchnorm,
            batchnorm_eps=batchnorm_eps,
        )

        self.output_norm = nn.BatchNorm2d(channels, eps=batchnorm_eps)
        self.readout = nn.Conv2d(channels, num_classes, kernel_size=1)
        # AdaptiveMaxPool generalises the reference's fixed MaxPool2d(150) to any input size.
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.class_norm = nn.BatchNorm2d(num_classes, eps=batchnorm_eps)
        self.classifier = nn.Linear(num_classes, num_classes)
        self._reset_readout_parameters()

    def _init_gabor_weights(self, path: Path) -> None:
        # The upstream .npy contains a pickled dict. Only unpickle the byte-exact
        # verified reference, never an arbitrary or silently substituted file.
        if not path.is_file():
            raise FileNotFoundError(
                "Required hGRU Gabor initialisation is not bundled. "
                f"Obtain {UPSTREAM_GABOR_URL}, place it at {path}, and verify "
                f"SHA-256 {DEFAULT_GABOR_SHA256}."
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != DEFAULT_GABOR_SHA256:
            raise ValueError("hGRU Gabor file differs from the expected upstream reference")
        payload = np.load(path, allow_pickle=True, encoding="latin1")
        tensorflow_weights = payload.item()["s1"][0]
        weights = np.asarray(tensorflow_weights).transpose(3, 2, 0, 1).copy()
        if tuple(weights.shape) != tuple(self.gabor.weight.shape):
            raise ValueError("The reference hGRU Gabor bank requires 25 channels")
        with torch.no_grad():
            self.gabor.weight.copy_(torch.from_numpy(weights))
            self.gabor.bias.zero_()

    def _reset_readout_parameters(self) -> None:
        init.xavier_normal_(self.readout.weight)
        init.zeros_(self.readout.bias)
        init.xavier_normal_(self.classifier.weight)
        init.zeros_(self.classifier.bias)

    def _drive(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(f"Expected images shaped [N, 1, H, W], got {tuple(images.shape)}")
        return self.gabor(images).square()

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Spatially averaged hGRU state, for use by multi-task heads."""
        state = self.recurrent_cell(self._drive(images))
        output = self.output_norm(state)
        return F.adaptive_avg_pool2d(output, (1, 1)).flatten(start_dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        state = self.recurrent_cell(self._drive(images))
        output = self.output_norm(state)
        output = F.leaky_relu(self.readout(output))
        output = self.global_pool(output)
        output = self.class_norm(output)
        return self.classifier(output.flatten(start_dim=1))


class MultiTaskHGRU(nn.Module):
    """Multi-task hGRU supporting binary connectivity and topological auxiliary regression."""

    def __init__(
        self,
        channels: int = 25,
        recurrent_kernel_size: int = 15,
        timesteps: int = 8,
        num_classes: int = 2,
        gabor_path: str | Path = DEFAULT_GABOR_PATH,
        gabor_trainable: bool = True,
        batchnorm: bool = True,
        batchnorm_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        from .resnet import NonNegativeRegressionHead

        self.hgru = HGRU(
            channels=channels,
            recurrent_kernel_size=recurrent_kernel_size,
            timesteps=timesteps,
            num_classes=num_classes,
            gabor_path=gabor_path,
            gabor_trainable=gabor_trainable,
            batchnorm=batchnorm,
            batchnorm_eps=batchnorm_eps,
        )
        self.head_betti0 = NonNegativeRegressionHead(channels)
        self.head_delta_betti0 = NonNegativeRegressionHead(channels)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.hgru.forward_features(images)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        drive = self.hgru._drive(images)
        state = self.hgru.recurrent_cell(drive)
        output = self.hgru.output_norm(state)

        # Connectivity classification path (paper-faithful readout)
        readout_out = F.leaky_relu(self.hgru.readout(output))
        pool_out = self.hgru.global_pool(readout_out)
        norm_out = self.hgru.class_norm(pool_out)
        logits_conn = self.hgru.classifier(norm_out.flatten(start_dim=1))

        # Auxiliary topological regression path (pooled recurrent state)
        pooled = F.adaptive_avg_pool2d(output, (1, 1)).flatten(start_dim=1)
        pred_betti = self.head_betti0(pooled)
        pred_delta = self.head_delta_betti0(pooled)

        return logits_conn, pred_betti, pred_delta
