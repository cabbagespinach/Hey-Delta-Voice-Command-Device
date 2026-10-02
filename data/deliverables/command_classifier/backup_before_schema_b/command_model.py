#!/usr/bin/env python3
"""
BC-ResNet command classifier: the wakeword project's BC-ResNet (../../model/bc_resnet.py, Kim et al. 2021) with a
30-way head instead of the single wakeword logit, on the 5 s command front end ([B, 1, 40, 497] -> [B, 30] logits).

CommandNet bundles the front end (log-mel + frozen normalisation) with the network, so the exported model takes the
raw 5 s capture, exactly as the wakeword ONNX does.
"""
from pathlib import Path
import sys

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "model"))
sys.path.insert(0, str(HERE.parent / "preprocessing"))
import bc_resnet                                                          # noqa: E402
import command_preprocessing as cp                                        # noqa: E402


class BCResNetCommands(bc_resnet.BCResNet):
    def __init__(self, n_classes: int, tau: float = 3.0, dropout: float = 0.1):
        super().__init__(tau=tau, dropout=dropout)
        last = self.head[-1]
        self.head[-1] = nn.Conv2d(last.in_channels, n_classes, 1)
        self.n_classes = n_classes

    def forward(self, x):
        return self.head(self.blocks(self.stem(x))).reshape(x.shape[0], self.n_classes)


class CommandNet(nn.Module):
    """[B, 80000] waveform (5 s, 16 kHz) -> [B, n_classes] logits."""

    def __init__(self, n_classes: int, tau: float = 3.0, normalize=True):
        super().__init__()
        self.front = cp.CommandPreprocessor.from_files(normalize=normalize)
        self.net = BCResNetCommands(n_classes, tau)

    def forward(self, wave, feature_hook=None):
        x = self.front(wave)
        if feature_hook is not None:                 # e.g. SpecAugment during training
            x = feature_hook(x)
        return self.net(x)


if __name__ == "__main__":
    for tau in (1, 3, 6):
        m = BCResNetCommands(30, tau).eval()
        with torch.no_grad():
            y = m(torch.zeros(2, 1, 40, 497))
        print(f"tau {tau}: {bc_resnet.count_params(m):,} parameters, output {tuple(y.shape)}")
