#!/usr/bin/env python3
"""
BC-ResNet for the "Hey Delta" wakeword: one logit per 1.5 s window.

Kim, Chang, Lee & Sung, "Broadcasted Residual Learning for Efficient Keyword Spotting", Interspeech 2021
(arXiv:2106.04140; reference code github.com/Qualcomm-AI-research/bcresnet). The layout follows the paper:
40 log-mel bins in, 12 BC-ResBlocks in 4 stages (2, 2, 4, 4), channel widths 8/12/16/20 x tau after a
16 x tau 5x5 stem, frequency stride 2 in the stem and at the start of stages 2 and 3, temporal dilation
1/2/4/8, SubSpectral Norm with 5 sub-bands, then a frequency-collapsing depthwise conv, a 1x1 conv to
32 x tau channels, global average pooling. The only change is the head: 1 output (wakeword logit) instead
of 12 classes.

A BC-ResBlock computes
    y = ReLU(x + f2(x) + broadcast(f1(avgpool_freq(f2(x)))))
where f2 is a frequency-depthwise 3x1 conv + SubSpectral Norm (2-D, cheap) and f1 is a temporal
depthwise 1x3 conv + BN + SiLU + 1x1 conv + dropout on the frequency-averaged map (1-D, cheaper). A
transition block (channel change) replaces the identity with a 1x1 conv and does not add x.

Input: normalised log-mel features [B, 1, 40, T] (the project front end gives T = 147).
Output: logits [B]. Every op exports to ONNX (opset 17).
"""
from __future__ import annotations

import torch
from torch import nn

N_MELS = 40


class SubSpectralNorm(nn.Module):
    """BatchNorm computed separately for each of S frequency sub-bands (Chang et al. 2021)."""

    def __init__(self, channels: int, sub_bands: int = 5):
        super().__init__()
        self.S = sub_bands
        self.bn = nn.BatchNorm2d(channels * sub_bands)

    def forward(self, x):
        b, c, f, t = x.shape
        x = self.bn(x.reshape(b, c * self.S, f // self.S, t))
        return x.reshape(b, c, f, t)


class BCResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, freq_stride: int = 1, dilation: int = 1, dropout: float = 0.1,
                 sub_bands: int = 5):
        super().__init__()
        self.transition = in_c != out_c
        self.pre = (nn.Sequential(nn.Conv2d(in_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU())
                    if self.transition else nn.Identity())
        self.f2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, (3, 1), stride=(freq_stride, 1), padding=(1, 0), groups=out_c, bias=False),
            SubSpectralNorm(out_c, sub_bands))
        self.f1 = nn.Sequential(
            nn.Conv2d(out_c, out_c, (1, 3), padding=(0, dilation), dilation=(1, dilation), groups=out_c, bias=False),
            nn.BatchNorm2d(out_c), nn.SiLU(),
            nn.Conv2d(out_c, out_c, 1, bias=False), nn.Dropout2d(dropout))
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.f2(self.pre(x))
        y = h + self.f1(h.mean(dim=2, keepdim=True))           # broadcast the 1-D branch over frequency
        if not self.transition:
            y = y + x
        return self.act(y)


class BCResNet(nn.Module):
    def __init__(self, tau: float = 1.0, n_mels: int = N_MELS, dropout: float = 0.1):
        super().__init__()
        if n_mels != 40:
            raise ValueError("the stem/stride layout assumes 40 mel bins (40 -> 20 -> 10 -> 5 for 5 sub-bands)")
        c = lambda k: int(round(k * tau))
        widths, blocks, strides, dils = [c(8), c(12), c(16), c(20)], [2, 2, 4, 4], [1, 2, 2, 1], [1, 2, 4, 8]
        self.tau = tau
        self.stem = nn.Sequential(nn.Conv2d(1, c(16), 5, stride=(2, 1), padding=2, bias=False),
                                  nn.BatchNorm2d(c(16)), nn.ReLU())
        layers, in_c = [], c(16)
        for w, n, s, d in zip(widths, blocks, strides, dils):
            for i in range(n):
                layers.append(BCResBlock(in_c, w, s if i == 0 else 1, d, dropout))
                in_c = w
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv2d(in_c, in_c, 5, padding=(0, 2), groups=in_c, bias=False),     # 5 freq bins -> 1
            nn.Conv2d(in_c, c(32), 1, bias=False), nn.BatchNorm2d(c(32)), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c(32), 1, 1))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x))).reshape(-1)


# No-argument classes for `--model-class bc_resnet:BCResNet3` in the evaluators.
class BCResNet1(BCResNet):
    def __init__(self):
        super().__init__(tau=1)


class BCResNet3(BCResNet):
    def __init__(self):
        super().__init__(tau=3)


class BCResNet6(BCResNet):
    def __init__(self):
        super().__init__(tau=6)


MODELS = {"bcresnet1": BCResNet1, "bcresnet3": BCResNet3, "bcresnet6": BCResNet6}


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for name, cls in MODELS.items():
        m = cls().eval()
        with torch.no_grad():
            y = m(torch.zeros(2, 1, 40, 147))
        print(f"{name}: {count_params(m):,} parameters, output {tuple(y.shape)}")
