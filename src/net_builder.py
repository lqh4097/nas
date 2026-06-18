"""
net_builder.py
--------------
把 ArchConfig / genome 翻译成可训练的 PyTorch 模型。

网络结构：
  stem → Stage1 → Stage2 → [Stage3] → head → classifier

每个 Stage 由若干 InvertedResidual 块组成（MobileNetV2 式），
第一个块 stride=2 做下采样，其余 stride=1。
"""

import torch
import torch.nn as nn

from search_space import ArchConfig, decode, random_genome

NUM_CLASSES = 40
STEM_CHANNELS = 16   # stem 固定输出通道数


# ── 基础模块 ───────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力。"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels), nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class InvertedResidual(nn.Module):
    """MobileNetV2 式倒残差块，可选 SE、可选捷径。"""
    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 expand_ratio: int, kernel_size: int, use_se: bool):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, bias=False),
                       nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [
            nn.Conv2d(hidden, hidden, kernel_size, stride,
                      kernel_size // 2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
        ]
        if use_se:
            layers.append(SEBlock(hidden))
        layers += [nn.Conv2d(hidden, out_ch, 1, bias=False),
                   nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out + x if self.use_res else out


def _make_stage(in_ch: int, out_ch: int, n_blocks: int,
                kernel: int, expand: int, use_se: bool) -> nn.Sequential:
    """构建一个 Stage：第一块 stride=2，其余 stride=1。"""
    blocks = [InvertedResidual(in_ch, out_ch, stride=2,
                               expand_ratio=expand, kernel_size=kernel,
                               use_se=use_se)]
    for _ in range(n_blocks - 1):
        blocks.append(InvertedResidual(out_ch, out_ch, stride=1,
                                       expand_ratio=expand, kernel_size=kernel,
                                       use_se=use_se))
    return nn.Sequential(*blocks)


# ── 主网络 ─────────────────────────────────────────────────────────────────────
class NASNet(nn.Module):
    def __init__(self, cfg: ArchConfig, num_classes: int = NUM_CLASSES):
        super().__init__()
        # Stem: 224×224 → 112×112
        self.stem = nn.Sequential(
            nn.Conv2d(3, STEM_CHANNELS, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(STEM_CHANNELS),
            nn.ReLU6(inplace=True),
        )
        # Stages
        in_ch = STEM_CHANNELS
        stage_modules = []
        for s in cfg.stages:
            stage_modules.append(
                _make_stage(in_ch, s.out_channels, s.n_blocks,
                            s.kernel_size, s.expand_ratio, s.use_se)
            )
            in_ch = s.out_channels
        self.stages = nn.Sequential(*stage_modules)

        # Head
        head_ch = max(in_ch * 4, 128)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, 1, bias=False),
            nn.BatchNorm2d(head_ch),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(head_ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        return self.classifier(x)


def build_net(genome: list[int], num_classes: int = NUM_CLASSES) -> NASNet:
    """genome → NASNet，直接可用于训练或推理。"""
    return NASNet(decode(genome), num_classes=num_classes)


# ── 快速验证 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    from search_space import search_space_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(42)
    dummy = torch.randn(2, 3, 224, 224).to(device)

    for i in range(3):
        g = random_genome(rng)
        net = build_net(g).to(device)
        out = net(dummy)
        params = sum(p.numel() for p in net.parameters()) / 1e6
        cfg = decode(g)
        print(f"[{i+1}] n_stages={cfg.n_stages}  "
              f"params={params:.2f}M  output={tuple(out.shape)}")
