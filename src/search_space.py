"""
search_space.py
---------------
NAS 搜索空间定义：16 维整数编码 ↔ 网络架构描述。

编码结构（MobileNet 式倒残差块）：
  d0       全局: n_stages ∈ {2, 3}
  d1-d5    Stage 1: [n_blocks, channels, kernel, expand, use_se]
  d6-d10   Stage 2: [n_blocks, channels, kernel, expand, use_se]
  d11-d15  Stage 3: [n_blocks, channels, kernel, expand, use_se]
           (n_stages=2 时 Stage 3 不接入主干，由 net_builder 忽略)

搜索空间大小：2 × (3×4×2×2×2)^3 ≈ 177 万
"""

import random
from dataclasses import dataclass

import numpy as np

# ── 维度定义 [lo, hi] 闭区间 ───────────────────────────────────────────────────
DIM_RANGES = [
    (0, 1),  # d0  n_stages
    (0, 2),  # d1  s1 n_blocks
    (0, 3),  # d2  s1 channels
    (0, 1),  # d3  s1 kernel
    (0, 1),  # d4  s1 expand
    (0, 1),  # d5  s1 use_se
    (0, 2),  # d6  s2 n_blocks
    (0, 3),  # d7  s2 channels
    (0, 1),  # d8  s2 kernel
    (0, 1),  # d9  s2 expand
    (0, 1),  # d10 s2 use_se
    (0, 2),  # d11 s3 n_blocks
    (0, 3),  # d12 s3 channels
    (0, 1),  # d13 s3 kernel
    (0, 1),  # d14 s3 expand
    (0, 1),  # d15 s3 use_se
]

NDIM = len(DIM_RANGES)  # 16

# ── 取值映射表 ─────────────────────────────────────────────────────────────────
_N_STAGES  = [2, 3]
_N_BLOCKS  = [1, 2, 3]
_KERNELS   = [3, 5]
_EXPANDS   = [3, 6]
_CHANNELS  = [
    [16, 24, 32, 48],     # Stage 1（浅层，通道少）
    [32, 48, 64, 96],     # Stage 2
    [64, 96, 128, 192],   # Stage 3（深层，通道多）
]


# ── 数据类 ─────────────────────────────────────────────────────────────────────
@dataclass
class StageConfig:
    n_blocks:     int
    out_channels: int
    kernel_size:  int
    expand_ratio: int
    use_se:       bool


@dataclass
class ArchConfig:
    n_stages: int
    stages:   list[StageConfig]

    def summary(self) -> str:
        lines = [f"n_stages={self.n_stages}"]
        for i, s in enumerate(self.stages):
            lines.append(
                f"  Stage{i+1}: blocks={s.n_blocks} ch={s.out_channels} "
                f"k={s.kernel_size} expand={s.expand_ratio} se={int(s.use_se)}"
            )
        return "\n".join(lines)


# ── 核心函数 ───────────────────────────────────────────────────────────────────
def decode(genome: list[int] | np.ndarray) -> ArchConfig:
    """16 维整数向量 → ArchConfig（仅包含实际使用的 stages）。"""
    genome = list(genome)
    assert len(genome) == NDIM, f"genome length must be {NDIM}, got {len(genome)}"
    for i, (lo, hi) in enumerate(DIM_RANGES):
        assert lo <= genome[i] <= hi, f"d{i}={genome[i]} out of [{lo},{hi}]"

    n_stages = _N_STAGES[genome[0]]
    stages = []
    for i in range(3):
        b = 1 + i * 5
        stages.append(StageConfig(
            n_blocks     = _N_BLOCKS[genome[b]],
            out_channels = _CHANNELS[i][genome[b + 1]],
            kernel_size  = _KERNELS[genome[b + 2]],
            expand_ratio = _EXPANDS[genome[b + 3]],
            use_se       = bool(genome[b + 4]),
        ))
    return ArchConfig(n_stages=n_stages, stages=stages[:n_stages])


def random_genome(rng: random.Random | None = None) -> list[int]:
    """均匀随机采样一个合法基因组。"""
    if rng is None:
        rng = random.Random()
    return [rng.randint(lo, hi) for lo, hi in DIM_RANGES]


def genome_to_array(genome: list[int]) -> np.ndarray:
    return np.array(genome, dtype=np.int32)


def search_space_size() -> int:
    size = 1
    for lo, hi in DIM_RANGES:
        size *= (hi - lo + 1)
    return size


# ── 快速验证 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"搜索空间维度 : {NDIM}")
    print(f"搜索空间大小 : {search_space_size():,}")
    print()

    rng = random.Random(42)
    for i in range(3):
        g = random_genome(rng)
        cfg = decode(g)
        print(f"[示例 {i+1}] genome = {g}")
        print(cfg.summary())
        print()
