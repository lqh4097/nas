"""
surrogate.py
------------
代理模型：替代完整训练，快速估算架构准确率。

两阶段工作流：
  1. proxy_eval(genome)   — 在 10% 训练子集上跑 3 轮（约 30-60s/架构）
  2. SurrogateModel       — 积累 ≥ MIN_SAMPLES 条记录后，用随机森林预测（<1s/架构）

NAS 主循环调用 smart_eval()，自动切换两种模式。
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader, Subset

from dataset import BirdDataset
from net_builder import build_net
from search_space import NDIM, decode, genome_to_array

# ── 超参 ──────────────────────────────────────────────────────────────────────
PROXY_EPOCHS   = 3        # proxy 训练轮数
PROXY_SUBSET   = 0.10     # 使用 10% 训练数据
PROXY_BATCH    = 256
PROXY_LR       = 1e-3
PROXY_WORKERS  = 8
MIN_SAMPLES    = 30       # 启用随机森林的最低样本数
CACHE_PATH     = Path("d:/NAS项目/data/surrogate_cache.json")
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 数据加载（懒加载，只初始化一次）──────────────────────────────────────────────
_proxy_loader: DataLoader | None = None
_val_loader:   DataLoader | None = None


def _get_loaders() -> tuple[DataLoader, DataLoader]:
    global _proxy_loader, _val_loader
    if _proxy_loader is None:
        train_ds = BirdDataset("train")
        n = len(train_ds)
        idx = np.random.default_rng(42).choice(n, int(n * PROXY_SUBSET), replace=False)
        _proxy_loader = DataLoader(
            Subset(train_ds, idx.tolist()),
            batch_size=PROXY_BATCH, shuffle=True,
            num_workers=PROXY_WORKERS, pin_memory=True,
        )
        _val_loader = DataLoader(
            BirdDataset("val"),
            batch_size=PROXY_BATCH, shuffle=False,
            num_workers=PROXY_WORKERS, pin_memory=True,
        )
    return _proxy_loader, _val_loader


# ── proxy 评估 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def _val_acc(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, n = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        n += len(labels)
    return correct / n


def proxy_eval(genome: list[int]) -> tuple[float, float]:
    """
    在 10% 训练子集上训练 PROXY_EPOCHS 轮，返回 (val_acc, elapsed_seconds)。
    结果自动写入缓存。
    """
    t0 = time.time()
    proxy_loader, val_loader = _get_loaders()
    model = build_net(genome).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=PROXY_LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(PROXY_EPOCHS):
        for imgs, labels in proxy_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

    acc = _val_acc(model, val_loader)
    elapsed = time.time() - t0

    _cache_append(genome, acc)
    return acc, elapsed


# ── 缓存 I/O ───────────────────────────────────────────────────────────────────
def _load_cache() -> tuple[list[list[int]], list[float]]:
    if not CACHE_PATH.exists():
        return [], []
    data = json.loads(CACHE_PATH.read_text())
    return data["genomes"], data["accs"]


def _cache_append(genome: list[int], acc: float) -> None:
    genomes, accs = _load_cache()
    key = str(genome)
    keys = [str(g) for g in genomes]
    if key in keys:
        return  # 已存在，跳过
    genomes.append(genome)
    accs.append(acc)
    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps({"genomes": genomes, "accs": accs}, indent=2))


# ── 随机森林代理模型 ────────────────────────────────────────────────────────────
class SurrogateModel:
    """
    输入：16 维整数 genome（可加衍生特征）
    输出：预测 val_acc（回归）
    """

    def __init__(self, n_estimators: int = 100):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self._fitted = False

    @staticmethod
    def _features(genomes: list[list[int]]) -> np.ndarray:
        rows = []
        for g in genomes:
            cfg = decode(g)
            extra = [
                cfg.n_stages,
                sum(s.n_blocks for s in cfg.stages),
                sum(s.out_channels for s in cfg.stages),
            ]
            rows.append(list(g) + extra)
        return np.array(rows, dtype=np.float32)

    def fit(self, genomes: list[list[int]], accs: list[float]) -> None:
        X = self._features(genomes)
        y = np.array(accs, dtype=np.float32)
        self.rf.fit(X, y)
        self._fitted = True

    def predict(self, genomes: list[list[int]]) -> np.ndarray:
        assert self._fitted, "先调用 fit()"
        return self.rf.predict(self._features(genomes))

    def fit_from_cache(self) -> int:
        """从缓存加载数据并拟合，返回样本数。"""
        genomes, accs = _load_cache()
        if len(genomes) >= MIN_SAMPLES:
            self.fit(genomes, accs)
        return len(genomes)


# ── 智能调度 ───────────────────────────────────────────────────────────────────
_surrogate = SurrogateModel()


def smart_eval(genome: list[int]) -> tuple[float, str]:
    """
    自动选择评估方式：
      - 缓存 < MIN_SAMPLES → proxy_eval（真实训练）
      - 缓存 ≥ MIN_SAMPLES → 随机森林预测

    返回 (acc, mode)，mode 为 "proxy" 或 "surrogate"。
    """
    genomes, accs = _load_cache()

    # 命中缓存直接返回
    key = str(genome)
    for g, a in zip(genomes, accs):
        if str(g) == key:
            return a, "cache"

    if len(genomes) >= MIN_SAMPLES:
        _surrogate.fit_from_cache()
        acc = float(_surrogate.predict([genome])[0])
        return acc, "surrogate"
    else:
        acc, _ = proxy_eval(genome)
        return acc, "proxy"


# ── 快速验证 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    from search_space import random_genome

    print(f"device: {DEVICE}")
    print(f"缓存路径: {CACHE_PATH}")

    genomes_cached, accs_cached = _load_cache()
    print(f"已有缓存: {len(genomes_cached)} 条\n")

    rng = random.Random(0)
    g = random_genome(rng)
    print(f"测试 genome: {g}")
    acc, elapsed = proxy_eval(g)
    print(f"proxy_eval → val_acc={acc:.4f}  耗时={elapsed:.1f}s")
