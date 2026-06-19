"""
surrogate.py
------------
供 NAS 主循环使用的两个工具：

  proxy_eval(genome)   — 在 10% 训练子集上训练几轮，返回真实（代理）准确率；
                         固定随机种子，同一架构两次评估结果一致（可复现）。
  SurrogateModel       — 随机森林，从已积累的 proxy 样本预测准确率（<1s/架构）。

本模块不持久化任何状态：样本的积累、缓存、代理拟合频率全部由调用方
（nsga2_eda.py）管理，保证每次实验从干净状态开始、可复现。
"""

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader, Subset

from dataset import BirdDataset
from net_builder import build_net
from search_space import decode

# ── 超参 ──────────────────────────────────────────────────────────────────────
PROXY_EPOCHS  = 3        # proxy 训练轮数
PROXY_SUBSET  = 0.10     # 使用 10% 训练数据
PROXY_BATCH   = 128      # 128 对搜索空间里最大的架构也安全（256 会 OOM）
PROXY_LR      = 1e-3
PROXY_WORKERS = 8
SEED          = 42
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 数据加载（懒加载，只初始化一次）──────────────────────────────────────────────
_proxy_loader: DataLoader | None = None
_val_loader:   DataLoader | None = None


def _get_loaders() -> tuple[DataLoader, DataLoader]:
    global _proxy_loader, _val_loader
    if _proxy_loader is None:
        train_ds = BirdDataset("train")
        n = len(train_ds)
        idx = np.random.default_rng(SEED).choice(n, int(n * PROXY_SUBSET), replace=False)
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


def proxy_eval(genome: list[int], seed: int = SEED) -> tuple[float, float]:
    """
    在 10% 训练子集上训练 PROXY_EPOCHS 轮，返回 (val_acc, elapsed_seconds)。
    固定随机种子 → 权重初始化与训练过程可复现。
    """
    torch.manual_seed(seed)
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

    # 释放显存：否则循环评估大量架构时会累积导致 OOM
    del model, optimizer
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return acc, elapsed


# ── 随机森林代理模型 ────────────────────────────────────────────────────────────
class SurrogateModel:
    """
    输入：16 维整数 genome（附加 3 个衍生特征）
    输出：预测 val_acc（回归）

    fit() 由调用方在每代开始时用全部已积累的 proxy 样本调用一次。
    """

    def __init__(self, n_estimators: int = 100):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
        )
        self.fitted = False

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
        self.rf.fit(self._features(genomes), np.array(accs, dtype=np.float32))
        self.fitted = True

    def predict(self, genomes: list[list[int]]) -> np.ndarray:
        assert self.fitted, "先调用 fit()"
        return self.rf.predict(self._features(genomes))


# ── 快速验证 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    from search_space import random_genome

    print(f"device: {DEVICE}")
    g = random_genome(random.Random(0))
    print(f"测试 genome: {g}")
    acc, elapsed = proxy_eval(g)
    print(f"proxy_eval → val_acc={acc:.4f}  耗时={elapsed:.1f}s")
