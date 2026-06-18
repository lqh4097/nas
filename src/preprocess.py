"""
preprocess.py
-------------
两件事：
  1. 扫描 train 集，估计 per-channel mean/std → data/stats.json
  2. 将 val 集按 seed=42 分层 50/50 切分为 val/test → data/split_index.json

不生成 .npy（132K 张 × 224×224×3 ≈ 80 GB，改为 DataLoader 实时读取 PNG）。
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

DATASET_ROOT = Path("d:/NAS项目/Mel_Augment_four")
DATA_DIR = Path("d:/NAS项目/data")
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
NUM_CLASSES = 40
SAMPLE_N = 5000   # 从 train 随机抽样估计 mean/std，够精确且快
SEED = 42


def compute_stats(sample_n: int = SAMPLE_N) -> dict:
    """随机抽 sample_n 张 train 图，估计 per-channel mean/std（RGB）。"""
    all_paths = list(TRAIN_DIR.rglob("*.png"))
    rng = random.Random(SEED)
    sampled = rng.sample(all_paths, min(sample_n, len(all_paths)))

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    count = 0

    for p in tqdm(sampled, desc="计算 mean/std"):
        img = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        channel_sum += img.mean(axis=(0, 1))
        channel_sq_sum += (img ** 2).mean(axis=(0, 1))
        count += 1

    mean = channel_sum / count
    std = np.sqrt(channel_sq_sum / count - mean ** 2)

    return {"mean": mean.tolist(), "std": std.tolist(), "n_samples": count}


def build_split_index() -> dict:
    """将 val 集按类别分层 50/50 切为 val/test，返回各自的文件路径列表。"""
    rng = random.Random(SEED)
    val_paths, test_paths = [], []

    for cls_id in range(NUM_CLASSES):
        cls_files = sorted((VAL_DIR / str(cls_id)).glob("*.png"))
        cls_files = [str(p) for p in cls_files]
        rng.shuffle(cls_files)
        mid = len(cls_files) // 2
        val_paths.extend(cls_files[:mid])
        test_paths.extend(cls_files[mid:])

    return {
        "val": val_paths,
        "test": test_paths,
        "n_val": len(val_paths),
        "n_test": len(test_paths),
        "seed": SEED,
    }


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("=== Step 1: 计算训练集统计量 ===")
    stats = compute_stats()
    stats_path = DATA_DIR / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"mean = {[f'{v:.4f}' for v in stats['mean']]}")
    print(f"std  = {[f'{v:.4f}' for v in stats['std']]}")
    print(f"已保存 → {stats_path}\n")

    print("=== Step 2: 生成 val/test 索引 ===")
    split = build_split_index()
    split_path = DATA_DIR / "split_index.json"
    split_path.write_text(json.dumps(split, indent=2))
    print(f"val : {split['n_val']} 张")
    print(f"test: {split['n_test']} 张")
    print(f"已保存 → {split_path}")


if __name__ == "__main__":
    main()
