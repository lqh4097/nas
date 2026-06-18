import json
import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

DATASET_ROOT = Path("d:/NAS项目/Mel_Augment_four")
DATA_DIR     = Path("d:/NAS项目/data")
TRAIN_DIR    = DATASET_ROOT / "train"
VAL_DIR      = DATASET_ROOT / "val"
NUM_CLASSES  = 40


def _load_stats() -> tuple[list, list]:
    stats = json.loads((DATA_DIR / "stats.json").read_text())
    return stats["mean"], stats["std"]


def _load_split() -> dict:
    return json.loads((DATA_DIR / "split_index.json").read_text())


class SpecAugment:
    """
    频谱图专用增强：随机遮挡若干频率行 / 时间列（SpecAugment 思想）。
    在归一化之后作用于张量 [C,H,W]，遮挡值设为 0（即归一化后的均值）。

    不对频谱图做水平翻转 —— 水平轴是时间轴，翻转等于把鸟鸣倒放，语义错误；
    也不做垂直翻转 —— 会打乱频率轴。
    """

    def __init__(self, freq_mask: int = 24, time_mask: int = 24,
                 n_freq: int = 2, n_time: int = 2, p: float = 0.5):
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq = n_freq
        self.n_time = n_time
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        _, h, w = tensor.shape
        for _ in range(self.n_freq):
            f = random.randint(0, self.freq_mask)
            if f > 0:
                f0 = random.randint(0, max(0, h - f))
                tensor[:, f0:f0 + f, :] = 0.0
        for _ in range(self.n_time):
            t = random.randint(0, self.time_mask)
            if t > 0:
                t0 = random.randint(0, max(0, w - t))
                tensor[:, :, t0:t0 + t] = 0.0
        return tensor


def _build_transform(split: str, mean: list, std: list) -> transforms.Compose:
    normalize = transforms.Normalize(mean=mean, std=std)
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
            SpecAugment(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


class BirdDataset(Dataset):
    """
    split: "train" | "val" | "test"
    返回 (image_tensor [3,224,224], label int)
    """

    def __init__(self, split: str = "train"):
        assert split in ("train", "val", "test"), f"unknown split: {split}"
        self.split = split
        mean, std = _load_stats()
        self.transform = _build_transform(split, mean, std)
        self.samples: list[tuple[Path, int]] = []

        if split == "train":
            for cls_id in range(NUM_CLASSES):
                cls_dir = TRAIN_DIR / str(cls_id)
                for p in sorted(cls_dir.glob("*.png")):
                    self.samples.append((p, cls_id))
        else:
            index = _load_split()
            for path_str in index[split]:
                p = Path(path_str)
                cls_id = int(p.parent.name)
                self.samples.append((p, cls_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        ds = BirdDataset(split)
        img, lbl = ds[0]
        print(f"{split:5s}: {len(ds):>7d} 张  shape={tuple(img.shape)}  label={lbl}")
