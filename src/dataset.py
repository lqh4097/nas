import json
from pathlib import Path

import numpy as np
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


def _build_transform(split: str, mean: list, std: list) -> transforms.Compose:
    normalize = transforms.Normalize(mean=mean, std=std)
    if split == "train":
        return transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=8),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
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
