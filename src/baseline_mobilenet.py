"""
baseline_mobilenet.py
---------------------
MobileNetV2 预训练微调，目标 Val Acc > 75%。
用法：
    python src/baseline_mobilenet.py
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from dataset import BirdDataset

# ── 超参 ──────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 20
LR_HEAD     = 1e-3   # 分类头学习率
LR_BACKBONE = 1e-4   # 骨干网络学习率（微调）
NUM_WORKERS = 8
CKPT_DIR    = Path("d:/NAS项目/checkpoints")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 40


# ── 模型 ──────────────────────────────────────────────────────────────────────
def build_model() -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


# ── 训练 / 验证 ───────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (out.argmax(1) == labels).sum().item()
        n          += len(labels)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * len(labels)
        correct    += (out.argmax(1) == labels).sum().item()
        n          += len(labels)
    return total_loss / n, correct / n


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}")

    train_ds = BirdDataset("train")
    val_ds   = BirdDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model().to(DEVICE)

    # 骨干和分类头用不同学习率
    optimizer = torch.optim.Adam([
        {"params": model.features.parameters(), "lr": LR_BACKBONE},
        {"params": model.classifier.parameters(), "lr": LR_HEAD},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc = evaluate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}  "
              f"{elapsed:.0f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            ckpt = CKPT_DIR / "baseline_mobilenet_best.pth"
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_acc": va_acc}, ckpt)
            print(f"  → 保存最优模型 val_acc={va_acc:.4f}")

    print(f"\n训练完成，最优 val_acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
