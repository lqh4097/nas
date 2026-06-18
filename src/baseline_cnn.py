"""
baseline_cnn.py
---------------
4层卷积 + 2层全连接，目标 Val Acc > 70%。
用法：
    python src/baseline_cnn.py
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BirdDataset

# ── 超参 ──────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 1e-3
NUM_WORKERS = 8
CKPT_DIR    = Path("d:/NAS项目/checkpoints")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 40


# ── 模型 ──────────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # block1: 3 → 32
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                        # 112×112
            # block2: 32 → 64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                        # 56×56
            # block3: 64 → 128
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                        # 28×28
            # block4: 128 → 256
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),                                # 4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


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

    model     = SimpleCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
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
              f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            ckpt = CKPT_DIR / "baseline_cnn_best.pth"
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_acc": va_acc}, ckpt)
            print(f"  → 保存最优模型 val_acc={va_acc:.4f}")

    print(f"\n训练完成，最优 val_acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
