"""
baseline_crnn.py
----------------
选做基线：Manual-CRNN（3层 CNN + 1层 GRU + FC），**从零训练**。

作用：验证"纯 CNN 能否超越 CNN+RNN"。频谱图的水平轴是时间轴——CNN 先提特征，
沿频率轴(H)平均后把时间轴(W)当作序列喂给 GRU 做时序建模。

⚠️ 部署 caveat：GRU 在 RK3566 NPU 上常不被支持、回退 CPU，延迟可能很差——
这恰是支持"纯 CNN 搜索空间"的论据；论文中报告其 RK3566 延迟时须显式标注。

与其它基线同口径：SpecAugment、Adam+Cosine、30 epoch、无预训练。

用法：
    python src/baseline_crnn.py
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
GRU_HIDDEN  = 64
CKPT_DIR    = Path("d:/NAS项目/checkpoints")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 40


# ── 模型 ──────────────────────────────────────────────────────────────────────
class CRNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, gru_hidden: int = GRU_HIDDEN):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 112
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 56
            nn.Conv2d(64, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 28
        )
        self.gru = nn.GRU(input_size=128, hidden_size=gru_hidden, batch_first=True)
        self.fc  = nn.Linear(gru_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)            # [B, 128, 28, 28]
        x = x.mean(dim=2)          # 沿频率轴(H)平均 → [B, 128, 28]
        x = x.permute(0, 2, 1)     # [B, 28(时间), 128(特征)]
        _, h = self.gru(x)         # h: [1, B, gru_hidden]
        return self.fc(h[-1])      # [B, num_classes]


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

    train_loader = DataLoader(BirdDataset("train"), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(BirdDataset("val"),   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = CRNN().to(DEVICE)
    params_M = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Manual-CRNN  参数量 = {params_M:.3f}M  (从零训练)")

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
            ckpt = CKPT_DIR / "baseline_crnn_best.pth"
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_acc": va_acc, "params_M": params_M, "model": "crnn"}, ckpt)
            print(f"  → 保存最优模型 val_acc={va_acc:.4f}")

    print(f"\n训练完成，Manual-CRNN  参数量 {params_M:.3f}M  最优 val_acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
