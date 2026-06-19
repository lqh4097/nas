"""
baseline_classic.py
--------------------
经典轻量网络基线（torchvision 直接调用），**从零训练**，统一口径。

支持：
    shufflenet_v2_x0_5   异范式（channel shuffle）核心基线
    mobilenet_v3_small   精度上限参照核心基线

与 Manual-CNN / MobileNetV2 0.5× 同口径：无 ImageNet 预训练、SpecAugment、
Adam+Cosine、30 epoch，保证与 NAS 搜出的架构可比。

用法：
    python src/baseline_classic.py --model shufflenet_v2_x0_5
    python src/baseline_classic.py --model mobilenet_v3_small
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from dataset import BirdDataset

# ── 超参 ──────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 1e-3
NUM_WORKERS = 8
CKPT_DIR    = Path("d:/NAS项目/checkpoints")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 40

MODELS = ("shufflenet_v2_x0_5", "mobilenet_v3_small")


# ── 模型 ──────────────────────────────────────────────────────────────────────
def build_model(name: str) -> nn.Module:
    """从零初始化指定网络，并把分类头替换为 NUM_CLASSES。"""
    if name == "shufflenet_v2_x0_5":
        model = models.shufflenet_v2_x0_5(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, NUM_CLASSES)
    else:
        raise ValueError(f"未知模型: {name}（可选 {MODELS}）")
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
    parser = argparse.ArgumentParser(description="经典轻量网络基线（从零训练）")
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}")

    train_ds = BirdDataset("train")
    val_ds   = BirdDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model(args.model).to(DEVICE)
    params_M = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{args.model}  参数量 = {params_M:.3f}M  (从零训练)")

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
            ckpt = CKPT_DIR / f"baseline_{args.model}_best.pth"
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_acc": va_acc, "params_M": params_M,
                        "model": args.model}, ckpt)
            print(f"  → 保存最优模型 val_acc={va_acc:.4f}")

    print(f"\n训练完成，{args.model}  参数量 {params_M:.3f}M  "
          f"最优 val_acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
