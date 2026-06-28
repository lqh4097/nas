"""
resolution_sweep.py
-------------------
分辨率 vs 精度 vs 计算量 扫描（用搜索空间的骨干 MobileNetV2 0.5×）。

动机：当前管线把梅尔谱一律拉到 224×224 喂网络，但 224 是从 ImageNet 抄来的惯例，
对梅尔谱很可能过大——而输入分辨率是边缘延迟最大的杠杆（MACs ∝ H×W）。
本实验在多个分辨率下**各自从零重训** MobileNetV2 0.5×（与基线同口径：SpecAugment、
30ep、Adam+Cosine），看精度在哪掉、计算量省多少，给"边缘该用小分辨率"提供实证。

- 224：复用已训好的基线 checkpoint（只评估，不重训），节省最慢一档。
- 128/96/64：各自重训。

⚠️ 诚实限制：只有 224 的 PNG、无原始音频，故小分辨率是把 224 图**降采样**得到，
   验证的是"小输入够不够识别"；最终落地的单通道原生小梅尔谱须等音频/上板那批。

用法：
    python src/resolution_sweep.py                       # 224(复用)+128+96+64
    python src/resolution_sweep.py --resolutions 128,96  # 只训指定档
    python src/resolution_sweep.py --epochs 20
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BirdDataset
from baseline_mobilenet import build_model, train_one_epoch, evaluate, WIDTH_MULT

BATCH_SIZE  = 64
NUM_WORKERS = 8
LR          = 1e-3
CKPT_DIR    = Path("d:/NAS项目/checkpoints")
BASE_CKPT   = CKPT_DIR / "baseline_mobilenet_best.pth"   # 已训好的 224 基线
OUT_JSON    = Path("d:/NAS项目/results/resolution_sweep.json")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_macs(model: nn.Module, res: int) -> float:
    """forward hook 统计 Conv2d/Linear MACs（MMACs）。"""
    macs = 0
    handles = []

    def conv_hook(m, inp, out):
        nonlocal macs
        oc, oh, ow = out.shape[1], out.shape[2], out.shape[3]
        kh, kw = m.kernel_size
        macs += oc * oh * ow * (m.in_channels // m.groups) * kh * kw

    def lin_hook(m, inp, out):
        nonlocal macs
        macs += m.in_features * m.out_features

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(lin_hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, res, res).to(DEVICE))
    for h in handles:
        h.remove()
    return macs / 1e6


def loaders_at(res: int):
    tr = DataLoader(BirdDataset("train", resolution=res), batch_size=BATCH_SIZE,
                    shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    va = DataLoader(BirdDataset("val", resolution=res), batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    te = DataLoader(BirdDataset("test", resolution=res), batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return tr, va, te


def train_at(res: int, epochs: int) -> dict:
    print(f"\n{'='*60}\n[res={res}] 从零训练 MobileNetV2 {WIDTH_MULT}×  {epochs} epoch\n{'='*60}")
    tr, va, te = loaders_at(res)
    torch.manual_seed(42)
    model = build_model().to(DEVICE)
    params_M = sum(p.numel() for p in model.parameters()) / 1e6
    macs = count_macs(model, res)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state = 0.0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        _, tr_acc = train_one_epoch(model, tr, optimizer, criterion)
        _, va_acc = evaluate(model, va, criterion)
        scheduler.step()
        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  ep {ep:02d}/{epochs}  train={tr_acc:.4f}  val={va_acc:.4f}  (best={best_val:.4f})")

    model.load_state_dict(best_state)
    _, test_acc = evaluate(model, te, criterion)
    mins = (time.time() - t0) / 60
    print(f"  → res={res}  val={best_val:.4f}  test={test_acc:.4f}  "
          f"MACs={macs:.1f}M  {mins:.1f}min")
    return {"resolution": res, "params_M": round(params_M, 4), "macs_M": round(macs, 1),
            "val_acc": round(best_val, 6), "test_acc": round(test_acc, 6),
            "train_min": round(mins, 1), "source": "trained"}


def eval_existing_224() -> dict | None:
    """复用已训好的 224 基线 checkpoint：只评估 val/test，不重训。"""
    if not BASE_CKPT.exists():
        return None
    print(f"\n[res=224] 复用已有基线 {BASE_CKPT.name}（仅评估）")
    blob = torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False)
    model = build_model().to(DEVICE)
    model.load_state_dict(blob["state_dict"])
    params_M = sum(p.numel() for p in model.parameters()) / 1e6
    macs = count_macs(model, 224)
    _, va, te = loaders_at(224)
    crit = nn.CrossEntropyLoss()
    _, val_acc = evaluate(model, va, crit)
    _, test_acc = evaluate(model, te, crit)
    print(f"  → res=224  val={val_acc:.4f}  test={test_acc:.4f}  MACs={macs:.1f}M  (复用)")
    return {"resolution": 224, "params_M": round(params_M, 4), "macs_M": round(macs, 1),
            "val_acc": round(val_acc, 6), "test_acc": round(test_acc, 6),
            "train_min": 0.0, "source": "reused"}


def main():
    ap = argparse.ArgumentParser(description="分辨率 vs 精度 vs 计算量 扫描（MobileNetV2 0.5×）")
    ap.add_argument("--resolutions", default="128,96,64", help="要重训的分辨率（逗号分隔）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--no-224", action="store_true", help="不复用 224 基线")
    args = ap.parse_args()
    print(f"device: {DEVICE}")

    rows = []
    if not args.no_224:
        r = eval_existing_224()
        if r:
            rows.append(r)
    for res in [int(x) for x in args.resolutions.split(",") if x.strip()]:
        rows.append(train_at(res, args.epochs))

    rows.sort(key=lambda r: -r["resolution"])
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # 汇总表 + 相对 224 的计算量/精度变化
    base = next((r for r in rows if r["resolution"] == 224), rows[0])
    print(f"\n{'='*72}\n分辨率扫描结果（MobileNetV2 {WIDTH_MULT}×）\n{'='*72}")
    print(f"{'分辨率':>8}{'MACs(M)':>10}{'相对MACs':>10}{'Val Acc':>10}{'Test Acc':>10}{'Δacc':>9}")
    for r in rows:
        rel = r["macs_M"] / base["macs_M"]
        dacc = (r["test_acc"] - base["test_acc"]) * 100
        print(f"{r['resolution']:>8}{r['macs_M']:>10.1f}{rel:>9.2f}×"
              f"{r['val_acc']:>10.4f}{r['test_acc']:>10.4f}{dacc:>+8.2f}%")
    print(f"\n结果已保存 → {OUT_JSON}")


if __name__ == "__main__":
    main()
