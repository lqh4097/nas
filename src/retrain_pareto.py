"""
retrain_pareto.py
-----------------
对 NAS 搜出的 Pareto 前沿架构做「完整训练 + 封存 test 集评估」，
得到论文用的真实准确率。

搜索期间的 val_acc 来自 proxy（3 轮 / 10% 数据）或随机森林代理，仅用于排序，
不能直接上报。本脚本对 results/final_pareto.json 中每个架构：
  1. 在完整 train 集上训练 EPOCHS 轮
  2. 取 val_acc 最优的权重
  3. 在封存的 test 集上评估，记录真实 test_acc
结果写入 results/retrain_result.json，权重存到 checkpoints/pareto/。

用法：
    python src/retrain_pareto.py                                   # 默认 full/seed42
    python src/retrain_pareto.py --run-dir results/no-eda/seed1    # 指定某消融组
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BirdDataset
from net_builder import build_net
from search_space import decode

# ── 超参 ──────────────────────────────────────────────────────────────────────
EPOCHS      = 30
BATCH_SIZE  = 128
LR          = 1e-3
NUM_WORKERS = 8
DEFAULT_RUN_DIR = Path("d:/NAS项目/results/full/seed42")
CKPT_DIR    = Path("d:/NAS项目/checkpoints/pareto")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    correct, n = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        correct += (out.argmax(1) == labels).sum().item()
        n += len(labels)
    return correct / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, n = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        n += len(labels)
    return correct / n


# 数据加载器按分辨率缓存（协同搜索：每个架构在自己的分辨率上训练）
_LOADERS: dict[int, tuple] = {}


def get_loaders(res: int) -> tuple:
    if res not in _LOADERS:
        _LOADERS[res] = (
            DataLoader(BirdDataset("train", resolution=res), batch_size=BATCH_SIZE,
                       shuffle=True, num_workers=NUM_WORKERS, pin_memory=True),
            DataLoader(BirdDataset("val", resolution=res), batch_size=BATCH_SIZE,
                       shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
            DataLoader(BirdDataset("test", resolution=res), batch_size=BATCH_SIZE,
                       shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
        )
    return _LOADERS[res]


def retrain_one(genome: list[int], idx: int, ckpt_dir: Path) -> dict:
    res = decode(genome).resolution
    train_loader, val_loader, test_loader = get_loaders(res)
    torch.manual_seed(42)
    model = build_net(genome).to(DEVICE)
    params_M = sum(p.numel() for p in model.parameters()) / 1e6
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        va_acc = evaluate(model, val_loader)
        scheduler.step()
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"    epoch {epoch:02d}/{EPOCHS}  train_acc={tr_acc:.4f} "
              f"val_acc={va_acc:.4f}  (best={best_val_acc:.4f})")

    # 用最优权重评估 test
    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"arch_{idx:02d}.pth"
    torch.save({"genome": genome, "state_dict": best_state,
                "val_acc": best_val_acc, "test_acc": test_acc,
                "resolution": res}, ckpt)

    return {
        "idx": idx,
        "genome": genome,
        "params_M": round(params_M, 4),
        "resolution": res,
        "val_acc": round(best_val_acc, 6),
        "test_acc": round(test_acc, 6),
        "ckpt": str(ckpt),
    }


def parse_args():
    p = argparse.ArgumentParser(description="对 NAS Pareto 前沿架构完整训练 + test 评估")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                   help="NAS 结果目录（含 final_pareto.json），默认 results/full/seed42")
    p.add_argument("--indices", type=str, default=None,
                   help="只复训指定下标（逗号分隔，对应 final_pareto.json 顺序），默认全部")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    pareto_path = run_dir / "final_pareto.json"
    assert pareto_path.exists(), f"找不到 {pareto_path}，请先跑 nsga2_eda.py"
    pareto_all = json.loads(pareto_path.read_text())

    # 选择要复训的架构（保留原始下标，checkpoint 命名稳定）
    if args.indices:
        sel = [int(i) for i in args.indices.split(",")]
        items = [(i, pareto_all[i]) for i in sel]
    else:
        items = list(enumerate(pareto_all))

    # checkpoint 按组分目录，避免不同消融组互相覆盖
    run_name = f"{run_dir.parent.name}_{run_dir.name}"   # 如 full_seed42
    ckpt_dir = CKPT_DIR / run_name

    print(f"device: {DEVICE}")
    print(f"结果目录: {run_dir}")
    print(f"待复训架构数: {len(items)} / {len(pareto_all)}（前沿总数）\n")

    results = []
    for n, (idx, item) in enumerate(items):
        g = item["genome"]
        print(f"[{n+1}/{len(items)}] 前沿#{idx} genome={g} res={decode(g).resolution}  "
              f"(搜索期估计 val_acc={item.get('val_acc')})")
        t0 = time.time()
        res = retrain_one(g, idx, ckpt_dir)
        res["search_val_acc"] = item.get("val_acc")
        results.append(res)
        print(f"  → val_acc={res['val_acc']:.4f}  test_acc={res['test_acc']:.4f}  "
              f"params={res['params_M']:.3f}M  res={res['resolution']}  "
              f"耗时 {(time.time()-t0)/60:.1f} min\n")

    # 合并进已有结果（部分重训不覆盖其它架构）：按前沿下标 idx 更新
    out_path = run_dir / "retrain_result.json"
    by_idx = {}
    if out_path.exists():
        for r in json.loads(out_path.read_text()):
            by_idx[r["idx"]] = r
    for r in results:
        by_idx[r["idx"]] = r          # 本次重训的覆盖旧值
    merged = sorted(by_idx.values(), key=lambda r: -r["test_acc"])
    out_path.write_text(json.dumps(merged, indent=2))

    print("=" * 60)
    print("复训完成（真实指标，按 test_acc 排序）：")
    for r in results:
        print(f"  test_acc={r['test_acc']:.4f}  val_acc={r['val_acc']:.4f}  "
              f"params={r['params_M']:.3f}M  genome={r['genome']}")
    print(f"\n结果已保存至 {run_dir}/retrain_result.json")


if __name__ == "__main__":
    main()
