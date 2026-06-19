"""
baseline_random_search.py
-------------------------
搜索方法基线：随机搜索（Random Search）。

从同一搜索空间均匀随机采样 N_SAMPLES 个架构，每个用 proxy_eval 真实评估，
记录 (val_acc, params_M)，输出非支配 Pareto 前沿。

作用：验证「引导式进化（NSGA-II + EDA）是否真的优于纯随机采样」。
公平性：与 nsga2_eda.py 使用完全相同的 proxy_eval 评估方式和搜索空间，
唯一区别是采样策略（随机 vs 进化引导）。

评估顺序完整保留在 records 中，便于事后画 HV–评估预算曲线。

用法：
    python src/baseline_random_search.py          # 默认 N_SAMPLES=200
    python src/baseline_random_search.py 3        # 指定数量（冒烟测试用）
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from nsga2_eda import compute_cost, fast_non_dominated_sort
from search_space import decode, random_genome
from surrogate import proxy_eval

# ── 超参 ──────────────────────────────────────────────────────────────────────
N_SAMPLES   = 200       # 随机采样架构数（研究方案：×200）
SEED        = 42
RESULT_DIR  = Path("d:/NAS项目/results")
PROGRESS_EVERY = 10     # 每评估多少个就落盘一次（防止长时间运行中途丢失）


def _top5_stats(records: list[dict]) -> dict:
    """最优 5 个架构的 val_acc 均值 ± std（研究方案 RQ1 要求）。"""
    accs = sorted((r["val_acc"] for r in records), reverse=True)[:5]
    arr = np.array(accs)
    return {"top5_acc": accs, "mean": float(arr.mean()), "std": float(arr.std())}


def _pareto_front(records: list[dict]) -> list[dict]:
    """从已评估记录中取非支配前沿（acc 越大、params 越小越好）。"""
    fitness = np.array([[r["val_acc"], -r["params_M"]] for r in records])
    fronts = fast_non_dominated_sort(fitness)
    pareto = [records[i] for i in fronts[0]]
    pareto.sort(key=lambda r: -r["val_acc"])
    return pareto


def _save(records: list[dict], final: bool = False) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pareto = _pareto_front(records)
    out = {
        "method": "random_search",
        "n_evaluated": len(records),
        "n_target": N_SAMPLES,
        "seed": SEED,
        "pareto_size": len(pareto),
        "pareto_front": pareto,
        "top5_stats": _top5_stats(records),
        "records": records,   # 含评估顺序，可事后画 HV 曲线
    }
    name = "random_search_result.json" if final else "random_search_progress.json"
    (RESULT_DIR / name).write_text(json.dumps(out, indent=2))


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else N_SAMPLES
    rng = random.Random(SEED)

    print("=" * 60)
    print(f"Random Search  N={n_samples}  seed={SEED}")
    print("=" * 60 + "\n")

    records: list[dict] = []
    seen: dict[str, dict] = {}   # 去重：同一 genome 不重复评估
    t_start = time.time()

    i = 0
    while len(records) < n_samples:
        g = random_genome(rng)
        key = str(g)
        if key in seen:
            continue   # 撞重，换一个（1.77M 空间里极少发生）

        acc, el = proxy_eval(g)
        cost = compute_cost(g)
        rec = {
            "genome": g,
            "val_acc": round(acc, 6),
            "params_M": round(cost, 4),
            "n_stages": decode(g).n_stages,
            "eval_seconds": round(el, 1),
        }
        records.append(rec)
        seen[key] = rec
        i += 1

        best_so_far = max(r["val_acc"] for r in records)
        print(f"[{i:>3d}/{n_samples}] acc={acc:.4f} params={cost:.3f}M "
              f"{el:.0f}s  (best={best_so_far:.4f})")

        if len(records) % PROGRESS_EVERY == 0:
            _save(records, final=False)

    _save(records, final=True)

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    pareto = _pareto_front(records)
    stats = _top5_stats(records)
    print(f"\n{'='*60}")
    print(f"完成，共评估 {len(records)} 个架构，总耗时 {(time.time()-t_start)/60:.1f} min")
    print(f"最优 5 个 val_acc 均值±std = {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"\nPareto 前沿（{len(pareto)} 个）：")
    for r in pareto:
        print(f"  val_acc={r['val_acc']:.4f}  params={r['params_M']:.3f}M  "
              f"genome={r['genome']}")
    print(f"\n结果已保存至 {RESULT_DIR}/random_search_result.json")


if __name__ == "__main__":
    main()
