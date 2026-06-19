"""
nsga2_eda.py
------------
NSGA-II（多目标进化）+ EDA（从精英学习采样分布）+ 随机森林代理 的 NAS 主循环。

两个优化目标（均取最大化）：
  f1 = val_acc              ← proxy_eval 真实评估 / 代理预测
  f2 = -params_M            ← 参数量（越小越好 → 取负）；后续替换为 -latency_ms

消融实验（RQ3，从 Full 留一法）—— 三个可独立关闭的组件：
  Init      智能初始化：分层（按 n_stages）+ 成本引导（覆盖参数量范围）；关 → 纯随机
  Surrogate 代理：代理预测 + 每代回填校准；                 关 → 每个架构都真实 proxy 评估
  EDA       生成：从精英学边缘分布概率采样；                关 → 交叉 + 变异（标准 GA）

  --ablation 选择实验组：
    full          三者全开（完整方案）
    no-init       关 Init
    no-surrogate  关 Surrogate
    no-eda        关 EDA
    baseline      三者全关（标准 NSGA-II）

代理策略（surrogate-in-the-loop，持续回填）：
  gen 0   : 整个种群用 proxy_eval 真实评估 → 代理的初始训练集
  gen ≥1  : 每代用「全部已积累的 proxy 样本」重新拟合代理；子代先由代理预测，
            挑预测最优的 N_INFILL 个真实评估并回填，其余用预测值 → 代理越来越准。

可复现：不复用任何历史磁盘缓存，每次运行从干净状态开始；结果按
  results/{ablation}/seed{seed}/ 分目录保存（5 组 × 5 seed 互不覆盖）。

注意：搜索期 val_acc 是 proxy/代理估计，仅用于排序；
最终 Pareto 架构须用 retrain_pareto.py 完整训练后再上报真实指标。

用法：
    python src/nsga2_eda.py                              # full, seed42, gen15
    python src/nsga2_eda.py --ablation no-eda --seed 1
    python src/nsga2_eda.py --ablation baseline --gen 15 --pop 20
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from net_builder import build_net
from search_space import DIM_RANGES, decode, random_genome
from surrogate import SurrogateModel, proxy_eval

# ── 超参（部分可由命令行覆盖）─────────────────────────────────────────────────
ELITE_RATIO = 0.5      # 精英比例（用于 EDA 学习 / GA 交配池）
EDA_SMOOTH  = 1.0      # Laplace 平滑系数（防止概率为 0）
MUTATE_PROB = 0.1      # 每维随机突变概率（保持多样性）
N_INFILL    = 3        # 每代用 proxy_eval 真实评估的子代数量（仅 Surrogate 开时生效）
RESULT_ROOT = Path("d:/NAS项目/results")

# 五个消融组 → 三个组件开关
ABLATIONS = {
    "full":         dict(use_init=True,  use_surrogate=True,  use_eda=True),
    "no-init":      dict(use_init=False, use_surrogate=True,  use_eda=True),
    "no-surrogate": dict(use_init=True,  use_surrogate=False, use_eda=True),
    "no-eda":       dict(use_init=True,  use_surrogate=True,  use_eda=False),
    "baseline":     dict(use_init=False, use_surrogate=False, use_eda=False),
}


# ── 代价估算（参数量，后续替换为 RKNN 延迟）─────────────────────────────────────
def compute_cost(genome: list[int]) -> float:
    """返回参数量（单位 M），用作硬件代价的代理指标。"""
    model = build_net(genome)
    return sum(p.numel() for p in model.parameters()) / 1e6


# ── NSGA-II ────────────────────────────────────────────────────────────────────
def dominates(fa: np.ndarray, fb: np.ndarray) -> bool:
    """fa 支配 fb：所有目标 fa>=fb 且至少一个严格大于。"""
    return bool(np.all(fa >= fb) and np.any(fa > fb))


def fast_non_dominated_sort(fitness: np.ndarray) -> list[list[int]]:
    """
    fitness: [N, 2]，两列均为"越大越好"。
    返回按支配关系分层的前沿列表，front[0] 是 Pareto 前沿。
    """
    n = len(fitness)
    dom_count = np.zeros(n, dtype=int)   # 支配 i 的解的数量
    dom_set   = [[] for _ in range(n)]   # i 支配的解的集合
    fronts    = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(fitness[i], fitness[j]):
                dom_set[i].append(j)
            elif dominates(fitness[j], fitness[i]):
                dom_count[i] += 1
        if dom_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dom_set[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def crowding_distance(front: list[int], fitness: np.ndarray) -> dict[int, float]:
    """计算 front 中每个解的拥挤距离（越大越好）。"""
    dist = {i: 0.0 for i in front}
    n_obj = fitness.shape[1]

    for m in range(n_obj):
        sorted_idx = sorted(front, key=lambda i: fitness[i, m])
        f_min = fitness[sorted_idx[0],  m]
        f_max = fitness[sorted_idx[-1], m]
        dist[sorted_idx[0]]  = float("inf")
        dist[sorted_idx[-1]] = float("inf")
        span = f_max - f_min if f_max != f_min else 1e-9
        for k in range(1, len(sorted_idx) - 1):
            dist[sorted_idx[k]] += (
                fitness[sorted_idx[k + 1], m] - fitness[sorted_idx[k - 1], m]
            ) / span

    return dist


def select_next_population(
    population: list[list[int]],
    fitness: np.ndarray,
    n: int,
) -> tuple[list[list[int]], np.ndarray]:
    """从 population 中按 NSGA-II 选出 n 个解。"""
    fronts = fast_non_dominated_sort(fitness)
    selected_idx: list[int] = []

    for front in fronts:
        if len(selected_idx) + len(front) <= n:
            selected_idx.extend(front)
        else:
            remaining = n - len(selected_idx)
            cd = crowding_distance(front, fitness)
            sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
            selected_idx.extend(sorted_front[:remaining])
            break

    sel_pop = [population[i] for i in selected_idx]
    sel_fit = fitness[selected_idx]
    return sel_pop, sel_fit


# ── 初始化：智能（分层+成本引导） vs 随机 ───────────────────────────────────────
def _cost_spread_select(pool: list[list[int]], k: int) -> list[list[int]]:
    """按参数量排序后等距取 k 个 → 覆盖从小到大的成本范围（成本引导）。"""
    if k <= 0:
        return []
    if len(pool) <= k:
        return list(pool)
    pool_sorted = sorted(pool, key=compute_cost)
    if k == 1:
        return [pool_sorted[len(pool_sorted) // 2]]
    idxs = [round(i * (len(pool_sorted) - 1) / (k - 1)) for i in range(k)]
    return [pool_sorted[i] for i in idxs]


def smart_init(pop_size: int, rng: random.Random) -> list[list[int]]:
    """
    分层 + 成本引导初始化：
      分层    —— 按 n_stages（2/3）各占一半，保证深浅结构都覆盖
      成本引导 —— 每层在过采样池里按参数量等距挑选，覆盖大小两端
    """
    pool = [random_genome(rng) for _ in range(pop_size * 6)]
    s2 = [g for g in pool if g[0] == 0]   # n_stages=2
    s3 = [g for g in pool if g[0] == 1]   # n_stages=3
    half = pop_size // 2
    chosen = _cost_spread_select(s2, half) + _cost_spread_select(s3, pop_size - half)
    while len(chosen) < pop_size:         # 兜底补齐
        chosen.append(random_genome(rng))
    rng.shuffle(chosen)
    return chosen[:pop_size]


def random_init(pop_size: int, rng: random.Random) -> list[list[int]]:
    return [random_genome(rng) for _ in range(pop_size)]


# ── 生成算子：EDA 概率采样 vs 交叉+变异 ────────────────────────────────────────
def eda_sample(elite: list[list[int]], n_samples: int, rng: random.Random) -> list[list[int]]:
    """从精英学每维边缘分布（UMDA），采样 n_samples 个新基因组；Laplace 平滑。"""
    offspring = []
    for _ in range(n_samples):
        genome = []
        for d, (lo, hi) in enumerate(DIM_RANGES):
            n_vals = hi - lo + 1
            counts = np.zeros(n_vals)
            for g in elite:
                counts[g[d] - lo] += 1
            probs = (counts + EDA_SMOOTH) / (len(elite) + EDA_SMOOTH * n_vals)
            val = lo + rng.choices(range(n_vals), weights=probs.tolist())[0]
            genome.append(val)
        offspring.append(genome)
    return offspring


def crossover(p1: list[int], p2: list[int], rng: random.Random) -> list[int]:
    """均匀交叉：每维以 50% 概率取自 p1 或 p2。"""
    return [p1[d] if rng.random() < 0.5 else p2[d] for d in range(len(p1))]


def mutate(genome: list[int], rng: random.Random, prob: float = MUTATE_PROB) -> list[int]:
    """对每个维度以概率 prob 随机重置，保持多样性。"""
    g = genome.copy()
    for d, (lo, hi) in enumerate(DIM_RANGES):
        if rng.random() < prob:
            g[d] = rng.randint(lo, hi)
    return g


def make_offspring(elite: list[list[int]], n: int, rng: random.Random,
                   use_eda: bool) -> list[list[int]]:
    """根据是否启用 EDA 选择生成方式。"""
    if use_eda:
        return [mutate(g, rng) for g in eda_sample(elite, n, rng)]
    # 标准 GA：交叉 + 变异
    offspring = []
    for _ in range(n):
        if len(elite) >= 2:
            p1, p2 = rng.sample(elite, 2)
        else:
            p1 = p2 = elite[0]
        offspring.append(mutate(crossover(p1, p2, rng), rng))
    return offspring


# ── 评估缓存管理 ───────────────────────────────────────────────────────────────
def ensure_cost(genome: list[int], cost_cache: dict) -> float:
    key = str(genome)
    if key not in cost_cache:
        cost_cache[key] = compute_cost(genome)
    return cost_cache[key]


def build_fitness(population, acc_cache, cost_cache) -> np.ndarray:
    """从缓存组装 fitness [N,2]：col0=val_acc，col1=-params_M。"""
    fitness = np.zeros((len(population), 2))
    for i, g in enumerate(population):
        acc, _ = acc_cache[str(g)]
        fitness[i] = [acc, -cost_cache[str(g)]]
    return fitness


def eval_offspring(offspring, surrogate, proxy_genomes, proxy_accs,
                   acc_cache, cost_cache, use_surrogate: bool) -> int:
    """
    评估子代，返回本代真实（proxy）评估的数量。
      use_surrogate=False : 全部子代真实评估（无代理预测）
      use_surrogate=True  : 先代理预测，挑预测最优的 N_INFILL 个真实回填，其余用预测值
    """
    for g in offspring:
        ensure_cost(g, cost_cache)

    unknown = [g for g in offspring if str(g) not in acc_cache]
    if not unknown:
        return 0

    if not use_surrogate:
        for g in unknown:
            acc, _ = proxy_eval(g)
            proxy_genomes.append(g)
            proxy_accs.append(acc)
            acc_cache[str(g)] = (acc, "proxy")
        return len(unknown)

    preds = (surrogate.predict(unknown) if surrogate.fitted
             else np.zeros(len(unknown)))
    ranked = sorted(zip(unknown, preds), key=lambda x: -x[1])

    n_real = 0
    for j, (g, p) in enumerate(ranked):
        if j < N_INFILL:
            acc, _ = proxy_eval(g)
            proxy_genomes.append(g)
            proxy_accs.append(acc)
            acc_cache[str(g)] = (acc, "infill")
            n_real += 1
        else:
            acc_cache[str(g)] = (float(p), "surrogate")
    return n_real


# ── 日志 / 结果保存 ────────────────────────────────────────────────────────────
def save_generation(gen, population, fitness, acc_cache, cost_cache, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    fronts = fast_non_dominated_sort(fitness)
    pareto = []
    for i in fronts[0]:
        g = population[i]
        acc, mode = acc_cache[str(g)]
        cfg = decode(g)
        pareto.append({
            "genome": g,
            "val_acc": round(acc, 6),
            "params_M": round(cost_cache[str(g)], 4),
            "n_stages": cfg.n_stages,
            "mode": mode,
        })
    pareto.sort(key=lambda x: -x["val_acc"])

    out = {
        "generation": gen,
        "pop_size": len(population),
        "pareto_size": len(pareto),
        "pareto_front": pareto,
    }
    (run_dir / f"gen_{gen:03d}.json").write_text(json.dumps(out, indent=2))
    return pareto


# ── 主循环 ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="NSGA-II + EDA + Surrogate NAS（含消融开关）")
    p.add_argument("--ablation", choices=list(ABLATIONS), default="full",
                   help="消融实验组（默认 full = 完整方案）")
    p.add_argument("--seed", type=int, default=42, help="随机种子（×5 重复时变化）")
    p.add_argument("--gen", type=int, default=15, help="进化代数（方案消融用 15）")
    p.add_argument("--pop", type=int, default=20, help="种群大小（方案用 20）")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ABLATIONS[args.ablation]
    pop_size, n_gen = args.pop, args.gen
    run_dir = RESULT_ROOT / args.ablation / f"seed{args.seed}"

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    proxy_genomes: list[list[int]] = []   # 真实(proxy)评估过的架构
    proxy_accs:    list[float]     = []   # 对应准确率 —— 代理训练集
    acc_cache:  dict = {}                 # str(genome) -> (acc, mode)
    cost_cache: dict = {}                 # str(genome) -> params_M
    surrogate = SurrogateModel()

    print("=" * 64)
    print(f"NAS 主循环  组={args.ablation}  seed={args.seed}  POP={pop_size}  GEN={n_gen}")
    print(f"  Init={'智能' if cfg['use_init'] else '随机'}  "
          f"Surrogate={'开' if cfg['use_surrogate'] else '关(全真实评估)'}  "
          f"生成={'EDA' if cfg['use_eda'] else '交叉+变异'}")
    print(f"  → {run_dir}")
    print("=" * 64 + "\n")

    # ── Gen 0: 初始化 + 整个种群真实评估 ───────────────────────────────────────
    population = smart_init(pop_size, rng) if cfg["use_init"] else random_init(pop_size, rng)
    print(f"[Gen 0] {'智能' if cfg['use_init'] else '随机'}初始化 {pop_size} 个，"
          f"全部 proxy_eval …")
    t0 = time.time()
    for i, g in enumerate(population):
        acc, el = proxy_eval(g)
        proxy_genomes.append(g)
        proxy_accs.append(acc)
        acc_cache[str(g)] = (acc, "proxy")
        ensure_cost(g, cost_cache)
        print(f"  [{i+1:>2d}/{pop_size}] acc={acc:.4f} "
              f"params={cost_cache[str(g)]:.3f}M  {el:.0f}s")
    print(f"  warmup 完成，耗时 {(time.time()-t0)/60:.1f} min\n")
    fitness = build_fitness(population, acc_cache, cost_cache)
    save_generation(0, population, fitness, acc_cache, cost_cache, run_dir)

    # ── 进化循环 ──────────────────────────────────────────────────────────────
    for gen in range(1, n_gen + 1):
        print(f"[Gen {gen}/{n_gen}]")
        t0 = time.time()

        # 1. 代理开启时，用全部已积累 proxy 样本重新拟合（每代一次）
        if cfg["use_surrogate"]:
            surrogate.fit(proxy_genomes, proxy_accs)

        # 2. 选精英（NSGA-II）
        n_elite = max(2, int(pop_size * ELITE_RATIO))
        elite_pop, _ = select_next_population(population, fitness, n_elite)

        # 3. 生成子代（EDA 或 交叉+变异）
        n_offspring = pop_size - n_elite
        offspring = make_offspring(elite_pop, n_offspring, rng, cfg["use_eda"])

        # 4. 评估子代
        n_real = eval_offspring(offspring, surrogate, proxy_genomes, proxy_accs,
                                acc_cache, cost_cache, cfg["use_surrogate"])
        offspring_fit = build_fitness(offspring, acc_cache, cost_cache)

        # 5. 合并父代+子代，NSGA-II 选回 pop_size
        population, fitness = select_next_population(
            population + offspring,
            np.vstack([fitness, offspring_fit]),
            pop_size,
        )

        # 6. 记录本代 Pareto 前沿
        pareto = save_generation(gen, population, fitness, acc_cache, cost_cache, run_dir)
        best_acc  = max(p["val_acc"]  for p in pareto)
        best_cost = min(p["params_M"] for p in pareto)
        print(f"  proxy样本={len(proxy_genomes)} 本代真实评估={n_real}  "
              f"Pareto={len(pareto)} best_acc={best_acc:.4f} "
              f"min_params={best_cost:.3f}M  {time.time()-t0:.0f}s\n")

    # ── 输出最终 Pareto 前沿 ──────────────────────────────────────────────────
    print("=" * 64)
    print("最终 Pareto 前沿（val_acc 为 proxy/代理估计，须 retrain_pareto.py 复训）：")
    fronts = fast_non_dominated_sort(fitness)
    final = []
    for i in fronts[0]:
        g = population[i]
        acc, mode = acc_cache[str(g)]
        final.append((acc, cost_cache[str(g)], g, mode))
    final.sort(key=lambda x: -x[0])
    for acc, cost, g, mode in final:
        print(f"  val_acc={acc:.4f}  params={cost:.3f}M  [{mode:9s}]  genome={g}")

    (run_dir / "final_pareto.json").write_text(json.dumps(
        [{"genome": g, "val_acc": round(a, 6), "params_M": round(c, 4), "mode": m}
         for a, c, g, m in final], indent=2))
    (run_dir / "proxy_samples.json").write_text(json.dumps(
        {"genomes": proxy_genomes, "accs": proxy_accs}, indent=2))
    (run_dir / "config.json").write_text(json.dumps(
        {"ablation": args.ablation, "seed": args.seed, "pop": pop_size,
         "gen": n_gen, **cfg, "n_proxy_evals": len(proxy_genomes)}, indent=2))

    print(f"\n结果已保存至 {run_dir}/")
    print(f"  - final_pareto.json   最终 Pareto 架构")
    print(f"  - proxy_samples.json  全部 {len(proxy_genomes)} 个真实样本")
    print(f"  - config.json         本次运行配置")


if __name__ == "__main__":
    main()
