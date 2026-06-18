"""
nsga2_eda.py
------------
NSGA-II（多目标进化）+ EDA（从精英学习采样分布）+ 随机森林代理 的 NAS 主循环。

两个优化目标（均取最大化）：
  f1 = val_acc              ← proxy_eval 真实评估 / 代理预测
  f2 = -params_M            ← 参数量（越小越好 → 取负）；后续替换为 -latency_ms

代理策略（surrogate-in-the-loop，持续回填）：
  gen 0   : 整个种群用 proxy_eval 真实评估 → 代理的初始训练集
  gen ≥1  : 每代用「全部已积累的 proxy 样本」重新拟合代理一次；
            子代先由代理预测，挑预测最优的 N_INFILL 个用 proxy_eval 真实评估并回填，
            其余直接用代理预测值。
            → 代理训练集随代数增长，代理越来越准（而非冻结在初始样本上）。

可复现性：本模块不复用任何历史磁盘缓存，每次运行从干净状态开始；
proxy 真实样本随结果保存到 RESULT_DIR/proxy_samples.json 以备复盘。

注意：搜索期间的 val_acc 是 proxy/代理估计值，仅用于排序；
最终 Pareto 架构必须用 retrain_pareto.py 完整训练后再上报真实指标。
"""

import json
import random
import time
from pathlib import Path

import numpy as np

from net_builder import build_net
from search_space import DIM_RANGES, decode, random_genome
from surrogate import SurrogateModel, proxy_eval

# ── 超参 ──────────────────────────────────────────────────────────────────────
POP_SIZE    = 20       # 种群大小
N_GEN       = 30       # 进化代数
ELITE_RATIO = 0.5      # 精英比例（用于 EDA 学习）
EDA_SMOOTH  = 1.0      # Laplace 平滑系数（防止概率为 0）
MUTATE_PROB = 0.1      # 每维随机突变概率（保持多样性）
N_INFILL    = 3        # 每代用 proxy_eval 真实评估的子代数量（回填代理）
RESULT_DIR  = Path("d:/NAS项目/results")
SEED        = 42


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


# ── EDA ────────────────────────────────────────────────────────────────────────
def eda_sample(
    elite: list[list[int]],
    n_samples: int,
    rng: random.Random,
) -> list[list[int]]:
    """
    从精英种群中学习每维度的边缘分布（UMDA），采样 n_samples 个新基因组。
    使用 Laplace 平滑防止概率为零。
    """
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


def mutate(genome: list[int], rng: random.Random, prob: float = MUTATE_PROB) -> list[int]:
    """对每个维度以概率 prob 随机重置，保持种群多样性。"""
    g = genome.copy()
    for d, (lo, hi) in enumerate(DIM_RANGES):
        if rng.random() < prob:
            g[d] = rng.randint(lo, hi)
    return g


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
                   acc_cache, cost_cache) -> int:
    """
    评估子代：
      - 已评估过的（在 acc_cache 中）跳过
      - 其余先用代理预测，挑预测最优的 N_INFILL 个用 proxy_eval 真实评估并回填，
        剩下的直接用代理预测值
    返回本代真实评估（infill）的数量。
    """
    for g in offspring:
        ensure_cost(g, cost_cache)

    unknown = [g for g in offspring if str(g) not in acc_cache]
    if not unknown:
        return 0

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
def save_generation(gen, population, fitness, acc_cache, cost_cache, result_dir):
    result_dir.mkdir(parents=True, exist_ok=True)
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
    (result_dir / f"gen_{gen:03d}.json").write_text(json.dumps(out, indent=2))
    return pareto


# ── 主循环 ─────────────────────────────────────────────────────────────────────
def main():
    rng = random.Random(SEED)
    np.random.seed(SEED)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    proxy_genomes: list[list[int]] = []   # 真实(proxy)评估过的架构
    proxy_accs:    list[float]     = []   # 对应准确率 —— 代理的训练集
    acc_cache:  dict = {}                 # str(genome) -> (acc, mode)
    cost_cache: dict = {}                 # str(genome) -> params_M
    surrogate = SurrogateModel()

    print("=" * 60)
    print(f"NAS 主循环  POP={POP_SIZE}  GEN={N_GEN}  每代回填={N_INFILL}")
    print("=" * 60 + "\n")

    # ── Gen 0: 整个种群真实评估，作为代理初始训练集 ──────────────────────────────
    population = [random_genome(rng) for _ in range(POP_SIZE)]
    print(f"[Gen 0] warmup: {POP_SIZE} 个架构全部 proxy_eval …")
    t0 = time.time()
    for i, g in enumerate(population):
        acc, el = proxy_eval(g)
        proxy_genomes.append(g)
        proxy_accs.append(acc)
        acc_cache[str(g)] = (acc, "proxy")
        ensure_cost(g, cost_cache)
        print(f"  [{i+1:>2d}/{POP_SIZE}] acc={acc:.4f} "
              f"params={cost_cache[str(g)]:.3f}M  {el:.0f}s")
    print(f"  warmup 完成，耗时 {(time.time()-t0)/60:.1f} min\n")
    fitness = build_fitness(population, acc_cache, cost_cache)
    save_generation(0, population, fitness, acc_cache, cost_cache, RESULT_DIR)

    # ── 进化循环 ──────────────────────────────────────────────────────────────
    for gen in range(1, N_GEN + 1):
        print(f"[Gen {gen}/{N_GEN}]")
        t0 = time.time()

        # 1. 用全部已积累的 proxy 样本重新拟合代理（每代仅一次）
        surrogate.fit(proxy_genomes, proxy_accs)

        # 2. 选精英（NSGA-II）
        n_elite = max(2, int(POP_SIZE * ELITE_RATIO))
        elite_pop, _ = select_next_population(population, fitness, n_elite)

        # 3. EDA 采样子代 + 突变
        n_offspring = POP_SIZE - n_elite
        offspring = [mutate(g, rng) for g in eda_sample(elite_pop, n_offspring, rng)]

        # 4. 评估子代（代理预测 + top-N_INFILL 真实回填）
        n_real = eval_offspring(offspring, surrogate, proxy_genomes, proxy_accs,
                                acc_cache, cost_cache)
        offspring_fit = build_fitness(offspring, acc_cache, cost_cache)

        # 5. 合并父代 + 子代，NSGA-II 选回 POP_SIZE
        population, fitness = select_next_population(
            population + offspring,
            np.vstack([fitness, offspring_fit]),
            POP_SIZE,
        )

        # 6. 记录本代 Pareto 前沿
        pareto = save_generation(gen, population, fitness, acc_cache, cost_cache, RESULT_DIR)
        best_acc  = max(p["val_acc"]  for p in pareto)
        best_cost = min(p["params_M"] for p in pareto)
        print(f"  proxy样本={len(proxy_genomes)} 本代真实评估={n_real}  "
              f"Pareto={len(pareto)} best_acc={best_acc:.4f} "
              f"min_params={best_cost:.3f}M  {time.time()-t0:.0f}s\n")

    # ── 输出最终 Pareto 前沿 ──────────────────────────────────────────────────
    print("=" * 60)
    print("最终 Pareto 前沿（val_acc 为 proxy/代理估计，需 retrain_pareto.py 复训）：")
    fronts = fast_non_dominated_sort(fitness)
    final = []
    for i in fronts[0]:
        g = population[i]
        acc, mode = acc_cache[str(g)]
        final.append((acc, cost_cache[str(g)], g, mode))
    final.sort(key=lambda x: -x[0])
    for acc, cost, g, mode in final:
        print(f"  val_acc={acc:.4f}  params={cost:.3f}M  [{mode:9s}]  genome={g}")

    (RESULT_DIR / "final_pareto.json").write_text(json.dumps(
        [{"genome": g, "val_acc": round(a, 6), "params_M": round(c, 4), "mode": m}
         for a, c, g, m in final], indent=2))
    (RESULT_DIR / "proxy_samples.json").write_text(json.dumps(
        {"genomes": proxy_genomes, "accs": proxy_accs}, indent=2))
    print(f"\n结果已保存至 {RESULT_DIR}/")
    print(f"  - final_pareto.json   最终 Pareto 架构")
    print(f"  - proxy_samples.json  全部 {len(proxy_genomes)} 个真实样本（复盘用）")


if __name__ == "__main__":
    main()
