"""
project_status.py
-----------------
自动扫描项目，生成《项目进度.md》详细报告。版面：
  概览 → 进度总览 → 方法实现状态 → 模型基线 → NAS 主搜索（含完整 Pareto）
  → Pareto 复训 + 部署仿真 → 搜索方法基线 & 消融矩阵 → 下一步 & 阻塞
  → 数据产物 & 代码清单

随时重跑刷新，不会过时：
    python src/project_status.py
（其中部署仿真表读取 results/deploy_sim.json，由 simulate_deploy.py 生成）
"""

import json
import re
from datetime import datetime
from pathlib import Path

import torch

ROOT     = Path("d:/NAS项目")
SRC      = ROOT / "src"
DATA     = ROOT / "data"
CKPT     = ROOT / "checkpoints"
RESULTS  = ROOT / "results"
MAIN_RUN = RESULTS / "full" / "seed42"     # NAS 主结果（full 组、seed42）
DEPLOY_JSON = RESULTS / "deploy_sim.json"
REPORT   = ROOT / "项目进度.md"

# ── 项目元信息（静态）──────────────────────────────────────────────────────────
META = [
    ("课题", "面向边缘硬件感知的多目标进化 NAS（NSGA-II + EDA + Surrogate）"),
    ("任务", "鸟鸣梅尔谱分类，40 类（Mel_Augment_four）"),
    ("目标设备", "RK3566（CPU 4×A55，NPU 0.8 TOPS）"),
    ("目标期刊", "Applied Soft Computing / EAAI（一区）；保底 IEEE Access"),
    ("当前阶段", "阶段 1–3 推进中（搜索流程已跑通，参数量占位）"),
    ("⚠️ 关键阻塞", "第二目标暂用**参数量**占位，真机延迟待 RK3566 板子到位再换；"
                  "依赖第二目标的大规模搜索/消融届时需重跑"),
]

# 模型基线：(显示名, 类别, checkpoint 文件名, 参数量兜底值 M)
BASELINES = [
    ("Manual-CNN",          "核心", "baseline_cnn_best.pth",                 2.508),
    ("MobileNetV2 0.5×",    "核心", "baseline_mobilenet_best.pth",           0.739),
    ("ShuffleNetV2 0.5×",   "核心", "baseline_shufflenet_v2_x0_5_best.pth",  0.383),
    ("MobileNetV3-Small",   "核心", "baseline_mobilenet_v3_small_best.pth",  1.559),
    ("Manual-CRNN",         "选做", "baseline_crnn_best.pth",                0.134),
    ("GhostNet 0.5×",       "选做", "baseline_ghostnet_050_best.pth",        1.358),
    ("EfficientNet-Lite0",  "选做", "baseline_efficientnet_lite0_best.pth",  3.422),
]

# 方法组件 / 贡献：(名称, 代码位置, 检测标记函数名)  —— 标记存在即判定已实现
METHOD_ITEMS = [
    ("组件一 · 代理评估（RF + 每代 infill 回填）", "nsga2_eda.py:eval_offspring",      "eval_offspring"),
    ("组件三 · 智能初始化（分层 + 成本引导）",      "nsga2_eda.py:smart_init",          "smart_init"),
    ("贡献② 部分A · 架构感知条件 EDA（门控+联合）", "nsga2_eda.py:conditional_eda_sample", "conditional_eda_sample"),
    ("贡献② 部分B · SG-EDA 代理质量加权",          "nsga2_eda.py:_elite_weights",      "_elite_weights"),
    ("贡献② 部分B · SG-EDA 特征重要度温度",        "nsga2_eda.py:_gene_temperatures",  "_gene_temperatures"),
]

# RQ3 主消融组（6 组）：组名 → 含义
ABLATION_GROUPS = {
    "full":         "完整方案（全开）",
    "no-init":      "关智能初始化（→随机）",
    "no-surrogate": "关代理（→全量真实评估）",
    "no-sgeda":     "关 SG-EDA（→普通计数条件 EDA，隔离贡献②部分B）",
    "no-eda":       "关 EDA（→交叉+变异 GA）",
    "baseline":     "全关 = 标准 NSGA-II",
}
ABLATION_SEEDS_PLANNED = 5

# 下一步 & 阻塞：(任务, 是否现在可做)
TODO = [
    ("抽 ~40 架构验证 proxy 排序可靠性（Spearman>0.8）", True),
    ("把输入分辨率/下采样深度纳入搜索空间（缓解高 MACs 偏置）", False),
    ("第二目标换 RK3566 真机延迟（改 compute_cost 一处）", False),
    ("跑 6 组消融 × 5 seed（待延迟目标定稿，避免重跑）", False),
    ("RQ1 大规模 full（gen50/pop40）× 5 seed", False),
    ("对比 SOTA NAS（NSGA-NetV2 / ProxylessNAS / OFA / MCUNet）", False),
    ("RKNN 上验证 depthwise / SE 算子支持，确认是否回退 CPU", False),
]

# 代码文件：角色描述
SCRIPT_ROLE = {
    "preprocess.py":            "数据预处理（mean/std + val/test 划分）",
    "dataset.py":               "数据集类（PNG→张量 + SpecAugment）",
    "baseline_cnn.py":          "基线：Manual-CNN",
    "baseline_mobilenet.py":    "基线：MobileNetV2 0.5×",
    "baseline_classic.py":      "基线：ShuffleNet/MobileNetV3/GhostNet/EffLite",
    "baseline_crnn.py":         "基线：Manual-CRNN（CNN+GRU）",
    "baseline_random_search.py":"搜索方法基线：随机搜索",
    "search_space.py":          "NAS 搜索空间（16 维编码）",
    "net_builder.py":           "基因组 → PyTorch 网络（MBConv 倒残差）",
    "surrogate.py":             "代理评估（proxy + 随机森林）",
    "nsga2_eda.py":             "NAS 主循环（NSGA-II + 条件EDA + SG-EDA + Surrogate，含消融）",
    "retrain_pareto.py":        "Pareto 架构完整复训",
    "simulate_deploy.py":       "边缘部署推理仿真（MACs/CPU延迟/INT8体积 + 谱图→鸟种 demo）",
    "project_status.py":        "本进度报告生成器",
}


# ── 工具函数 ───────────────────────────────────────────────────────────────────
def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def ckpt_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        return {k: obj[k] for k in ("val_acc", "test_acc", "epoch", "params_M") if k in obj}
    except Exception as e:
        return {"error": str(e)}


def _flag(done: bool) -> str:
    return "✅" if done else "⬜"


def _genome_str(g) -> str:
    return "[" + ",".join(str(x) for x in g) + "]"


# ── 数据采集 ───────────────────────────────────────────────────────────────────
def collect_baselines() -> list[dict]:
    rows = []
    for label, cat, fname, p_fallback in BASELINES:
        m = ckpt_metrics(CKPT / fname)
        done = bool(m and "val_acc" in m)
        params = m.get("params_M") if (done and m.get("params_M") is not None) else p_fallback
        rows.append({"label": label, "cat": cat, "done": done, "params": params,
                     "val_acc": (m.get("val_acc") if done else None),
                     "epoch": (m.get("epoch") if done else None)})
    return rows


def collect_random_search() -> list[dict]:
    rows = []
    for d in sorted((RESULTS / "random_search").glob("seed*")):
        rs   = read_json(d / "random_search_result.json")
        prog = read_json(d / "random_search_progress.json")
        data = rs or prog
        if data:
            rows.append({"seed": d.name, "data": data, "done": bool(rs)})
    return rows


def collect_runs() -> list[dict]:
    rows = []
    for cfg_path in sorted(RESULTS.glob("*/seed*/config.json")):
        cfg = read_json(cfg_path)
        if not cfg:
            continue
        fp = read_json(cfg_path.parent / "final_pareto.json")
        rows.append({"cfg": cfg, "pareto": fp, "done": bool(fp)})
    return rows


def detect_method_status() -> list[dict]:
    src = (SRC / "nsga2_eda.py").read_text(encoding="utf-8")
    out = []
    for name, loc, marker in METHOD_ITEMS:
        out.append({"name": name, "loc": loc,
                    "done": bool(re.search(rf"def {re.escape(marker)}\b", src))})
    return out


# ── 报告各板块 ─────────────────────────────────────────────────────────────────
def section_meta() -> list[str]:
    L = ["## 🎯 项目概览", ""]
    for k, v in META:
        L.append(f"- **{k}**：{v}")
    return L + [""]


def section_overview(baselines, rs_rows, runs) -> list[str]:
    env_ok = bool(read_json(DATA / "stats.json") and read_json(DATA / "split_index.json"))
    n_done = sum(b["done"] for b in baselines)
    n_core = sum(b["done"] for b in baselines if b["cat"] == "核心")
    n_opt  = sum(b["done"] for b in baselines if b["cat"] == "选做")
    nas_full = read_json(MAIN_RUN / "final_pareto.json")
    retrain  = read_json(MAIN_RUN / "retrain_result.json")
    n_rs_done = sum(r["done"] for r in rs_rows)
    std_nsga2 = any(r["cfg"].get("ablation") == "baseline" and r["done"] for r in runs)
    # 消融：按主组统计已完成 seed 数（去重）
    done_by_group = {g: set() for g in ABLATION_GROUPS}
    for r in runs:
        c = r["cfg"]
        if r["done"] and c.get("ablation") in done_by_group:
            done_by_group[c["ablation"]].add(c.get("seed"))
    n_ablation_done = sum(len(s) for s in done_by_group.values())
    n_ablation_total = len(ABLATION_GROUPS) * ABLATION_SEEDS_PLANNED

    L = ["## 📊 进度总览", "", "| 阶段 | 状态 | 说明 |", "|------|------|------|"]
    L.append(f"| ① 环境 & 数据 | {_flag(env_ok)} | "
             f"{'val/test 已划分、stats 已算' if env_ok else '未完成'} |")
    L.append(f"| ② 模型基线 | {n_done}/7 | 核心 {n_core}/4，选做 {n_opt}/3 |")
    L.append(f"| ③ NAS 主搜索 (full) | {_flag(bool(nas_full))} | "
             f"{('Pareto '+str(len(nas_full))+' 个') if nas_full else '未运行'} |")
    L.append(f"| ④ 搜索方法基线 | {n_rs_done + int(std_nsga2)}/2 | "
             f"随机搜索 {n_rs_done} seed，标准NSGA-II {_flag(std_nsga2)} |")
    L.append(f"| ⑤ Pareto 复训（真实指标）| {_flag(bool(retrain))} | "
             f"{('已复训 '+str(len(retrain))+' 个') if retrain else '未运行'} |")
    L.append(f"| ⑥ 消融实验 (RQ3) | {n_ablation_done}/{n_ablation_total} | "
             f"{len(ABLATION_GROUPS)} 组 × {ABLATION_SEEDS_PLANNED} seed |")
    return L + [""]


def section_method_status() -> list[str]:
    L = ["## 🔬 方法实现状态", "",
         "> 三组件加在标准 NSGA-II 之上（三者全关 = `baseline` 组）；"
         "贡献②=代理引导的架构感知 EDA（部分A 条件结构 + 部分B 代理引导）。", "",
         "| 组件 / 贡献 | 代码位置 | 状态 |", "|---|---|---|"]
    for m in detect_method_status():
        L.append(f"| {m['name']} | `{m['loc']}` | {_flag(m['done'])} |")
    return L + [""]


def section_baselines(baselines) -> list[str]:
    L = ["## ① 模型基线（完整训练，真实 Val Acc）", "",
         "> 任务准确率饱和（基线普遍 95–98%），对比重点在**等准确率下的参数量/延迟**。", "",
         "| 类别 | 模型 | 参数量(M) | Val Acc | 状态 |",
         "|------|------|-----------|---------|------|"]
    for b in baselines:
        p = f"{b['params']:.3f}" if b["params"] is not None else "—"
        if b["done"]:
            L.append(f"| {b['cat']} | {b['label']} | {p} | **{b['val_acc']:.4f}** | ✅ |")
        else:
            L.append(f"| {b['cat']} | {b['label']} | {p} | — | ⬜ 未训练 |")
    return L + [""]


def section_nas() -> list[str]:
    L = ["## ② NAS 主搜索（full：NSGA-II + EDA + Surrogate）", ""]
    cfg = read_json(MAIN_RUN / "config.json")
    pareto = read_json(MAIN_RUN / "final_pareto.json")
    if cfg:
        comps = []
        for key, name in [("use_init", "智能初始化"), ("use_surrogate", "代理"),
                          ("use_eda", "EDA"), ("use_cond_eda", "条件结构"),
                          ("use_sg_eda", "SG-EDA")]:
            if key in cfg:
                comps.append(f"{name}{'开' if cfg[key] else '关'}")
        L += [f"- 配置：pop={cfg.get('pop','?')}、gen={cfg.get('gen','?')}、"
              f"真实评估 {cfg.get('n_proxy_evals','?')} 次　组件：{('、'.join(comps)) or '—'}",
              f"- ⚠️ 该批结果为 SG-EDA 实现**之前**所跑（config 无 use_sg_eda 键）；"
              f"val_acc 为 proxy 估计值，仅用于排序，真实指标见 ③", ""]
    if pareto:
        best = max(pareto, key=lambda x: x.get("val_acc", 0))
        smallest = min(pareto, key=lambda x: x.get("params_M", 9e9))
        L += [f"- 最终 Pareto 前沿 **{len(pareto)}** 个（已去重）；"
              f"最高 proxy acc {best['val_acc']:.4f}（{best['params_M']}M）、"
              f"最小 {smallest['params_M']}M（proxy acc {smallest['val_acc']:.4f}）", "",
              "<details><summary>完整 Pareto 前沿（点开）</summary>", "",
              "| proxy Acc | Params(M) | mode | genome |",
              "|-----------|-----------|------|--------|"]
        for p in sorted(pareto, key=lambda x: -x.get("val_acc", 0)):
            L.append(f"| {p['val_acc']:.4f} | {p['params_M']} | {p.get('mode','—')} | "
                     f"`{_genome_str(p['genome'])}` |")
        L += ["", "</details>"]
    else:
        L += ["- ⬜ 未运行"]
    return L + [""]


def section_retrain_deploy() -> list[str]:
    L = ["## ③ Pareto 复训（真实 Test Acc）+ 边缘部署仿真", ""]
    retrain = read_json(MAIN_RUN / "retrain_result.json")
    deploy = read_json(DEPLOY_JSON) or []
    by_idx = {d["idx"]: d for d in deploy}
    if not retrain:
        return L + ["- ⬜ 未运行（先 `nsga2_eda.py` → `retrain_pareto.py`）", ""]

    L += ["> Test Acc 为完整训练后封存 test 集实测；MACs/INT8/CPU延迟 来自 "
          "`simulate_deploy.py`（CPU 为 RK3566 代理，非 NPU 绝对值）。", "",
          "| arch | Test Acc | Val Acc | Params(M) | MACs(M) | INT8(MB) | CPU延迟(ms) |",
          "|------|----------|---------|-----------|---------|----------|-------------|"]
    for r in retrain:
        d = by_idx.get(r["idx"], {})
        L.append(f"| #{r['idx']:02d} | **{r['test_acc']:.4f}** | {r['val_acc']:.4f} | "
                 f"{r['params_M']} | {d.get('macs_M','—')} | {d.get('int8_mb','—')} | "
                 f"{d.get('cpu_lat_ms','—')} |")

    # 部署洞察：在可部署(acc≥0.98)子集里，参数量排序 ≠ 计算量排序
    dep = [d for d in deploy if (d["test_acc"] or 0) >= 0.98]
    if len(dep) >= 2:
        a_p = min(dep, key=lambda x: x["params_M"])     # 参数最小
        a_m = min(dep, key=lambda x: x["macs_M"])       # 计算量最小
        if a_p["idx"] != a_m["idx"]:
            L += ["",
                  f"> 📌 **参数量 ≠ 计算量/延迟**（acc≥0.98 子集）："
                  f"参数最小的 arch_{a_p['idx']:02d}"
                  f"（{a_p['params_M']}M / {a_p['macs_M']} MMACs / {a_p['cpu_lat_ms']}ms）"
                  f"计算量反而**高于**参数更大的 arch_{a_m['idx']:02d}"
                  f"（{a_m['params_M']}M / {a_m['macs_M']} MMACs / {a_m['cpu_lat_ms']}ms）"
                  f"——印证第二目标须换真机延迟，按参数量选会选错部署点。"]
        rec = a_m   # 同精度下计算量/延迟更优者更宜部署
        L.append(f"> 📌 推荐部署点 arch_{rec['idx']:02d}（acc≥0.98 中计算量最小）："
                 f"test_acc={rec['test_acc']:.4f}、{rec['params_M']}M、"
                 f"{rec['macs_M']} MMACs、INT8≈{rec['int8_mb']}MB、CPU {rec['cpu_lat_ms']}ms。")
    return L + [""]


def section_search_baselines(rs_rows, runs) -> list[str]:
    L = ["## ④ 搜索方法基线 & 消融实验", "", "**随机搜索（Random Search，多 seed）**", ""]
    if rs_rows:
        L += ["| seed | 评估 | top5 Acc | Pareto | 状态 |",
              "|------|------|----------|--------|------|"]
        for r in rs_rows:
            d, t5 = r["data"], r["data"].get("top5_stats", {})
            L.append(f"| {r['seed']} | {d.get('n_evaluated','?')}/{d.get('n_target','?')} | "
                     f"{t5.get('mean',0):.4f}±{t5.get('std',0):.4f} | "
                     f"{d.get('pareto_size','?')} | {_flag(r['done'])} |")
    else:
        L += ["- ⬜ 未运行"]

    # 消融矩阵：6 主组 × 已完成 seeds
    done_by_group = {g: {} for g in ABLATION_GROUPS}
    for r in runs:
        c = r["cfg"]
        if c.get("ablation") in done_by_group and r["done"]:
            done_by_group[c["ablation"]][c.get("seed")] = c.get("gen")
    L += ["", "**消融矩阵（RQ3 留一法，6 组 × 5 seed）**", "",
          "| 组 `--ablation` | 含义 | 已完成 seed (gen) | 进度 |",
          "|---|---|---|---|"]
    for g, desc in ABLATION_GROUPS.items():
        seeds = done_by_group[g]
        seed_str = "、".join(f"{s}(g{gn})" for s, gn in sorted(seeds.items())) or "—"
        L.append(f"| `{g}` | {desc} | {seed_str} | {len(seeds)}/{ABLATION_SEEDS_PLANNED} |")
    L += ["", "> 注：`baseline` 组即标准 NSGA-II；代码另保留 `no-condeda` 开关作可选探查，不进主表。"]
    return L + [""]


def section_todo() -> list[str]:
    L = ["## ⑤ 下一步 & 阻塞", "",
         "> 🟢 = 不依赖板子、现在可做；🔴 = 待 RK3566 / 延迟目标定稿。", "",
         "| 任务 | 时机 |", "|---|---|"]
    for task, now in TODO:
        L.append(f"| {task} | {'🟢 现在' if now else '🔴 待板子/延迟'} |")
    return L + [""]


def section_data_and_code() -> list[str]:
    L = ["## ⑥ 数据产物 & 代码清单（参考）", "", "**数据产物**", ""]
    stats = read_json(DATA / "stats.json")
    split = read_json(DATA / "split_index.json")
    if stats:
        mean = ", ".join(f"{v:.3f}" for v in stats["mean"])
        std  = ", ".join(f"{v:.3f}" for v in stats["std"])
        L += [f"- `stats.json`：mean=[{mean}] std=[{std}]（抽样 {stats.get('n_samples','?')}）"]
    if split:
        L += [f"- `split_index.json`：val {split.get('n_val','?')} / "
              f"test {split.get('n_test','?')}（seed {split.get('seed','?')}）"]
    if read_json(DEPLOY_JSON):
        L += [f"- `results/deploy_sim.json`：{len(read_json(DEPLOY_JSON))} 个架构的部署仿真指标"]
    if not (stats or split):
        L += ["- ⬜ 未生成"]

    L += ["", "**代码文件**", "", "| 文件 | 角色 |", "|------|------|"]
    for name in sorted(p.name for p in SRC.glob("*.py")):
        L.append(f"| `{name}` | {SCRIPT_ROLE.get(name, '（未登记）')} |")
    return L + [""]


def main():
    baselines = collect_baselines()
    rs_rows   = collect_random_search()
    runs      = collect_runs()

    parts = ["# 🐦 鸟鸣 NAS 项目进度", "",
             f"> 自动生成 {datetime.now():%Y-%m-%d %H:%M}　｜　"
             f"刷新：`python src/project_status.py`", "", "---", ""]
    parts += section_meta()
    parts += ["---", ""]
    parts += section_overview(baselines, rs_rows, runs)
    parts += ["---", ""]
    parts += section_method_status()
    parts += ["---", ""]
    parts += section_baselines(baselines)
    parts += section_nas()
    parts += section_retrain_deploy()
    parts += section_search_baselines(rs_rows, runs)
    parts += ["---", ""]
    parts += section_todo()
    parts += ["---", ""]
    parts += section_data_and_code()

    REPORT.write_text("\n".join(parts), encoding="utf-8")
    print(f"报告已生成 → {REPORT}")


if __name__ == "__main__":
    main()
