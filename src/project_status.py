"""
project_status.py
-----------------
自动扫描项目，生成《项目进度.md》报告。版面分层：
  - 顶部「进度总览」：一眼看清各阶段状态
  - 实验结果：按阶段组织（模型基线 / NAS 搜索 / 搜索方法基线 / 复训 / 消融）
  - 参考信息：数据产物、代码清单（放最后）

随时重跑刷新，不会过时：
    python src/project_status.py
"""

import json
from datetime import datetime
from pathlib import Path

import torch

ROOT     = Path("d:/NAS项目")
SRC      = ROOT / "src"
DATA     = ROOT / "data"
CKPT     = ROOT / "checkpoints"
RESULTS  = ROOT / "results"
MAIN_RUN = RESULTS / "full" / "seed42"     # NAS 主结果（full 组、seed42）
REPORT   = ROOT / "项目进度.md"

# 模型基线：(显示名, 类别, checkpoint 文件名, 参数量兜底值 M)
# 参数量兜底值用于 checkpoint 未存 params_M（或尚未训练）时显示——架构固定，参数量已知。
BASELINES = [
    ("Manual-CNN",          "核心", "baseline_cnn_best.pth",                 2.508),
    ("MobileNetV2 0.5×",    "核心", "baseline_mobilenet_best.pth",           0.739),
    ("ShuffleNetV2 0.5×",   "核心", "baseline_shufflenet_v2_x0_5_best.pth",  0.383),
    ("MobileNetV3-Small",   "核心", "baseline_mobilenet_v3_small_best.pth",  1.559),
    ("Manual-CRNN",         "选做", "baseline_crnn_best.pth",                0.134),
    ("GhostNet 0.5×",       "选做", "baseline_ghostnet_050_best.pth",        1.358),
    ("EfficientNet-Lite0",  "选做", "baseline_efficientnet_lite0_best.pth",  3.422),
]

# 代码文件：角色描述（功能取自 docstring；放最后作参考）
SCRIPT_ROLE = {
    "preprocess.py":            "数据预处理（mean/std + val/test 划分）",
    "dataset.py":               "数据集类（PNG→张量 + SpecAugment）",
    "baseline_cnn.py":          "基线：Manual-CNN",
    "baseline_mobilenet.py":    "基线：MobileNetV2 0.5×",
    "baseline_classic.py":      "基线：ShuffleNet/MobileNetV3/GhostNet/EffLite",
    "baseline_crnn.py":         "基线：Manual-CRNN（CNN+GRU）",
    "baseline_random_search.py":"搜索方法基线：随机搜索",
    "search_space.py":          "NAS 搜索空间（16 维编码）",
    "net_builder.py":           "基因组 → PyTorch 网络",
    "surrogate.py":             "代理评估（proxy + 随机森林）",
    "nsga2_eda.py":             "NAS 主循环（NSGA-II+EDA+Surrogate，含消融）",
    "retrain_pareto.py":        "Pareto 架构完整复训",
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


# ── 数据采集（供总览与详情共用）────────────────────────────────────────────────
def collect_baselines() -> list[dict]:
    rows = []
    for label, cat, fname, p_fallback in BASELINES:
        m = ckpt_metrics(CKPT / fname)
        done = bool(m and "val_acc" in m)
        params = m.get("params_M") if (done and m.get("params_M") is not None) else p_fallback
        rows.append({
            "label": label, "cat": cat, "done": done,
            "params": params,                          # 始终有值（兜底）
            "val_acc": (m.get("val_acc") if done else None),
            "epoch": (m.get("epoch") if done else None),
        })
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
    """所有 nsga2_eda 运行（含 full 与各消融组），按 config.json 识别。"""
    rows = []
    for cfg_path in sorted(RESULTS.glob("*/seed*/config.json")):
        cfg = read_json(cfg_path)
        if not cfg:
            continue
        fp = read_json(cfg_path.parent / "final_pareto.json")
        rows.append({"cfg": cfg, "pareto": fp, "done": bool(fp)})
    return rows


def _flag(done: bool) -> str:
    return "✅" if done else "⬜"


# ── 报告各板块 ─────────────────────────────────────────────────────────────────
def section_overview(baselines, rs_rows, runs) -> list[str]:
    stats = read_json(DATA / "stats.json")
    split = read_json(DATA / "split_index.json")
    env_ok = bool(stats and split)

    n_done = sum(b["done"] for b in baselines)
    n_core = sum(b["done"] for b in baselines if b["cat"] == "核心")
    n_opt  = sum(b["done"] for b in baselines if b["cat"] == "选做")

    nas_full = read_json(MAIN_RUN / "final_pareto.json")
    retrain  = read_json(MAIN_RUN / "retrain_result.json")
    n_rs_done = sum(r["done"] for r in rs_rows)
    std_nsga2 = any(r["cfg"].get("ablation") == "baseline" and r["done"] for r in runs)
    n_ablation_done = sum(r["done"] for r in runs)

    L = ["## 📊 进度总览", "",
         "| 阶段 | 状态 | 说明 |",
         "|------|------|------|"]
    L.append(f"| ① 环境 & 数据 | {_flag(env_ok)} | "
             f"{'val/test 已划分、stats 已算' if env_ok else '未完成'} |")
    L.append(f"| ② 模型基线 | {n_done}/7 | 核心 {n_core}/4，选做 {n_opt}/3 |")
    L.append(f"| ③ NAS 主搜索 (full) | {_flag(bool(nas_full))} | "
             f"{('Pareto '+str(len(nas_full))+' 个，待复训') if nas_full else '未运行'} |")
    L.append(f"| ④ 搜索方法基线 | {n_rs_done + int(std_nsga2)}/2 | "
             f"随机搜索 {n_rs_done} seed，标准NSGA-II {_flag(std_nsga2)} |")
    L.append(f"| ⑤ Pareto 复训（真实指标）| {_flag(bool(retrain))} | "
             f"{'已出真实 test acc' if retrain else '未运行'} |")
    L.append(f"| ⑥ 消融实验 (RQ3) | {n_ablation_done}/25 | 5 组 × 5 seed |")
    return L + [""]


def section_baselines(baselines) -> list[str]:
    L = ["## ① 模型基线（完整训练，真实 Val Acc）", "",
         "> 任务准确率饱和（基线普遍 97–98%），对比重点在**等准确率下的参数量/延迟**。", "",
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
    pareto = read_json(MAIN_RUN / "final_pareto.json")
    if pareto:
        best = max(pareto, key=lambda x: x.get("val_acc", 0))
        smallest = min(pareto, key=lambda x: x.get("params_M", 9e9))
        L += [f"- 最终 Pareto 前沿：**{len(pareto)}** 个架构（已去重）",
              f"- 最高 proxy acc：{best['val_acc']:.4f}（{best['params_M']}M）",
              f"- 最小架构：{smallest['params_M']}M（proxy acc {smallest['val_acc']:.4f}）",
              "- ⚠️ 上述为 proxy 估计值，仅用于排序；真实指标见 ④ 复训"]
    else:
        L += ["- ⬜ 未运行"]
    return L + [""]


def section_retrain() -> list[str]:
    L = ["## ③ Pareto 架构复训（真实 Test Acc）", ""]
    retrain = read_json(MAIN_RUN / "retrain_result.json")
    if retrain:
        L += ["| Test Acc | Val Acc | Params(M) |",
              "|----------|---------|-----------|"]
        for r in retrain[:12]:
            L.append(f"| **{r['test_acc']:.4f}** | {r['val_acc']:.4f} | {r['params_M']} |")
    else:
        L += ["- ⬜ 未运行（先跑 `nsga2_eda.py`，再 `retrain_pareto.py`）"]
    return L + [""]


def section_search_baselines(rs_rows, runs) -> list[str]:
    L = ["## ④ 搜索方法基线 & 消融实验", ""]

    L += ["**随机搜索（Random Search，多 seed）**", ""]
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
    L.append("")

    L += ["**消融实验组（RQ3 留一法；`baseline` 组即标准 NSGA-II）**", ""]
    if runs:
        L += ["| 组 | seed | gen | 真实评估 | Pareto | 状态 |",
              "|----|------|-----|---------|--------|------|"]
        for r in runs:
            c = r["cfg"]
            L.append(f"| {c.get('ablation','?')} | seed{c.get('seed','?')} | "
                     f"{c.get('gen','?')} | {c.get('n_proxy_evals','—')} | "
                     f"{len(r['pareto']) if r['pareto'] else '—'} | {_flag(r['done'])} |")
    else:
        L += ["- ⬜ 未运行（5 组 × 5 seed = 25 次）"]
    return L + [""]


def section_data_and_code() -> list[str]:
    L = ["## ⑤ 数据产物 & 代码清单（参考）", "", "**数据产物**", ""]
    stats = read_json(DATA / "stats.json")
    split = read_json(DATA / "split_index.json")
    if stats:
        mean = ", ".join(f"{v:.3f}" for v in stats["mean"])
        std  = ", ".join(f"{v:.3f}" for v in stats["std"])
        L += [f"- `stats.json`：mean=[{mean}] std=[{std}]（抽样 {stats.get('n_samples','?')}）"]
    if split:
        L += [f"- `split_index.json`：val {split.get('n_val','?')} / "
              f"test {split.get('n_test','?')}（seed {split.get('seed','?')}）"]
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
             f"刷新：`python src/project_status.py`", "",
             "---", ""]
    parts += section_overview(baselines, rs_rows, runs)
    parts += ["---", ""]
    parts += section_baselines(baselines)
    parts += section_nas()
    parts += section_retrain()
    parts += section_search_baselines(rs_rows, runs)
    parts += ["---", ""]
    parts += section_data_and_code()

    REPORT.write_text("\n".join(parts), encoding="utf-8")
    print(f"报告已生成 → {REPORT}")


if __name__ == "__main__":
    main()
