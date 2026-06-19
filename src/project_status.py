"""
project_status.py
-----------------
自动扫描项目，生成《项目进度.md》报告：
  1. 代码文件清单     —— 每个 src/*.py 的功能（取自模块 docstring）+ 产物状态
  2. 数据产物         —— stats.json / split_index.json 的关键内容
  3. 模型与运行结果   —— 从 checkpoint 和 results/*.json 读出准确率等指标

随时重跑即可刷新，不会过时：
    python src/project_status.py
"""

import ast
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

# 每个脚本的角色 + 预期产物（用于「状态」列；功能描述自动取 docstring）
SCRIPT_META = {
    "preprocess.py":            ("数据预处理", DATA / "stats.json"),
    "dataset.py":               ("数据集类（工具模块）", None),
    "baseline_cnn.py":          ("基线：手工 CNN", CKPT / "baseline_cnn_best.pth"),
    "baseline_mobilenet.py":    ("基线：MobileNetV2 0.5× 从零", CKPT / "baseline_mobilenet_best.pth"),
    "baseline_classic.py":      ("基线：ShuffleNet/MobileNetV3", CKPT / "baseline_mobilenet_v3_small_best.pth"),
    "baseline_random_search.py":("基线：随机搜索", RESULTS / "random_search_result.json"),
    "search_space.py":          ("NAS 搜索空间（工具模块）", None),
    "net_builder.py":           ("基因组→网络（工具模块）", None),
    "surrogate.py":             ("代理评估（工具模块）", None),
    "nsga2_eda.py":             ("NAS 主循环（含消融开关）", MAIN_RUN / "final_pareto.json"),
    "retrain_pareto.py":        ("Pareto 架构复训", MAIN_RUN / "retrain_result.json"),
    "project_status.py":        ("本进度报告脚本", None),
}

# 缺 docstring 的文件的功能兜底描述
DOC_FALLBACK = {
    "dataset.py": "PyTorch Dataset：实时读 PNG（RGBA→RGB）+ z-score 归一化 + 训练集 SpecAugment；"
                  "train 扫目录，val/test 读 split_index.json。",
}


def module_doc(path: Path) -> str:
    """安全提取模块 docstring 的首段（不执行文件）。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        doc = ast.get_docstring(tree)
    except Exception:
        doc = None
    if not doc:
        return DOC_FALLBACK.get(path.name, "（无模块说明）")
    # 取 docstring 中第一段有意义的文字（跳过文件名标题行和分隔线）
    for line in doc.splitlines():
        s = line.strip()
        if s and not set(s) <= set("-=") and s != path.name:
            return s
    return doc.strip().splitlines()[0]


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def ckpt_metrics(path: Path) -> dict | None:
    """读 checkpoint 里的标量指标（不关心权重）。"""
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


# ── 各板块 ─────────────────────────────────────────────────────────────────────
def section_code() -> list[str]:
    lines = ["## 一、代码文件清单", "",
             "| 文件 | 角色 | 功能 | 产物 | 状态 |",
             "|------|------|------|------|------|"]
    py_files = sorted(p.name for p in SRC.glob("*.py"))
    for name in py_files:
        role, artifact = SCRIPT_META.get(name, ("（未登记）", None))
        doc = module_doc(SRC / name)
        if artifact is None:
            prod, status = "—", "工具/无产物"
        elif artifact.exists():
            prod, status = artifact.name, "✅ 已生成"
        else:
            prod, status = artifact.name, "⬜ 未运行"
        lines.append(f"| `{name}` | {role} | {doc} | {prod} | {status} |")
    return lines + [""]


def section_data() -> list[str]:
    lines = ["## 二、数据产物", ""]
    stats = read_json(DATA / "stats.json")
    split = read_json(DATA / "split_index.json")
    if stats:
        mean = ", ".join(f"{v:.4f}" for v in stats["mean"])
        std  = ", ".join(f"{v:.4f}" for v in stats["std"])
        lines += [f"- **stats.json**：mean=[{mean}]  std=[{std}]  "
                  f"（抽样 {stats.get('n_samples','?')} 张）"]
    else:
        lines += ["- **stats.json**：⬜ 未生成"]
    if split:
        lines += [f"- **split_index.json**：val {split.get('n_val','?')} 张 / "
                  f"test {split.get('n_test','?')} 张（seed={split.get('seed','?')}）"]
    else:
        lines += ["- **split_index.json**：⬜ 未生成"]
    return lines + [""]


def section_results() -> list[str]:
    lines = ["## 三、模型与运行结果", ""]

    # 模型基线
    lines += ["### 模型基线", "",
              "| 模型 | 参数量(M) | Val Acc | 记录轮 | 状态 |",
              "|------|-----------|---------|--------|------|"]
    for label, fname in [("Manual-CNN",          "baseline_cnn_best.pth"),
                         ("MobileNetV2 0.5×",    "baseline_mobilenet_best.pth"),
                         ("ShuffleNetV2 0.5×",   "baseline_shufflenet_v2_x0_5_best.pth"),
                         ("MobileNetV3-Small",   "baseline_mobilenet_v3_small_best.pth")]:
        m = ckpt_metrics(CKPT / fname)
        if m and "val_acc" in m:
            p = f"{m['params_M']:.3f}" if "params_M" in m else "—"
            lines.append(f"| {label} | {p} | {m['val_acc']:.4f} | "
                         f"epoch {m.get('epoch','?')} | ✅ |")
        else:
            lines.append(f"| {label} | — | — | — | ⬜ 未训练 |")
    lines.append("")

    # 随机搜索
    rs = read_json(RESULTS / "random_search_result.json")
    lines += ["### 随机搜索基线", ""]
    if rs:
        t5 = rs.get("top5_stats", {})
        lines += [f"- 已评估 **{rs.get('n_evaluated','?')}/{rs.get('n_target','?')}** 个架构",
                  f"- 最优 5 个 Val Acc：{t5.get('mean',0):.4f} ± {t5.get('std',0):.4f}",
                  f"- Pareto 前沿：{rs.get('pareto_size','?')} 个架构"]
    else:
        prog = read_json(RESULTS / "random_search_progress.json")
        if prog:
            lines += [f"- ⏳ 进行中：已评估 {prog.get('n_evaluated','?')}/"
                      f"{prog.get('n_target','?')} 个"]
        else:
            lines += ["- ⬜ 未运行"]
    lines.append("")

    # NAS 搜索结果（full 主结果）
    pareto = read_json(MAIN_RUN / "final_pareto.json")
    lines += ["### NAS 搜索（full 组，NSGA-II + EDA + Surrogate）", ""]
    if pareto:
        lines += [f"- 最终 Pareto 前沿：**{len(pareto)}** 个架构"]
        best = max(pareto, key=lambda x: x.get("val_acc", 0))
        lines += [f"- 最高（搜索期估计）Val Acc：{best['val_acc']:.4f} "
                  f"（{best['params_M']}M）"]
    else:
        lines += ["- ⬜ 未运行"]
    lines.append("")

    # 消融实验组扫描
    lines += ["### 消融实验组（RQ3）", ""]
    ablation_rows = []
    for cfg_path in sorted(RESULTS.glob("*/seed*/config.json")):
        cfg = read_json(cfg_path)
        if not cfg:
            continue
        fp = read_json(cfg_path.parent / "final_pareto.json")
        done = "✅" if fp else "⏳"
        n_par = len(fp) if fp else "—"
        ablation_rows.append(
            f"| {cfg.get('ablation','?')} | seed{cfg.get('seed','?')} | "
            f"{cfg.get('gen','?')} | {cfg.get('n_proxy_evals','—')} | {n_par} | {done} |")
    if ablation_rows:
        lines += ["| 组 | seed | gen | 真实评估数 | Pareto | 状态 |",
                  "|----|------|-----|-----------|--------|------|"] + ablation_rows
    else:
        lines += ["- ⬜ 未运行（5 组 × 5 seed = 25 次运行）"]
    lines.append("")

    # 复训真实指标
    retrain = read_json(MAIN_RUN / "retrain_result.json")
    lines += ["### Pareto 架构复训（真实指标）", ""]
    if retrain:
        lines += ["| Test Acc | Val Acc | Params(M) |",
                  "|----------|---------|-----------|"]
        for r in retrain[:10]:
            lines.append(f"| {r['test_acc']:.4f} | {r['val_acc']:.4f} | {r['params_M']} |")
    else:
        lines += ["- ⬜ 未运行（需先跑 nsga2_eda.py 再跑 retrain_pareto.py）"]
    return lines + [""]


def main():
    parts = [f"# 项目进度报告", "",
             f"> 自动生成于 {datetime.now():%Y-%m-%d %H:%M}　"
             f"（重跑 `python src/project_status.py` 刷新）", ""]
    parts += section_code()
    parts += section_data()
    parts += section_results()

    REPORT.write_text("\n".join(parts), encoding="utf-8")
    print(f"报告已生成 → {REPORT}")
    print("\n" + "\n".join(parts))


if __name__ == "__main__":
    main()
