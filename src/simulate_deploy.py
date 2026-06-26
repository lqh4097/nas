"""
simulate_deploy.py
------------------
边缘部署「推理仿真」—— 上 RK3566 真机前的彩排。

用真实训练好的 Pareto 权重 + 真实 test 谱图，在 PC 上模拟设备端推理，给出
部署关心的全套指标，并演示「梅尔谱进 → 鸟种 + 置信度出」的单样本识别链路。

它仿真什么 / 不仿真什么（诚实边界）：
  ✓ 模型功能：谱图 → CNN → 类别 + 置信度（部署推理链路，缺的只有 FFT 那一步）
  ✓ 计算量 MACs：硬件无关，是真机延迟的良好代理（NAS 论文标准指标）
  ✓ 模型体积：FP32 实际大小 + INT8 理论大小（NPU 用 INT8）
  ✓ CPU 延迟：x86 多线程代理，仅用于「相对排序」，不等于 NPU 绝对延迟
  ✗ 音频→梅尔谱：Mel_Augment_four 的生成参数未知，无法忠实仿真（须定参数重训，
                  那是上板那一步）。真机上这一步在 CPU 跑 STFT+梅尔，约几毫秒。
  ✗ NPU 绝对延迟 / INT8 实测掉点：必须上 RK3566 用 RKNN 实测（即「真实测评」）。

用法：
    python src/simulate_deploy.py
    python src/simulate_deploy.py --runs 50 --threads 4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset import BirdDataset
from net_builder import build_net

CKPT_DIR = Path("d:/NAS项目/checkpoints/pareto/full_seed42")
RETRAIN  = Path("d:/NAS项目/results/full/seed42/retrain_result.json")


# ── 计算量（MACs）：forward hook 统计 Conv2d / Linear ────────────────────────────
def count_macs(model: nn.Module, shape=(1, 3, 224, 224)) -> float:
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
        model(torch.zeros(shape))
    for h in handles:
        h.remove()
    return macs / 1e6   # MMACs


def model_size_mb(model: nn.Module) -> tuple[float, float, int]:
    """返回 (FP32 实际大小 MB, INT8 理论权重大小 MB, 参数量)。"""
    n_params = sum(p.numel() for p in model.parameters())
    fp32_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    fp32_bytes += sum(b.numel() * b.element_size() for b in model.buffers())  # BN 统计量
    int8_bytes = n_params * 1   # INT8：每权重 1 字节（NPU 部署形态）
    return fp32_bytes / 1e6, int8_bytes / 1e6, n_params


@torch.no_grad()
def cpu_latency_ms(model: nn.Module, runs: int, warmup: int = 5) -> tuple[float, float]:
    """单样本(bs=1) CPU 推理延迟，返回 (mean_ms, p90_ms)。"""
    model.eval()
    x = torch.zeros(1, 3, 224, 224)
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(ts)), float(np.percentile(ts, 90))


def main():
    ap = argparse.ArgumentParser(description="边缘部署推理仿真（上板前彩排）")
    ap.add_argument("--runs", type=int, default=30, help="延迟测量次数")
    ap.add_argument("--threads", type=int, default=4, help="CPU 线程数（模拟 RK3566 四核 A55）")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)   # RK3566 是 4×Cortex-A55
    torch.manual_seed(42)

    retrain = {r["idx"]: r for r in json.loads(RETRAIN.read_text())}
    ckpts = sorted(CKPT_DIR.glob("arch_*.pth"))
    assert ckpts, f"找不到权重：{CKPT_DIR}"

    print("=" * 88)
    print(f"边缘部署推理仿真   CPU={args.threads}线程(模拟RK3566四核)   延迟×{args.runs}   "
          f"输入=[1,3,224,224]")
    print("=" * 88)
    header = (f"{'架构':<10}{'参数M':>8}{'MACs(M)':>10}{'FP32(MB)':>10}"
              f"{'INT8(MB)':>10}{'CPU延迟ms':>11}{'p90ms':>9}{'test_acc':>10}")
    print(header)
    print("-" * 88)

    rows = []
    for ck in ckpts:
        idx = int(ck.stem.split("_")[1])
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        model = build_net(blob["genome"])
        model.load_state_dict(blob["state_dict"])

        macs = count_macs(model)
        fp32, int8, n_params = model_size_mb(model)
        lat_mean, lat_p90 = cpu_latency_ms(model, args.runs)
        test_acc = retrain.get(idx, {}).get("test_acc", blob.get("test_acc"))

        rows.append(dict(idx=idx, params_M=n_params / 1e6, macs=macs, fp32=fp32,
                         int8=int8, lat=lat_mean, p90=lat_p90, acc=test_acc))

    rows.sort(key=lambda r: r["params_M"])   # 从小到大（部署关心小模型）
    for r in rows:
        print(f"arch_{r['idx']:02d}{'':<4}{r['params_M']:>8.4f}{r['macs']:>10.1f}"
              f"{r['fp32']:>10.2f}{r['int8']:>10.3f}{r['lat']:>11.2f}{r['p90']:>9.2f}"
              f"{r['acc']:>10.4f}")
    print("-" * 88)

    # 推荐部署点：test_acc≥0.98 中参数量最小的
    deployable = [r for r in rows if (r["acc"] or 0) >= 0.98]
    rec = min(deployable, key=lambda r: r["params_M"]) if deployable else rows[0]
    print(f"\n📌 推荐部署架构 arch_{rec['idx']:02d}："
          f"{rec['params_M']:.4f}M / {rec['macs']:.1f} MMACs / "
          f"INT8≈{rec['int8']:.3f}MB / test_acc={rec['acc']:.4f}")

    # ── 单样本识别 demo：梅尔谱 → 鸟种 + 置信度 ───────────────────────────────────
    print("\n" + "=" * 88)
    print("「谱图 → 鸟种」单样本识别 demo（部署推理链路，仅缺真机上的 audio→梅尔谱 FFT 一步）")
    print("=" * 88)
    blob = torch.load(CKPT_DIR / f"arch_{rec['idx']:02d}.pth", map_location="cpu",
                      weights_only=False)
    model = build_net(blob["genome"]); model.load_state_dict(blob["state_dict"]); model.eval()

    test_ds = BirdDataset("test")
    rng = np.random.default_rng(42)
    n_show, n_correct = 5, 0
    for k in rng.choice(len(test_ds), n_show, replace=False):
        x, true_label = test_ds[int(k)]
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(x.unsqueeze(0))
        ms = (time.perf_counter() - t0) * 1e3
        prob = torch.softmax(logits, 1)[0]
        top3 = torch.topk(prob, 3)
        pred = int(top3.indices[0]); n_correct += (pred == true_label)
        top3_str = "  ".join(f"类{int(i):>2d}:{float(p)*100:4.1f}%"
                             for i, p in zip(top3.indices, top3.values))
        mark = "✓" if pred == true_label else "✗"
        print(f"  真实=类{true_label:<2d} → 预测=类{pred:<2d} {mark}  [{top3_str}]  {ms:.1f}ms")
    print(f"\n  demo {n_show} 样本命中 {n_correct}/{n_show}"
          f"（完整 test 集准确率见上表 {rec['acc']:.4f}）")

    print("\n下一步（上板真实测评）：模型→ONNX→RKNN(INT8 量化)→RK3566 实测 NPU 延迟与掉点；"
          "音频→梅尔谱用板上 CPU(librosa/RK音频库)，与本仿真链路对接。")


if __name__ == "__main__":
    main()
