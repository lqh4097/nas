# NAS 项目 — 面向边缘硬件感知的多目标进化神经架构搜索

> 研究生课题：面向 RK3566 边缘设备的鸟鸣识别轻量化模型自动搜索。导师汇报用。

## 核心目标

在 RK3566（4×Cortex-A55, NPU 0.8 TOPS）上，用 NSGA-II/MOEA/D 多目标进化算法自动搜索最优鸟鸣分类轻量模型，联合优化分类精度和推理延迟。

| 指标 | 最低 | 理想 |
|------|------|------|
| Top-1 Accuracy | > 80% | > 90% |
| 推理延迟 | < 50ms | < 20ms |
| 参数量 | < 2M | < 500K |
| INT8 模型大小 | < 5MB | < 1MB |

## 当前进度 (2026-06-04)

### 已完成
- [x] 研究方案设计（[研究方向_终案_副本.md](研究方向_终案_副本.md)）
- [x] 学术汇报 PPT 初版（21 页，[projects/enas_research_report_ppt169_20260528/](projects/enas_research_report_ppt169_20260528/)）
- [x] xeno-canto 数据爬取脚本（[crawl_xeno_canto.py](crawl_xeno_canto.py)）
- [x] 数据集鸟种确认（7 种，全部 B 级 ≥100 条）

### 进行中
- [x] 运行 `crawl_xeno_canto.py` 下载鸟鸣数据（7 种 × 100 条 = 700 条）
- [ ] baseline CNN 训练代码（test.py 是空文件）

### 待开始
- [ ] 数据预处理 pipeline（resample → mel-spectrogram → .npy）
- [ ] MobileNetV3 基线训练
- [ ] 搜索空间代码（search_space.py）
- [ ] NSGA-II 主循环（moe_nas.py）
- [ ] 改进 A（分层初始化）、改进 B（代理模型）
- [ ] 实验 + 论文撰写

## 数据集

| 鸟种 | 学名 | B 级量 | 频率特征 |
|------|------|--------|----------|
| 大杜鹃 | Cuculus canorus | 677 | 低频标志声 |
| 乌鸫 | Turdus merula | 1106 | 1.5–3.5 kHz |
| 大山雀 | Parus cinereus | 1580 | 2–6 kHz |
| 麻雀 | Passer montanus | 356 | 高频短促 |
| 雕鸮 | Bubo bubo | 134 | 极低频 0.2–1 kHz |
| 喜鹊 | Pica serica | 378 | 1–4 kHz |
| 远东山雀 | Parus minor | 120 | 2–7 kHz |
| 背景噪声 | — | — | Freesound/ESC-50 补充 |

数据采集：
- **方式 A（主）**：xeno-canto API v3，`crawl_xeno_canto.py`，质量 ≥B，时长 5-30s
- **方式 B（补充）**：Kaggle BirdCLEF 2024
- **方式 C（自采）**：实验室数据

## 项目文件结构

```
NAS/
├── CLAUDE.md                    ← 本文件（项目说明）
├── crawl_xeno_canto.py          ← 数据爬取脚本
├── test.py                      ← 空文件，待写 baseline
├── 研究方向_终案_副本.md          ← 完整研究方案
├── 研究计划_副本.md              ← 初步想法
├── .venv/                       ← Python 虚拟环境
├── bird_data/                   ← 鸟鸣原始数据（crawl_xeno_canto.py 下载）
└── projects/
    └── enas_research_report_ppt169_20260528/   ← PPT 项目（已完成 21 页）
```

## 常用命令

```powershell
# Python 用虚拟环境
.venv\Scripts\python.exe <script.py>

# 数据爬取
.venv\Scripts\python.exe crawl_xeno_canto.py --dry-run    # 查看数据量
.venv\Scripts\python.exe crawl_xeno_canto.py              # 正式下载
.venv\Scripts\python.exe crawl_xeno_canto.py --species 0  # 只下载某一种
```
