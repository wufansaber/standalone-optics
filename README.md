# standalone-optics

用于 Codex 的成像光学设计与独立验证技能。从项目规格出发，建立候选结构、优化实光线像质，并按统一口径复核。适用于物镜、相机镜头、中继与投影、望远成像和变焦等项目的设计规划。具体结构由可表达该模型的求解器计算。

## 通用设计流程

设计前建立[加工与装配边界](references/manufacturing-boundaries.md)。插件可在 CPU/CUDA 候选评分前执行同轴实体检查，项目配置作为硬门槛。报告区分失败、数据未完成和已通过的名义几何，并保留最薄厚度和最小净距的实际位置及余量。

1. **写清验收口径。** 定义共轭、视场、光谱、NA 或 F 数、探测器、包络和制造约束。把验收项、优化目标和诊断量分开。
2. **建立多个结构起点。** 用一阶成像条件解组功率、间距和光阑位置，再用实光线检查通光、像面和结构可行性。
3. **分级优化。** 低采样筛结构，保留多个结构分支。用多视场、多波长残差精修入围方案，先处理占主导的像差和硬约束。
4. **独立复核与交付。** 加密视场和瞳面，复核最坏工况、实体几何和公差。冻结处方、求解设置及验收结果。

完整方法包括结构搜索、删片和缩短总长、评价函数、CPU/GPU 粗筛、求解器选择和验收模板。UV355 物镜是其中一个应用例子。[通用成像光学设计方法](references/imaging-design-method.md)

## 当前计算覆盖

独立内核使用 Python 和 NumPy 计算受支持的同轴顺序折射系统，运行计算无需安装 Zemax。支持有限和无限物距分析，可输出 RMS 波前、光斑、常用几何像差、点阵图、扇形图及 MTF。内置初始结构生成器面向有限共轭三组模型。离轴、反射、自由曲面和多构型等项目可沿用设计流程，由支持相应模型的求解器承担追迹与验收。[已验证的计算范围](references/validated-scope.md)

## 安装

使用有仓库访问权限的 GitHub 账号，将本仓库克隆到 Codex 的技能目录。以下是 PowerShell 命令。若设置了 CODEX_HOME，使用其中的 skills 目录。

```powershell
git clone https://github.com/wufansaber/standalone-optics.git "${HOME}/.codex/skills/standalone-optics"
python -m pip install "numpy>=1.26,<3"
```

需要 Python 3.10 或更新版本。打开新的 Codex 对话后可调用 `$standalone-optics`。也可以在仓库根目录直接运行内置示例。

```powershell
python scripts/standalone_optics.py --demo --out demo.json
```

## 计算常用像差

```powershell
python scripts/standalone_optics.py --model lens.zmx --glass-dir Glasscat --aberrations --mtf --mtf-all-waves --plot-dir plots --out analysis.json
```

`--aberrations` 输出几何像差曲线。`--mtf` 输出像方频率（lp/mm）上的多色 MTF，追加 `--mtf-all-waves` 可同时输出各单色曲线。自动模式在衍射计算触发回退条件时使用低频几何计算，并标记所用方法。`--plot-dir plots` 输出各视场的点阵图和光线扇形图 SVG。各类分析可独立运行。计算受支持的 ZMX 时请提供自己的 AGF 材料目录。也可使用预先导出的中性模型 JSON。仓库包含技能代码、说明和验证摘要。

- [安装与使用](references/install-and-use.md)
- [验证结果与适用范围](references/validated-scope.md)

## 生成初始结构和并行粗筛

```powershell
python scripts/optimize_optics.py seed --demo --workers 4 --dense-top 8 --out seed-report.json --best-model-out best-seed.json
```

按有限共轭的总长、倍率、组功率和间距生成三组初值。支持把已有处方的曲率候选分批并行追迹，再以密采样复核入围结果。可使用 `--backend cuda` 选择 NVIDIA GPU 粗筛。已有多片组可用 `groups` 保留完整组矩阵并搜索组间空气间距，也可配置联动曲率模式搜索组光焦度。[初始结构与多核筛选](references/optimization.md)
