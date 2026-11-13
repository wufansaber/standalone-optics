# 安装与使用

## 在另一台电脑安装

1. 将完整的 `standalone-optics` 文件夹复制到 Codex 的 `skills` 目录。默认 Windows 目录为 `%USERPROFILE%\.codex\skills`，macOS 和 Linux 为 `~/.codex/skills`。如果已设置 `CODEX_HOME`，使用其中的 `skills` 目录。复制后应能直接找到 `skills/standalone-optics/SKILL.md`。
2. 准备 Python 3.10 或更高版本，在所选 Python 环境安装 NumPy。程序运行时只需本地 Python、NumPy、脚本和输入数据。已离线准备 NumPy 的电脑可以直接使用现有环境。

   ```text
   python -m pip install "numpy>=1.26,<3"
   ```

3. 在 Codex 中使用 `$standalone-optics`，提供本机上的处方、材料数据和要检查的指标。也可直接运行技能目录内的 `scripts/standalone_optics.py`。

需要有约束连续优化时，另外安装 SciPy；解析追迹导数本身只依赖 NumPy。CUDA 是可选后端，普通 CPU 电脑可以执行局部优化和公开案例。

```text
python -m pip install -r standalone-optics/requirements-optimize.txt
python standalone-optics/examples/local-optimizer-20261109/run.py --help
```

适配实际镜头的参数、约束、误差场景和独立验收见[局部优化接口](local-optimization.md)。公开合成镜头的验收分配仅用于复现算法，不能沿用为其他镜头的设计要求。

## 直接运行

把下面的 `standalone-optics` 路径换成复制后的技能目录。使用其他 Python 解释器时替换 `python`。

```text
python standalone-optics/scripts/standalone_optics.py --help
python standalone-optics/scripts/standalone_optics.py --demo --out demo.json
python standalone-optics/scripts/standalone_optics.py --model lens.zmx --glass-dir Glasscat --sampling rect64 --out evaluation.json
python standalone-optics/scripts/standalone_optics.py --model model.json --out evaluation.json
```

先运行内置示例确认 Python 环境和输出。对支持的文本 ZMX，提供授权的本地 AGF 材料目录，脚本根据目录计算折射率。内置色散公式包括 Schott 和 Sellmeier 1。若目标电脑没有材料目录，可在原电脑追加 `--neutral-out model.json`，将已解析的几何和折射率表另存，再把该私有文件带到另一台电脑。便携技能包只包含代码、使用说明和数值复核汇总。

常规计算使用 `--sampling gq20`，也可以指定 `rect32` 或 `rect64`。实际像高视场的 `--field-reference measurement` 按当前波长反求主光线，对齐单色 RMS 操作数。`--field-reference primary` 固定主波长的物方发射条件，适合横向色差及逐光线对照。报告中的 `catalogs_resolved_outside_header` 提示处方目录列表以外的材料来源。和 Zemax 比较前确认它加载了相同目录。

带负传播段的模型在默认模式下会记录失败。只有做数值复现时才加 `--allow-signed-segments`。此时 `signed_comparison_only`、负段涉及的光线数和最短距离会保留在结果中。实体间隙检查应独立进行。

## 模型准备

| 需要固定的输入 | 核对要点 |
|---|---|
| 曲面和材料 | 曲面顺序、顶点间距、曲率符号、圆锥常数、偶次非球面项与每段介质相同。追迹几何用 mm，波长用 µm。折射率来自已知系数或同波长指数表。 |
| 视场和发射 | 区分物高、物方角和像高，明确坐标方向、视场归一化、物距及有效瞳面。高 NA 时先验证发射方向和光阑交点。 |
| 像面和瞳面 | 固定像面，不独立重调焦。核对通光孔径、权重、采样分布及遮拦。相同采样名称仍须核对实际瞳点规则。 |
| 波前参考 | 明确参考球中心和半径、符号、活塞和去倾斜基底。各段累计光程按折射率乘几何路程计算，再传播到参考球求 OPD。 |

如果只有 Zemax 存档光线的发射条件，可先将它们作为两套追迹器的共同输入，验证曲面求交、折射和像面位置。此对照范围标记为“共同发射条件下的追迹复现”。若要验证独立发射，则从处方的物方条件和瞳定义自行生成射线，并另外比对原生发射方向或瞄准后的光阑截点。

## 比较结果

优先保留以下数值，以便别人复查。

- 处方哈希、波长与视场覆盖、两边的采样定义和有效光线数。
- 同一条射线的截点及出射方向差。失效或被遮拦的射线应按原因统计。
- 原始 OPD 的符号及参考差，统一参考后的 OPD 差、RMS 波前差与 RMS 光斑差。
- 子午与弧矢最佳焦点的差。两套程序使用同一像方焦面定义，场曲以同波长轴上平均焦面作基准。

当前适用范围为脚本支持的同轴顺序折射几何与参考球波前。模型转换时应逐项核对光阑、材料和曲面，留存结构一致性的证据。用表中的计算差异描述数值复现精度。衍射像质、偏振、公差与制造放行分别使用各自的模型、门槛和验证数据。

## 常用像差与 MTF

```text
python standalone-optics/scripts/standalone_optics.py --model lens.zmx --glass-dir Glasscat --aberrations --mtf --mtf-all-waves --plot-dir plots --out analysis.json
```

- `aberrations.axial` 包含轴向色差、轴上光线与光轴交点、以同波长近轴焦点为基准的纵向球差。
- `aberrations.fields` 包含固定物方发射下相对主波长的横向色差、光线扇形图、畸变和焦面数据。径向 T/S 扇形图以各波长主光线为基准。用于原生 Ray Fan 对照的固定 X/Y 图以主波长主光线为基准。径向 T/S 扇形的偶对称分量可用于观察彗差与更高阶偶对称项。
- 几何场曲以同波长轴上的 T/S 平均焦面为零。像散取 T−S。有限共轭的物方等效值使用逐波长的一阶轴向放大率换算。近轴 T/S 焦面单独保留，零点为固定像面。UV355 项目的 0.493 µm 限值应应用到物方几何值和项目规定的视场网格。
- `mtf` 各行包含 `frequency_lpmm`、`tangential`、`sagittal` 和采样诊断。默认输出多色曲线，`--mtf-all-waves` 同时输出单色，`--mtf-samples 512` 可复核采样收敛，变化仍明显时可提高到 1024。衍射 MTF 基于标量 FFT，使用固定像面和相干瞳函数。高像差或相位欠采样时默认切换为基于实际像面截点的几何 MTF，按像斑大小选取低频段。多色均合成复 OTF，再求调制度。比较时须统一单位、参考位置和轴方向。出瞳映射明显非线性、采样混叠或高倾角等情况应查看诊断，并与原生分析交叉检查。
- 解析范围和有效性以输出的状态、警告和失败光线记录为准。出现带符号光路时将结果保留为数值对照。MTF 曲线供判读。

### 选择 MTF 方法和频段

- `--mtf-method auto` 是默认设置。大横向像差或高像差下的相位欠采样会触发几何近似。单独的高 OPD 诊断不会触发切换。
- `--mtf-method fft` 强制标量衍射 FFT，可用于追查采样。以 `method` 和 `selection.fft_diagnostics` 确认曲线口径。
- `--mtf-method geometric` 直接用加权光线截点。光斑与色间质心差决定默认频带；`--mtf-max-frequency` 可指定像方上限。若报告光线网格可能混叠，增加采样或缩小频带。几何 MTF 用于高像差系统低频判断，其输出范围和精度与衍射算法分开记录。

### 点阵图和光线扇形图

`--plot-dir` 为每个保存视场输出两张 SVG，并在 JSON 的 `field_plots` 保留路径和点阵数据。点阵图使用固定像面和主波长主光线基准，保留不同颜色之间的位移。两轴统一为 µm 并使用等比例刻度。`--spot-samples 64` 加密点阵图。扇形图采用与 Zemax Ray Fan 一致的固定 Y/Py 和 X/Px 曲线。JSON 同时保留沿实际视场方向的子午和弧矢扇形。`--fan-samples` 可调整瞳面点数。无效光线在曲线上留空，带符号路径复现会标明状态。
