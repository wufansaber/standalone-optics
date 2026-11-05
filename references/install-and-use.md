# 安装与使用

## 在另一台电脑安装

1. 将完整的 `standalone-optics` 文件夹复制到 Codex 的 `skills` 目录。默认 Windows 目录为 `%USERPROFILE%\.codex\skills`，macOS 和 Linux 为 `~/.codex/skills`。如果已设置 `CODEX_HOME`，使用其中的 `skills` 目录。复制后应能直接找到 `skills/standalone-optics/SKILL.md`。
2. 准备 Python 3.10 或更高版本，在所选 Python 环境安装 NumPy。程序运行时只需本地 Python、NumPy、脚本和输入数据。已离线准备 NumPy 的电脑可以直接使用现有环境。

   ```text
   python -m pip install "numpy>=1.26,<3"
   ```

3. 在 Codex 中使用 `$standalone-optics`，提供本机上的处方、材料数据和要检查的指标。也可直接运行技能目录内的 `scripts/standalone_optics.py`。

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
