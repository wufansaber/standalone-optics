# 初始结构搜索与 CPU 粗筛

## 从规格生成可追迹的起点

```powershell
python scripts/optimize_optics.py seed --demo --workers 4 --dense-top 8 --verify-serial --out seed-report.json --best-model-out best-seed.json
python scripts/standalone_optics.py --model best-seed.json --aberrations --plot-dir plots --out analysis.json
```

内置算例是 240 mm、−2×、物方 NA 0.035 的有限共轭系统，物方最大场半径 2 mm。它先在四段轴向间距和中组光焦度上采样，通过 ABCD 的 `A=m` 和 `B=0` 求首末组光焦度，再按功率负担、符号和间距差异保留种子。第二步把三个薄组转换成对称球面厚透镜，重新匹配主波长倍率和共轭，检查球面定义域、通光半径、边缘厚度和空气间隙。实光线粗筛后，可用 `--dense-top` 对前 K 个候选复排。最优种子可以导出为私有中性模型 JSON。

把以下内容保存为 `seed-config.json` 后用 `seed --config seed-config.json` 调整规格。省略 `realization` 时仅生成近轴种子。提供空对象时使用默认的示例厚度、波长和折射率。自己的镜头使用逐波长逐组的折射率表。

```json
{
  "search": {
    "track_mm": 240,
    "magnification": -2,
    "gap_bounds_mm": [[30, 70], [25, 100], [25, 100], [30, 80]],
    "power_bounds_mm_inv": [[-0.07, 0.07], [-0.07, 0.07], [-0.07, 0.07]],
    "object_na": 0.035,
    "object_field_radius_mm": 2,
    "stop_group": 2,
    "max_group_radius_mm": 24,
    "samples": 4096,
    "count": 24,
    "random_seed": 0
  },
  "realization": {
    "center_thicknesses_mm": [3, 3, 3],
    "wavelengths_um": [0.5876],
    "refractive_indices": [[1.52, 1.52, 1.52]]
  }
}
```

间距依次为物面到首组中心、两个组间距、末组中心到像面。光阑组编号为 1–3。仅在同轴、有限共轭、空气中使用此三片模板。厚透镜会重算首末组功率，并按配置重新检查半径与功率上限。报告中的 `calculated` 表示完成这一级实光线筛选。实际宽视场、高 NA 或近无焦校正组需要有对应内部结构的多片模板。

## 用已有处方做多核粗筛和局部迭代

```powershell
python scripts/optimize_optics.py screen --model lens.zmx --glass-dir Glasscat --config screen-config.json --workers 8 --rounds 3 --dense-top 8 --verify-serial --out screening.json
```

输入也可用中性模型 JSON。曲率变量使用从 S1 开始的光学面编号，改变量单位为 mm⁻¹，区间须包含零。以下为示例。

```json
{
  "candidate_count": 48,
  "random_seed": 0,
  "curvature_variables": [
    {"surface": 1, "min_delta_mm_inv": -0.000001, "max_delta_mm_inv": 0.000001},
    {"surface": 2, "min_delta_mm_inv": -0.000001, "max_delta_mm_inv": 0.000001}
  ],
  "screening": {"field_indices": [0, 1], "wave_indices": [0]}
}
```

- 粗筛可选的视场和波长索引从零开始。也可配置 `wfe_limit_waves`、`spot_limit_um` 做筛选门槛。缺省使用处方保存的全部视场和波长。
- 首轮包含原处方、单变量边界和分层样本。后续轮次在当前领先者周围缩小扰动窗口，继续使用已预载数据的进程池。排序依次看最坏 RMS 波前、平均 RMS 波前和最坏 RMS 光斑。报告保留失败原因、每轮领先候选和改变量。
- `--dense-top` 使用 GQ20×64 复核入围处方的所有保存视场和波长，原处方作为对照一并计算。`--verify-serial` 额外重算串行结果并核对逐项一致性，单列进程启动和预热时间。
- 当前局部迭代固定曲面顶点和像面。筛选结果用于挑选下一步候选，完整倍率、场曲、机械间隙和制造约束按项目口径另行复核。命令将结果写入单独的 JSON。
