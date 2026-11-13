# 有约束局部精修、自动最坏点与设计余量

与 Zemax MF 对接时按 [MF 工作集与预算](merit-function-budget.md) 编译行依赖、保护制造根并测量代价。工作 MF 通过后仍调用固定的完整审计；新增极值由项目适配器生成相应行后再编译，不能凭求解器返回值接受候选。

`scripts/ai_zemax/standalone_local_solver.py` 提供 `EfficientSolveMixin`，把光学问题的实光线评价与求解器分开。它可接入独立追迹、CUDA 批量追迹或能够给出同一定义评价量的其他后端，不内置具体处方与项目验收值。初始结构生成和多结构粗筛继续使用 [optimization.md](optimization.md)；局部精修不能代替结构拓扑搜索。

## 适配器接口

适配器应实现：

| 接口 | 含义 |
|---|---|
| `bounds` | `n×2` 有限参数边界，参数应先按实际响应选择尺度。允许固定参数上下界相等。 |
| `batch(qs)` | 对一批参数做完整评价，每个候选返回一行数据；实光线失败等不可用候选返回 `None`。 |
| `slacks(row)` | 有序字典，每块为非空有限数组；所有约束均按 `slack >= 0` 定义。 |
| `model(q)` | 生成可序列化的完整候选模型，供保存和独立验收。 |
| `condition_scale` | 可选的残差响应 SVD 条件化尺度，缺省为不启用。 |
| `compensate_trial(...)` | 可选的二阶模型误差补偿接口；兼容已有 FieldFit 调用约定。 |

用 `class NewFit(EfficientSolveMixin, ExistingAdapter): ...` 将求解器置于继承顺序前方，然后调用：

```python
report = fit.solve(
    directory=output_dir,
    iters=30,
    start_trust=0.5,
    variant=0.2,
    damping=0.0007,
    use_cones=True,
    second_order=False,
    tolerances={"__default__": 1e-8, "geometry": 1e-10},
    phase2_iterations=10,
    margin_keys=["geometry", "telecentricity", "wavefront"],
    margin_scales={"geometry": 0.02, "telecentricity": 0.01,
                   "wavefront": 0.1},
    strict_validator=independent_project_acceptance,
)
```

此例中的块名与数字仅展示接口。未知块名会报错；必须替换成实际 `slacks` 中的字段与本项目分配。每项硬约束只要出现在 `slacks` 中，便同时保留在两个求解阶段。实际光线通光、实体边界和制造状态还应由适配器的候选接受链检查，不能只用像质软罚项代替。

## 收敛与验收分别记录

第一阶段降低各约束块的最坏违反量，其评价为最大块违反量加 `variant` 倍各块违反量之和。停止采用每块显式数值容差，避免“非负评价值必须下降固定常数，但停止又要求精确为零”导致的近可行空转。改善判断使用相对下降量和较小的数值底限，不要求每一步下降相同绝对值。

- `numerical_converged`：全部求解器约束满足配置的数值停止容差。
- `strict_feasible`：当前采样下原始 `slack` 全部非负，不使用停止容差。
- `final_acceptance`：独立验收回调返回的结论。没有提供回调时明确为 `status="not_run", passed=null`。

数值收敛不会自动将最终验收改成通过。`strict_validator(model)` 可返回布尔值，或含布尔字段 `passed` 的字典，并附带独立加密复核、几何范围及未完成项。限制与采样覆盖范围由项目定义，局部求解器不声称全局最优或制造放行。

## 缓存与信赖域

每个已接受的基点建立一次单边差分雅可比，并缓存签名波前/光斑残差的 QR 压缩与可选 SVD。拒绝候选、只缩小信赖域时，保留这些结果，重新求解子问题并精确追迹新候选。接受新基点后重新建立模型。差分的首选方向失败时试用反向和较小步长；仍无法求导的参数只在当前线性化内冻结，并计入 `derivative_failures`。

信赖域依据 `rho = 实际下降量 / 局部模型预测下降量` 调整。低一致性会缩小步长；预测可靠且使用了大部分可用步长时才扩大。非有限、模型约束违反过大或比可行初始解更差的子问题答案不会进入昂贵追迹。SLSQP 的 `success=False` 仍可能包含可用近似解，因此同时核对有限性、模型约束残差和目标值，记录状态与迭代数。

改变视场、波长、瞳面、机械模型、硬约束定义或参数排列后，应重新调用 `solve` 并传入 `initial_q`，使缓存失效。一次调用内约束块的名称和形状改变会报错。自动最坏点交换应放在外层：粗工作集求解后扫描完整检查集，加入最坏位置，再重启局部求解；最后用独立密采样和另一种瞳面求积复核。

## 签名残差与自定义权重

当数据行提供 `wave_vector`、`spot_vector` 及相应 `wfe_target`、`axis_target`、`spot_target` 时，兼容的 `wfe`、`axis`、`spot` 块使用签名残差的仿射响应范数构造局部模型。QR 压缩保留这个仿射向量的欧氏范数。仅对 RMS 标量作线性近似容易丢失抵消方向；保留向量更适合接近低波前区间的精修。

求解器先核对该范数模型与适配器当前 `slacks` 是否一致，避免把自定义权重替换成单位权重。不一致的自动推断块保留实际测量的标量雅可比，并计入 `cone_linear_fallbacks`。自定义块可显式声明，例如

```python
local_norm_specs = {
    "weighted_wave": {
        "vector": "wave_vector",
        "gain": 2.0 / wave_limit,
        "offset": 2.0,
    }
}
```

它表示 `slack = offset - gain * ||vector||`，`gain` 必须为正，可按块广播。`axis=True` 表示使用 `vector[:, 0]`。显式声明与实测约束不一致时直接报错。兼容 FieldFit 的 `order_scale` 时，主采样的波前/光斑顺序块也使用向量范数响应；其他块仍使用各自测得的标量雅可比。

## 达标后的余量阶段

设置 `phase2_iterations > 0` 后，数值可行时切换到第二阶段：最大化所选物理约束的最小 `slack / margin_scale`，同时保留所有原约束。默认从余量目标中排除以 `_order` 结尾的排序目标，仍保留其原约束。实际项目宜明确指定关键物理字段，并依据公差或工程预算设置 `margin_scales`，避免混合量纲或评价权重支配结果。

第二阶段开始时记录每项原约束的下界 `min(entry_slack, 0)`。原本非负的约束不能变为负值；仅有的入口数值残差不能进一步恶化。候选用精确追迹重算后逐项检查此下界，不能花费停止容差换取目标余量。

需要误差场景优化时，可在适配器 `batch` 中同时评价名义和指定扰动场景，并将所有场景约束作为稳定形状的块返回。场景选择、补偿器数量、行程、垫片分辨率及误差相关性由项目决定。单参数扰动通过不是完整良率结论；先做快速敏感度筛选，再对入围设计检查同时误差和全部验收量。

## 外部导数与计时

`jacobian_provider(q, base, keys)` 可返回按 `slacks` 顺序展开的完整矩阵，形状为 `约束数×参数数`。这样局部模型不再运行逐参数差分探针。若要同时保留签名向量模型，可返回：

```python
{"slacks": slack_jacobian,
 "vectors": {"wave_vector": wave_vector_jacobian,
             "spot_vector": spot_vector_jacobian}}
```

每个向量导数的形状为 `原向量形状 + (参数数,)`。外部导数只提供标量矩阵时使用标量线性模型，不能据此声称具有完整的签名波前导数。解析/自动微分必须在同一定义下与中心差分核对，并检查孔径、遮挡、分支和无效光线附近的适用范围。

返回的 `counters` 包括实际批调用、候选追迹数、线性化次数、导数缓存复用、QR/SVD 次数、接受/拒绝步数及导数失败数。外部导数提供器内部的工作不计入 `traced_candidates`，其耗时计入 `derivatives` 和总时间。`trace` 是所有批追迹耗时，`derivatives` 包含导数所需追迹，因此两者有重叠，不应直接相加。`compression`、`subproblem` 分别记录残差压缩和子问题求解；`optimization_seconds` 不含最终独立验证，`validation` 单列，`seconds` 包含二者。

比较版本时固定输入处方、硬件、环境、参数边界、验收定义和独立复核集，分别报告冷启动/预热情况、达到完整验收的总时间、成功率和最终最小余量。仅减少迭代次数或降低软评价函数不能证明整体改进。

## 2026-11-13：直接SOCP、严格硬门与非线性投影

`solve(..., subproblem_backend="clarabel")` 使用 `standalone_conic.py` 的直接SOCP。依赖安装见根目录 `requirements-optimize.txt`；默认 `"slsqp"` 保持既有行为。`"auto"` 在可用且局部模型可用凸锥表示时选择Clarabel，否则用SLSQP并在子问题记录 `fallback_reason`。显式选择Clarabel而缺依赖或含非凸范数差时会报错，不静默换模型。

QR压缩的向量范数直接进入二阶锥，标量门按原雅可比进入线性约束；两后端使用相同的逐块最大违反量和余量目标。`Solved`/`AlmostSolved`仍须通过独立的局部模型约束残差、边界和目标检查；真实接受层仍完整追迹。条件化保留全部参数及零空间，固定列保持固定。

`hard_gate_keys=["geometry", ...]` 指定始终保持的块。起点必须严格非负；子问题和真实试探都保留这些门。未列出的性能块可在可行性阶段修复，不能用总体评分代替最终逐项验收。

可设置 `trial_retractor(trial, lower, upper)`；它返回同一信赖盒内的新点或None。修正后重新追迹，并以实际总步长重算预测。`standalone_retraction.project_feasible`提供实际非线性约束的有界SLSQP投影；其很小的内部缓冲由调用者按约束单位选择，最终仍按原门严格非负检查。无论求解器状态如何，负的物理余量不给通过。

## 2026-11-13：确定性公差场景与有界补偿器

`standalone_scenarios.ScenarioProblem`接受设计边界、具名场景、物理评价回调和模型构造回调。`evaluate(design, payload, controls)`必须在实际模型/追迹中应用控制量，返回 `gates` 和可选有符号 `vectors`。所有场景共享设计变量，各场景控制量有明确边界并覆盖全部视场/波长；默认不添加补偿器，禁止补偿的场景用空边界列表。

有解析导数时提供 `local_jacobian`，最后一轴按“共同设计变量+该场景局部控制量”排列，`problem.jacobian`负责映射全局列。独立差分测试覆盖变量映射和不会串用其他场景的控制量。任一场景无效或含非有限量，整个候选失败；约束布局改变须新建问题。

完整实际使用见[公开运行案例](../examples/runtime-upgrade-20261113/README.md)及其 `scenario_compensation.py`。合成镜头示例给每个扰动装配一个±0.02mm像面控制量，名义装配没有补偿；这是案例假设，不是任何物镜都允许的装调自由度。所有指定场景通过仍不代表统计良率或制造放行。
