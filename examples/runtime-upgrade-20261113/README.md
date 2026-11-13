# 可执行优化内核升级与同预算对照

2026-11-13。本次发布可执行代码：逐片工程约束及新光线包络接受链、解析导数、缓存式局部求解、直接SOCP、最坏点交换、真实目录材料组合，以及确定性公差场景与有界补偿器。

这与[14片绿光Zemax/MF案例](../green8-14element/README.md)分开：本页使用公开合成单片镜头验证算法，无需Zemax或厂商目录。这里的数值不作为其他镜头的默认规格。原14片案例文件保持不变，制造放行状态也保持未完成。

## 运行

在仓库根目录安装可选优化依赖：

```text
python -m pip install -r requirements-optimize.txt
python -m unittest discover -s tests
python examples/runtime-upgrade-20261113/run.py --out output/backend-comparison --repeats 3
python examples/runtime-upgrade-20261113/scenario_compensation.py --out output/compensated-scenarios
```

普通独立分析仍只需NumPy；SLSQP局部优化需要SciPy；直接SOCP额外使用Clarabel。CUDA不参与此处计时。算法接口见[局部优化说明](../../references/local-optimization.md)。

## 相同预算下比较什么

两条路线使用同一4维起点、参数边界、解析导数、35步上限、初始信赖半径0.5、6步余量阶段及独立审查。唯一选择项是子问题后端：SLSQP或Clarabel。运行顺序交替，分别运行3次。

实际光线覆盖540/550/560 nm、名义及曲率/像面4个扰动场景；最终审查覆盖11个径向物高、GQ12×48与GQ20×48两套完整瞳积分，并复核所有场景的实体与实际光线包络。工作集自动补入独立审查发现的最坏场点。验收为WFE≤0.25 λ、RMS光斑≤18 µm、主光线像高误差≤80 µm，系本合成例分配。

本机最终对照的总耗时中位数为SLSQP **10.079 s**、SOCP **9.536 s**，本算例约缩短 **5.4%**。每次候选评价分别为110与98次，解析导数调用为18与16次；两者每次都经历2轮最坏点交换。

实测结果、环境和逐次耗时见 [benchmark.json](benchmark.json)。两后端均通过本例全部加密光学与物理验收；它们得到不同可行折衷，不宣称各个单项都同时改善。耗时包含导数、构模、子问题、真实追迹、独立加密及实体复核，不仅是求解器内部计时。本例的耗时差不外推为高NA镜头或全部项目的加速倍数。

## 公差与补偿器实例

`scenario_compensation.py`真实移动每个扰动装配的像面，行程限制为±0.02 mm。每个装配的全部波长和视场共用同一个调节量；名义场景不增加补偿器。设计变量跨场景共享。优化后按相同独立加密光学与实体门逐场景复核，结果见 [scenario-compensation.json](scenario-compensation.json)。

`ScenarioProblem`把场景局部的解析雅可比正确映射到共同设计变量和各自的补偿器列。任一场景无效会使整个候选失败；不取可用场景的子集或平均值作为通过。指定场景通过不是统计良率，也不代表任意装调结构都允许独立调焦。

## 发布与验证边界

- 单元/回归测试包含六个原被忽略的逐片字段、共享面和接口优先级、CA/实体越界、严格硬门、导数对照、信赖域内投影、后端缺失和多场景失败路径。
- `hard_gate_keys`必须从非负起点启动，并在每个真实候选上逐项保持；停止容差不给物理硬门额外空间。
- `trial_retractor`返回的修正点必须仍在原信赖盒内，随后完整重新追迹。`project_feasible`用真实非线性约束作有界投影，最终按原约束严格非负检查，不借用优化器成功标志。
- GPU运行需要独立的CuPy/CUDA环境；本次无该环境的测试明确跳过，未报告GPU实测加速。
- 未迁移项目专有的高NA原生参考约定和全部移动OTF适配器。公开通用接口、合成验证与项目专有研究代码的范围分别保留。

本次完整回归运行140项：138项通过，2项因CuPy/CUDA条件缺失而跳过，详见 [validation.json](validation.json)。不安装Clarabel时的兼容检查见 [validation-without-clarabel.json](validation-without-clarabel.json)。发布包含源码哈希，便于将性能数据对应到实测代码版本。
