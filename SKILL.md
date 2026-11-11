---
name: standalone-optics
description: Plan, initialize, optimize, and verify geometric imaging systems with project manufacturing boundaries (成像光学设计、初始结构、镜头优化、加工边界、实体避碰、独立光线追迹、波前、像差、同处方复核). Use for new designs, redesigns, or independent checks. The standalone backend handles supported coaxial sequential refractive models and finite-conjugate seed screening. Route other geometries to a compatible solver.
---

# 成像光学设计与独立验证

从任务类型选择路径。新设计和结构重构按[通用成像光学设计方法](references/imaging-design-method.md)执行。已有处方的计算和同处方对照按下列独立验证流程执行。跨电脑安装见[安装与使用](references/install-and-use.md)。

## 从规格建立和优化设计

1. 固定共轭、视场坐标和半径或全幅口径、光谱、NA 或 F 数、像面和机械要求。把验收限值、优化目标和诊断量分开。先读[加工与装配边界](references/manufacturing-boundaries.md)，建立有来源的项目加工配置，确定实体、机械口径、CT/全径最薄厚度、净距、材料、面型和公差余量。对缺失规格标注假设，单独记录影响选型或验收的未决项。
2. 用一阶模型生成多个可行结构，保留不同组功率、光阑位置和镜组布局。重构已有处方时，以同一基准面上的光线映射为参照，先恢复成像、通光和实体间隙，再恢复像质。
3. 先做低采样可行性和像质粗筛，再精修入围结构。锁定已有硬约束，用精确实光线重算候选。最后独立加密视场、波长和瞳面，复核最坏工况、实体几何及项目要求的公差。
4. 按[验证记录](references/validated-scope.md)选求解器。独立 NumPy 内核用于受支持的同轴顺序折射模型。离轴、反射、自由曲面、多构型或其他超出范围的模型，调用能够完整表示相关特性的求解器。当前没有适用求解器时，交付一阶结构和待验证项，并标明完成阶段。

候选的接受和回退使用硬几何门槛，复核完整机械径向域和内部极值。用户要求全球面时，逐面锁定圆锥常数和全部非球面项为零。通光口径扩大后重新检查实体和支承域。原项目的 0.30 mm 空气隙、厚度分档和限定材料分别保留在原项目配置，跨项目按实际工艺选择。输出光学、名义几何、安装路径及制造验收各自的完成状态。

使用加工配置和候选排名时，先查[当前已知接受风险](references/manufacturing-audit-20261109.md)。六项逐片覆盖和通光包络联动尚未修复，涉及这些条件时需按[当前实现限制](references/manufacturing-boundaries.md)另做检查；名义实体 `passed` 不能推出完整制造或通光验收。

## 已有处方的独立验证

使用 [standalone_optics.py](scripts/standalone_optics.py) 进行可复现的光线追迹和像差评价。脚本在普通 Python 环境中使用 NumPy。

1. 确认任务是独立计算还是与原生软件对照。对照需要同一处方的参考结果，记录其来源、版本、配置和文件哈希。读取源处方并把输出写入独立工作目录。
2. 核对顺序、同轴、折射模型是否可由脚本表达。固定材料或每波长折射率、曲面顶点与面型、孔径光阑、像面、物方发射条件和波长。遇到坐标断点、镜面反射、非顺序部件或其他未实现的面型时，先说明对应系统超出当前验证范围。
3. 运行脚本的 `--help` 查看参数。用 `--aberrations` 输出常用几何像差，用 `--mtf` 输出 MTF，追加 `--mtf-all-waves` 得到单色曲线。用 `--plot-dir` 导出各视场的点阵图和扇形图。`--model` 接受受支持的文本 ZMX 或中性处方 JSON。ZMX 需要 `--glass-dir` 指向用户自己的 AGF 目录。逐波长逐视场检查波前、光斑、焦点、失败光线面号和实际采样数。
4. 做同处方对照时，先比像面截点和方向，再比同一采样下的 RMS 波前与光斑。分别报告原始 OPD 和对齐符号、活塞、倾斜后的差值。T/S 焦面和场曲必须使用同一评价定义。
5. 默认拦截负传播段。用户要复现带符号的 Zemax 计算时，可用 `--allow-signed-segments` 并报告最短路径、目标面和 `signed_comparison_only`。把数值复现结论与实体间隙分开。检查 MTF 的方法字段、频率范围、采样收敛和输出警告，按相同频率、参考位置与坐标约定对比曲线。偏振和公差使用各自的独立验证。

## 内置初始结构和粗筛

按[初始结构与并行粗筛](references/optimization.md)使用 `scripts/optimize_optics.py`。用 `seed` 搜索有限共轭三组种子，用 `screen` 粗筛已有处方的曲率候选，用 `groups` 搜索多片组间距和配置的有界曲率模式。在 `screening.manufacturing_policy` 提供项目配置，使 CPU、CUDA 和密采样复排执行相同几何门槛。缺少机械数据的探索结果保留未完成标记。保存基线和淘汰原因。NVIDIA CUDA 环境可使用 `--backend cuda` 做粗筛，密采样决选走 CPU。先以 `--verify-serial` 核对同一候选集。

## 方法案例

结构调整、分级优化、同处方复核和显式工程边界的实际应用见 [UV355 5 mm 案例](examples/uv355-5mm/README.md)。案例区分已验证的名义性能和待完成的制造验收，处方保持私有。

## 显微高精度加工、公差与反射镜评审

显微成像及高精度球面、非球面、反射镜任务，先读[显微加工与结构约束](references/microscope-manufacturing-constraints.md)，按其[项目登记表](references/microscope-constraint-register.example.json)逐片、逐面、逐接口分配有来源的限值。球面检查半径/曲率、面形、粗糙度、楔角与定心；非球面追加偏离、名义斜率与斜率误差、空间频率及检测基准；反射镜追加折返包络、镀后面形、支承/重力/热和反射波前预算。区分教材参考、供应商能力和项目验收限值，不自动套用统一高精度数值。

把当前运行时支持的几何项写入 `manufacturing_policy`；登记表本身不能传给该参数。其余用兼容后端或证据检查，缺失项保持 `incomplete`。名义面形斜率不等于制造斜率误差；面形 PV/RMS、粗糙度、反射波前和系统波前分别报告。此规则覆盖反射镜评审，不扩展当前折射追迹内核的计算范围。
