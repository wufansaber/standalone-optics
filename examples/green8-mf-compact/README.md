# 下一版：MF 工作集、预算与绿光精简案例

2026-11-13。使用[已验收的同一14片处方](../green8-14element/README.md)，新增编译和工作集功能；本次没有重新优化或改变光学结构。原始大MF及其下载包保留。

- [下载精简版 Zemax、MF、源行图与依赖报告](MF_COMPACT_NEXT.zip)
- [算法、适用条件与预算接口](../../references/merit-function-budget.md)
- [行数与输入身份](summary.json) · [原生验证和实测](native-validation.json) · [软件验证](validation.json)

## 实际变化

| 用途 | 源行数 | 新行数 | 波前查询行 |
|---|---:|---:|---:|
| 核心等价编译 `CORE_EQUIVALENT` | 23,030 | 22,126 | 510→510 |
| 含焦面引导的等价编译 `FULL_EQUIVALENT` | 203,427 | 202,523 | 510→510 |
| 核心工作集 `WORKING_CORE` | 23,030 | **6,375** | 510→80 |
| 含焦面引导的工作集 `WORKING` | 203,427 | **18,454** | 510→80 |

等价编译仅合并904行重复依赖，保留所有源行映射和各个有权重根。主要收益来自显式工作集：光学代表样点、已知局部最坏点，以及经过证明的球面采样冗余精简。后者保持源采样几何可行集合，改变越界时的罚函数；完整工作集还停用了部分光学采样约束，因此不是原目标函数的等价替换。

工作视场为0、0.3125、0.625、0.9375、1.25 mm，8个绿光波长的 RWCE/RWRE 均保留完整原积分参数。焦面引导选择570、580、595、610、620 nm；595 nm场曲保留轴上、边缘及冻结局部最坏物高0.8833608046477396 mm。保留全部原生控制行、固定窗口和接口、一阶规格、面通光见证及非冗余实体/定义域根。蓝光与790 nm像质监测行不再全部放入工作MF；全15波长物理通光仍属于强制外部验收。

`WORKING_CORE` 不含完整焦面图，只适合相应阶段的工作引导；需要轴向色差和场曲引导时使用 `WORKING`。两者都不能替代原独立验收。

## 原生验证与耗时

OpticStudio 2022 R2.02：冻结处方及 S21 半径±0.001%两次扰动，共3个点，分别核对4种MF的全部映射非控制行。最大绝对行值差约1.00×10⁻¹⁰（各行按自身单位），等价编译另检查整体MF数值。4套原生ZMX均保存重载，确认13个配置、行数和物理处方一致。扰动点用于语义验证，不作为新的合格设计。

| 配对测量，3次中位数 | 原MF单次评价 | 工作MF单次评价 | 本次比值 |
|---|---:|---:|---:|
| 核心 | 1.244 s | 0.208 s | 5.97× |
| 含焦面图 | 1.304 s | 0.210 s | 6.22× |

仅测 `CalculateMeritFunction`，排除加载、保存和Python解析；同一原生实例交替顺序。没有测Jacobian、实际优化迭代或最终验收耗时，不推算整次设计加速。18个逻辑CPU可用于原生进程，两个保留给界面；本机结果不作为其他电脑保证。

原生输入字节与公开重建结果逐一一致。公开验证器仅统一行尾为LF；执行时的SHA与发布文件的SHA分别记录。输入元数据中的本地路径已隐去，未改变原生操作数或处方。

## 下载与复现

ZIP 中 `Zemax/WORKING.ZMX`、`.MF` 为含焦面引导的工作版，`WORKING_CORE` 为核心工作版；另附两套 `EQUIVALENT` 对照。ZMX嵌入原生已计算的MF；独立MF按编译结果导出，缓存值初始为零，加载后计算即可。所有变量保持固定，释放变量、换玻璃或改面半径后重新执行完整验收。不得继续使用不再满足固定面半径假设的旧冗余证书。

`Inputs/*.json.gz` 提供标准库gzip压缩的源图和必要元数据；`*_REPORT.json` 包含一基行号映射、停用根和成本统计，`SPHERICAL_SUPPORT.json` 为冗余证明清单。`CASE_MANIFEST.json` 校验全部文件。整份厂家OHARA目录不随包发布；按原案例相同的SHA自行提供。

```powershell
# 解压下载包后校验、准备自己的冻结玻璃目录
python verify_and_prepare_catalogs.py --case-dir . --verify-only
python verify_and_prepare_catalogs.py --case-dir . --ohara-path "D:/Glasscat/OHARA.AGF"

# 在仓库根目录运行，输出目录必须是新目录
python scripts/build_green8_working_merit.py --source "D:/MFCase/Inputs" --out output/mf-rebuild
python -m unittest discover -s tests -v
```

在项目目录模式下打开ZMX，核对 `OH9AD9D2` 和 `FIXWIN` 材料。独立重跑原生对照时，需要解压并准备[原始包](../green8-14element/ZEMAX_AND_MF.zip)，Python环境安装NumPy、pythonnet、psutil以及有效OpticStudio许可。下列验证器只启动一个自有实例，在新目录复制工程，关闭时只关闭该实例：

```powershell
python examples/green8-mf-compact/verify_native.py `
  --source-project "D:/OriginalCase/Zemax" --graphs "D:/MFCase/Inputs" `
  --compiled output/mf-rebuild --out output/mf-native-check `
  --install "C:/Program Files/Ansys Zemax OpticStudio 2022 R2.02" --cpu-count 18
```

若pythonnet装在独立目录，可加 `--pythonnet-dir`。原生对照需要Windows和匹配版本API；普通MF编译不需要原生许可。包中的验证器副本应从仓库上述位置运行，或使用仓库中的同名脚本，以便导入插件模块。

## 接受边界

此版本提供固定案例的初始工作集和通用依赖编译，不自动替该案例运行动态原生优化。后续工作集更新按 `run_constraint_exchange` 接入项目求解与独立审计；新的最坏视场/波长需要重新生成相应行图。预算超限会停止生成，不能静默放宽验收。

正式验收仍保留原始绿光波前65场及局部搜索、570–620 nm 51波长焦面、595 nm 65场及局部场曲、全15波长完整NA物理检查和完整实体域。原始导出的203,427行MF使用工作瞳积分，本身也不代替这些独立检查。制造公差、装调及制造放行仍未完成。
