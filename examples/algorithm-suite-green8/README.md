# 六项算法升级：实际14片复消色差验证

这次交付包含可运行代码、冻结输入、原生Zemax/MF、完整结果与失败场景。[算法逻辑与接口](../../references/algorithm-suite.md) · [下载新版Zemax/MF（约4.95MB）](ZEMAX_MF_ALGORITHMS.zip) · [包哈希](package.json)。原始[ZEMAX_AND_MF.zip](../green8-14element/ZEMAX_AND_MF.zip)字节保持不变。

## 实际结果

| 策略 | 原始起点：总时间 / 最坏WFE÷目标 | S21半径−0.01%起点：总时间 / 最坏WFE÷目标 | 完整通过 |
|---|---|---|---|
| finite-difference | 161.43s / 0.943091 | 170.67s / 0.951138 | 2/2 |
| analytic | 62.36s / 0.944749 | 71.31s / 0.946219 | 2/2 |
| lifted-reduced | 75.60s / 0.942919 | 101.72s / 0.972385 | 2/2 |

两起点各运行一次；预热18个工作进程，限制逻辑CPU 0–17，保留18–19给界面，BLAS每进程1线程。每轮最多5次可行性和3次余量迭代，最多3轮，最终均重新执行完整验收。计时包括自适应审计和最终验证，预热另列。[环境](results/benchmark-environment.json)、[逐项结果](results/benchmark.json)、[完整优化历史](results/benchmark-details.json.gz)均公开。解析策略相对逐参数差分在这两个起点分别约2.59×、2.39×；样本量小，不能当作其他处方或硬件的普遍加速比。零空间/跨度辅助量策略也通过，但本次比完整解析空间更慢，保持为显式可选路径。

原生闭环实际运行两轮，工作MF从15,153增至15,164行，波前查询48次，74次实际原生工作MF评价；总记录时间100.88s（含原生启动、审计与重载，源图预处理另计）。外部受约束求解器驱动原生MF评价，限定两个曲率变量；此路径不代表完整46维解析优化。最终独立WFE/目标=0.970173205893，LCA约0.582313μm，FC约0.535928μm，所有原始名义门通过。两套ZMX/MF分别保存重载，非控制操作数最大差约1e−11。[原生记录](results/native.json)。

51个焦点原有1,275对、2,550个上下界，数值epigraph只需103个不等式；原生MAXX/MINN形式采用连续代理和一个跨度上限。完整MF由203,427行编译到196,375行，仍保留计算焦点所需的统计量。三处曲率核对原生值。跨度可行集合保持，罚函数改变；不能把边界数量缩减当作整体评价同倍数加速。

材料搜索复核冻结允许目录的全部166种玻璃、61波长，在四个位置取12个全色散候选，再加入一组双材料替换、原玻璃与三种几何起点，共42分支。追加预算[1,2,4]分配104个局部迭代单位，全部等额跑满需294；厚系统恢复、失败尝试、加密精修和验收另计，不能据此宣称实际CPU耗时节省。8个入围分支最终全部通过原验收，其中7个采用真实玻璃替换。名义最佳保留原材料，WFE/目标=0.935515963707；没有证据证明替换材料必然更优。[目录扫描](results/glass-scan.json)、[分支预算](results/branches.json)、[各轮淘汰](results/branch-rungs.json)。筛选完成的三个预算轮次复用检查点，修复字段分块后对全部最终处方重新独立审计；筛选分数不作为验收。

公差控制器在23组（1名义+22单误差）固定集合上验证。两次提案改善活动工作集，却恶化完整域，均被过滤并回退；返回起点的完整名义验收仍通过。原案例6组波前失败，前面的名义精修后为2组，本轮公差搜索未再减少：还有2组WFE超限、共5组未通过当前探针门。该结果明确passed=false，制造放行未完成。探针仅17视场GQ12×48+RECT16，不包含同时误差、统计良率或全焦面误差验收。[完整场景与回退摘要](results/robust.json)、[逐约束求解原始记录](results/robust-details.json.gz)。

## 冻结规格与几何审查

原始模型SHA256、规格身份、目录及61波长扩展输入见[INPUT_PINS](INPUT_PINS.json)与[SPECIFICATION](SPECIFICATION.json)。14片成像玻璃及固定0.725mm窗口；球面、NA0.5、物方半视场1.25mm，固定像面395.6035303459926mm。六个指定绿波长上限0.06723036845265126λ，595nm上限0.04568539236791058λ，610nm上限0.08λ；其余蓝光和790nm按原范围监测，所有15波长保留真实物理通光。

完整名义验收采用65视场加局部极值、GQ12×48与RECT16、全瞳缘/主光线及全部物理面；LCA使用570–620nm每1nm共51波长，FC595使用65视场加局部搜索与GQ20×64。LCA/FC物方上限均0.7μm；一阶、倍率、远心性、畸变、六绿光横向色差、物镜D及端口按冻结规格执行，未放宽。完整球面/平肩实体另用80位数值复核。[全验收原始数据](results/full-acceptance.json.gz)、[全部波长RMS](results/all-wavelength-RMS.json)、[候选处方](results/candidate-models.json.gz)。

Tube外径44mm、CA直径42mm。交付原生方案四片全实体最薄厚度约2.03120、2.67100、2.67063、2.03100mm；这包含整个体径。名义分配含0.03mm暂定单误差厚度预算；安装、倒角、支承、面形、应力、热与供应商工艺仍需确认。[原生方案逐片审查](results/engineering-native.json)、[最佳名义方案逐片审查](results/engineering-best-nominal.json)。最薄余量及位置全部保留，名义实体通过与完整制造通过分别记录。

## 重现独立优化

从仓库根目录运行，输出必须使用新目录。默认单进程；下列18核命令按本次20逻辑CPU机器设置，其他机器自行降低。

```powershell
python -m pip install -r examples/algorithm-suite-green8/requirements.txt
python examples/algorithm-suite-green8/run.py --stage baseline --out output/base --workers 18 --cpu-count 18
python examples/algorithm-suite-green8/run.py --stage derivatives --out output/derivatives --workers 18 --cpu-count 18
python examples/algorithm-suite-green8/run.py --stage benchmark --out output/benchmark --workers 18 --cpu-count 18 --rounds 3 --iters 5 --margin-iters 3
python examples/algorithm-suite-green8/branches.py --out output/materials --workers 18 --cpu-count 18
python examples/algorithm-suite-green8/run.py --stage robust --out output/robust --workers 18 --cpu-count 18 --rounds 4 --iters 6 --margin-iters 3 --initial-q output/materials/polish-current-base/RESULT.json
python -m unittest discover -s tests -q
```

公差命令的起点是前一步current-base完整验收后的精修坐标；可以使用任何在同一Design(BASE)坐标中且完整验收通过的RESULT.json，不能把另一个玻璃分支的坐标直接套在原材料上。公开结果保留这次实际检查点处方，重新搜索可能受数值求解和并发排序影响。解析方向差分覆盖11/46维实际模型；回归测试还覆盖超过四视场的分块物理包络与扰动主光线导数，防止内存分块改变约束数量。

重做166材料扫描需自己的冻结私有索引：

```powershell
python examples/algorithm-suite-green8/verify_glass_scan.py --index D:/private/glass-index.json --out output/glass-scan.json
```

该命令逐材料按61波长重算、核对目录哈希和短名单，结果不含厂商全目录，目录成员资格不作为采购确认。

## 重现原生MF闭环与打开交付

直接使用本页ZIP：解压，在新目录按包内说明运行字节检查并提供自己SHA匹配的OHARA目录，打开Zemax/ADAPTIVE_WORKING.ZMX及同名MF；完整审计为ADAPTIVE_FULL_SPAN.ZMX与FULL_SPAN.MF。保留13构型和项目内GLASSCAT。包内包含哈希清单、证明、原生对照、重载记录与独立验收。

重新优化需安装有效OpticStudio/ZOS-API和pythonnet，解压原[14片案例](../green8-14element/ZEMAX_AND_MF.zip)到original，解压[精简案例](../green8-mf-compact/MF_COMPACT_NEXT.zip)到compact，先用原包脚本准备自己精确匹配的目录。随后运行：

```powershell
python -m pip install -r examples/algorithm-suite-green8/requirements-native.txt
python examples/algorithm-suite-green8/native.py --source-project original/Zemax --graphs compact/Inputs --out output/native --install "C:/Program Files/Ansys Zemax OpticStudio 2022 R2.02" --workers 18 --cpu-count 18
```

脚本只创建并关闭自己的一套原生实例，写私有项目副本，首先核对冻结处方/目录及原生源测量值。原生CLI、有限远心发射/移动参考和配置偏移均是本案例适配器；通用内核没有因此获得任意多构型、反射或自由曲面支持。算法总体没有全局最优证明，制造放行保留未完成状态。

本次验证运行171项测试，169项通过、2项GPU环境测试跳过；源文件哈希、编译、技能及案例包校验见[验证清单](validation.json)。
