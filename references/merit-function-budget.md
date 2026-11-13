# MF 工作集、依赖编译和计算预算

MF 行数是结构成本之一。一次波前或 MTF 计算通常比一行加法昂贵；先统计波前、光线、MTF、算术、有权重根和控制行，再测原生评价与 Jacobian 耗时。完整验收域由规格冻结，工作 MF 的预算按当前阶段和实际耗时配置。本模块不设置跨项目统一行数上限。

## 两种精简必须分清

`scripts/compact_merit.py` 接受有序 JSON 行图，每行包含 `operand`、8 个 `params`、`target` 和非负 `weight`。行号从 1 开始。它只处理已声明语义的操作数，遇到未知操作数、非法引用、非有限值或负权重立即停止。

- **等价编译**：默认保留所有源行的映射，包括诊断行；复用相同状态下相同的测量和表达式。每个有权重根保持独立，目标、权重及约束不变。`CONF` 和 `PRIM` 的作用域进入测量缓存身份；状态不完整时不跨控制段复用。零权重行可能是后续算式的依赖，不能直接删除。
- **工作集编译**：显式列出 `active_rows`、`protected_rows`，保留其传递依赖及全部控制行。未选中的源权重停用并写入报告。必须附完整外部验收的身份；编译器始终输出 `final_acceptance=false`。少量工作点通过不能代表完整设计通过。

一般数学操作数也不能按外观猜依赖：`SUMM` 是两行相加，`OSUM` 是范围求和，`MAXX/MINN` 操作连续行区间。当前编译器不支持这些范围操作数，不能把它们当作两个引用重新编号。[Ansys 官方数学操作数说明](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v26102/en/OpticStudio_User_Guide/OpticStudio_Help/topics/General_Math_Operands.html)

```powershell
python scripts/compact_merit.py --graph source_rows.json --out output/equivalent
python scripts/compact_merit.py --graph source_rows.json --out output/working `
  --working-selection selection.json --max-rows 20000 --max-wavefront-queries 80
```

这里的数值仅演示预算接口。`selection.json` 示例：

```json
{"active_rows":[51,64],"protected_rows":[13,19],"audit_identity":"frozen-spec-domain-and-definition-sha256"}
```

API 为 `ai_zemax.standalone_merit_graph.compact`。`audit` 可接收 `evaluation_seconds`、`jacobian_seconds` 实测值；未测的时间不填。CLI 也可限制 `--max-ray-queries`、`--max-mtf-queries`。超预算会写报告并停止生成原生 MF，不能靠自动删除硬约束满足预算。需要改变工作集时，显式修改后重新编译。

## 球面实体约束的可证明冗余

`standalone_merit_sphere.spherical_support` 只接受已匹配的球面 sag 依赖图和固定面半径、固定平肩模型。在各段内，球面 sag 差对半径单调；两端采样的上下限可蕴含该段全部原采样限值。根号定义域 `1-c²r²≥L>0` 由同一曲率的最大半径约束支配。不同径向点的压圈/外缘约束保持原样。

函数检查实际 DAG、曲率查询、半径裁剪、上下限和正定义域覆盖，任何不匹配即停止。所有保留端点及定义域约束需要继续执行，不能只选当前处方最薄的一点。此化简保持源采样可行集合，**改变不可行区域的加权罚函数**；不适用于一般非球面、活动面半径或任意自由曲面，也不证明连续实体域或完整通光。完整实体检查保持独立。

## 优化与验收闭环

1. 冻结完整域及模型、材料、指标定义的身份，单独定义每个光学目标与制造硬门。
2. 初始工作集覆盖中心、边缘、代表波长和已知最坏点。波前保留完整积分规则，不能随意抽删瞳点再沿用旧权重。制造硬门保持，数学证明冗余的约束另附证书。
3. 工作 MF 优化后，用完整独立域追迹，检查失效光线、波前、焦面、实体及各个约束族的最坏点。可用已有 `run_constraint_exchange` 编排求解和加密。
4. 把新增失败点和移动的极值加入工作集，重建行图、引用映射及局部模型。需要新视场/波长时由适配器生成对应原生行，不能复用旧样点冒充。每轮保留历史、触发原因及预算报告。
5. 预算无法容纳新增点时停止该轮并报告，调整分块策略或显式预算后继续。最后用冻结完整域和加密/局部极值检查验收，不能从工作 MF 的低数值推断通过。

焦面范数和统计量优先由独立后端批量算出；不为每个中间算式无限展开原生行。使用约束活跃集、同指标极值交换或等价的 epigraph 表达时，必须标明可行集合与目标罚函数是否改变。

本版本提供依赖编译、预算、球面支持点证明及案例工作集构建；没有新增通用 Zemax 原生优化服务。固定案例的工作集构建不是动态极值跟踪。现有约束交换控制器需要由项目适配器接入真实求解与完整验收。

[绿光案例的实际行数、原生对照、耗时及下载](../examples/green8-mf-compact/README.md)。原始大 MF 保留作为审计和对照资料。
