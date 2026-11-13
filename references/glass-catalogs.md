# 本地玻璃目录、精确色散与选材

本模块把用户取得的 AGF 文件变成可追溯的本地材料索引，供成像光学处方分析和离散材料候选搜索使用。支持按厂商、牌号、目录版本、nd/Vd、色散有效波段和数据一致性查询；光线追迹使用原始色散系数，不能用 nd/Vd 代替实际波长的折射率。

厂商目录正文不随公开插件分发。官方来源、核查日期及文件哈希见 [来源注册表](glass-catalog-sources.json)。下载与更新都须显式调用；分析、追迹和优化过程不会联网更新材料，也不会自动重写旧处方。

## 本次核查的目录

核查日为 **2026-11-09**。“当前”表示该日官网提供的相应下载，不代表未来日期仍为最新版，也不代表文件内每种玻璃都在产。

| 来源 ID | 版本与范围 | 记录数 | 默认当前目录 |
|---|---|---:|---|
| `cdgm-current` | CDGM 202609，目录头为 2026 年 9 月；未确认发布日期的具体日 | 352 | 是 |
| `schott-current` | SCHOTT June-2025-B，preferred/inquiry/AR 目录 | 366 | 是 |
| `ohara-current` | OHARA 2026-05-29，当前范围目录 | 166 | 是 |
| `ohara-all` | OHARA 2026-07-01，含特殊订购和停产类型的全目录 | 433 | 否 |

SCHOTT 下载包的 HTTP 修改时间为 2026-02-06，但其 AGF 的目录头为 June 2025、文件名带 `June-2025-B`，不能将其写成“2026 修订目录”。OHARA 两个目录的用途不同，不能因为全目录日期更晚，就将它替代当前范围目录。OHARA 两库有重叠记录，表中的数量不能相加当成独立牌号数。

官方入口：[CDGM 下载](https://www.cdgmgd.com/go.htm?k=ge_lei_xia_zai&url=downList)、[SCHOTT 光学玻璃](https://www.schott.com/en-us/products/optical-glass)、[OHARA 目录](https://www.ohara-inc.co.jp/en/product/catalog/)。

## 下载已核查版本

以下命令从插件根目录运行。`python` 指用户已经配置的 Python 环境，输出目录应放在用户的私有数据区。

```powershell
python scripts/update_glass_catalogs.py --list-sources
python scripts/update_glass_catalogs.py --out "$env:USERPROFILE\.codex\skills\standalone-optics\local-data\glass-20261109"
```

默认取得三个标记为 `default_current` 的目录。需要 OHARA 全目录时显式选择：

```powershell
python scripts/update_glass_catalogs.py --out "D:\optics-private\glass-with-history-20261109" --include-historical
```

也可用重复的 `--source` 指定来源 ID，例如 `--source ohara-current`。`--from-cache` 用已取得的原始下载包离线导入，仍会核对注册表中的包哈希和 AGF 哈希。缓存布局为 `<cache>/<source_id>/<archive_name>`；具体文件名见来源注册表。

此命令取得的是注册表固定的已核查字节。服务器文件改变时，哈希校验会拒绝，不能自动把新响应认作已验证版本。更新来源注册表前应重新核对官网、日期依据、包内文件和哈希，再写入新的输出目录或索引文件。下载时间、官网核查时间、HTTP 修改时间及目录版本日期是不同信息。

## 导入用户自己的目录

可直接导入本机已有 AGF，不需要网络：

```powershell
python scripts/glass_catalog.py import "D:\Zemax\Glasscat\SCHOTT.AGF" --index "D:\optics-private\local-glass\glass-index.json" --vendor SCHOTT
```

`import` 接受多个 AGF 路径或一个包含 AGF 的本地目录。只有精确的 `CDGM`、`SCHOTT`、`OHARA` 文件主名可提供默认厂商身份；任意重命名文件应显式给 `--vendor`，多厂商目录可用 `--metadata` 文件逐个声明。文件名里的年份不会自动变成经过验证的版本。

`--metadata` 接受以输入文件名或绝对路径为键的 JSON。例如：

```json
{
  "SCHOTT.AGF": {
    "vendor": "SCHOTT",
    "catalog_name": "SCHOTT_LOCAL",
    "scope": "user_selected_local_catalog",
    "default_current": true
  }
}
```

```powershell
python scripts/glass_catalog.py import "D:\Zemax\Glasscat\SCHOTT.AGF" --index "D:\optics-private\local-glass-v2\glass-index.json" --metadata "D:\optics-private\import-metadata.json"
```

手动指定 `default_current` 是用户的目录选择，不是“官方最新”证明。本地导入默认显示 `local_unverified_release`。只填写官网地址和版本说明，也不能证明本地字节对应该版本；只有明确的 AGF SHA-256 与核查元数据绑定，才显示对应的已声明官方版本匹配状态。`latest_verified` 还必须有明确的 `latest_verified_as_of`，且其含义只限于该日期；今天复制旧缓存或重新取得相同字节，不会把官网核查日推进到今天。

索引文件不覆盖已有文件。原始 AGF 复制到其旁边的 `catalog-snapshots/<SHA256>.agf`；以后更换源目录文件不会改变已有快照。精确计算会核对快照哈希。复制到另一台电脑时，应同时复制索引和整个 `catalog-snapshots` 目录；来源路径是历史出处，计算使用相对快照路径。

## 查询与精确折射率

```powershell
python scripts/glass_catalog.py list --index "D:\optics-private\glass\glass-index.json" --vendor OHARA --current-only
python scripts/glass_catalog.py search BK7 --index "D:\optics-private\glass\glass-index.json" --vendor SCHOTT --current-only --eligible-only
python scripts/glass_catalog.py search --index "D:\optics-private\glass\glass-index.json" --nd-min 1.45 --nd-max 1.65 --vd-min 50 --wavelength 0.355 --eligible-only --current-only
python scripts/glass_catalog.py lookup N-BK7 --index "D:\optics-private\glass\glass-index.json" --vendor SCHOTT --current-only
python scripts/glass_catalog.py index N-BK7 --index "D:\optics-private\glass\glass-index.json" --vendor SCHOTT --current-only --wavelength 0.5875618
```

波长单位为 µm。`index` 返回原 AGF 公式计算的折射率及材料出处；`lookup` 返回目录元数据和警告，不以模糊匹配选取同名材料。

- `--current-only` 只选择索引显式注册的当前目录 ID，排除 OHARA 全历史目录。普通本地导入没有注册时，此过滤不会自动猜出一个目录。
- `--catalog-id` 指定包含厂商、目录名和快照 SHA 的完整身份。同一厂商/牌号存在多个目录版本时，应指定 ID，或显式使用当前目录注册表。
- `--ready-only` 表示当前内核支持其公式、系数及已知有效波段；`--eligible-only` 进一步排除系数与 NM 元数据严重不一致的记录。
- `--status preferred`、`--status obsolete` 或原始状态码可过滤 AGF 状态。原始码与解释来源同时保存；厂商自定义含义仍需核对。

“可以精确计算折射率”只表示色散数据可供当前求解器计算，不等于该材料适合本项目。

## 色散、透过率与加工信息分开

当前精确支持 AGF 公式 1（Schott）、2（Sellmeier 1）、3（Herzberger），其余公式明确阻止计算。缺失有效波段、系数不足或六系数公式带未定义的非零额外系数，也会阻止使用；不以近似玻璃代替，不作波段外外推。

CDGM 的 11 条 Herzberger 记录是 HWS 红外材料，必须按各自 LD 范围计算；支持公式 3 并不使它们成为 355 nm 候选。可见光 d/F/C 不在完整有效域时，不做相应 nd/Vd 一致性检查。

索引保留下列信息：

| AGF 记录 | 保存内容 | 使用范围 |
|---|---|---|
| NM、CD、LD | 原始 nd/Vd、状态、系数字段、色散有效范围及快照身份 | 精确色散与一致性诊断；LD 不代表透过率 |
| IT | 采样波长、内透过率、参考厚度及原始字段 | 展示已有数据；不自动补齐未知波长或指定其它厚度的透过率 |
| ED | 两个温区的 CTE、密度、dPgF 和原始字段 | CTE 按 AGF 的 `1e-6/K` 转换；加工/热设计另行评价 |
| TD | D0/D1/D2/E0/E1/Ltk、参考温度及原始字段 | 当前追迹使用参考温度色散；解析 TD 不代表已做热补偿 |
| OD | 相对 BK7 的成本及 CR/FR/SR/AR/PR 字段 | 缺失值保留未知，不当成实际报价或供应证明 |

采购、实际内透过率、荧光、激光损伤、均匀性、应力双折射、可购尺寸与加工方式仍需本项目证据。nd/Vd 和 LD 不能证明 355 nm 透明，也不能证明低自发荧光。

## 已发现的厂商数据异常

CDGM 202609 的 H-TK9，NM 写为 `nd=1.587166、Vd=75.902131`；同一文件 CD 系数在标准谱线计算约为 `nd=1.62075001、Vd=60.30091368`。CDGM [官方牌号表](https://www.cdgmgd.com/database/toWebDatabase.htm?k=Products_Data&pageIndex=11&url=database)的 H-TK9 列值为约 `1.62075、60.30`，支持其 NM 元数据存在异常。

模块保留厂商原文件和 NM/CD 原值，显示差异并默认阻止该记录的折射率应用与处方材料准备。不会擅自改写官方文件。`index --allow-flagged-data` 或 Python 的 `allow_flagged_data=True` 仅供明确调查原始数据；覆盖行为会记录，不会把记录改成“已修复”。

一般一致性诊断在完整 LD 范围内比较 d/F/C 系数结果与 NM，使用保守预警阈值 `|Δnd|>1e-4` 或 `|ΔVd|>0.1`，容纳目录的数值舍入。这是数据筛查阈值，不是镜头或材料验收限值，也不是独立的 Zemax 结果验证。本次 CDGM 352 条记录的公式均受支持，其中 H-TK9 被隔离，默认可准备 351 条；不能据此宣称所有条目已经通过实物或采购认证。

## 接入处方与材料候选搜索

```python
from ai_zemax.standalone_glass_catalog import load_index

catalog = load_index("D:/optics-private/glass/glass-index.json")
record = catalog.lookup("N-BK7", vendor="SCHOTT", current_only=True)
index_at_d = catalog.refractive_index("N-BK7", 0.5875618,
                                    vendor="SCHOTT", current_only=True)
candidates = catalog.search(vendor="SCHOTT", nd_min=1.45, vd_min=50,
                            wavelength_um=0.55, current_only=True,
                            selection_eligible_only=True)
prepared, provenance = catalog.prepare_indices(model, {
    1: {"vendor": "SCHOTT", "name": "N-BK7", "current_only": True}
})
```

`prepare_indices` 的键是**面后介质索引**：索引 0 为物方介质，索引 i 为第 i 面之后的介质。不得把镜片编号或 `surface_rows` 的零基索引当成该编号。只允许替换已声明的实玻璃介质，不靠材料准备函数增加或删除镜片。

默认保留未选择通道的原折射率表；只有显式选择的通道重算。`recompute_existing=True` 才请求重算已有材料。新建且没有折射率表的模型，会根据明确的目录身份生成材料表。全部波长槽位、重复波长及权重保持，不能合并重复槽位来改变光谱要求。

CLI 可从 JSON 选择表执行同样的动作：

```json
{
  "1": {"vendor": "SCHOTT", "name": "N-BK7", "current_only": true}
}
```

```powershell
python scripts/glass_catalog.py prepare --index "D:\optics-private\glass\glass-index.json" --model input.json --selections material-selection.json --out material-variant.json
```

输入处方不覆盖。结果附每个已解析介质的厂商、目录 ID、快照 SHA、逐波长折射率和版本出处，去除旧的 `nominal_verification` 并标记须重新验收。换玻璃之后，先恢复组功率、共轭及通光，再优化像差并复核全部工程边界。材料库扩大不会自动放开项目的材料约束；例如固定只用 C79-80/CaF2 的 UV355 项目，仍须遵守该项目规则。

## 数据与公开发布

公开仓库只保留代码、合成测试、来源注册表和方法说明。用户自己的 AGF、压缩包、完整玻璃索引及快照留在私有本地数据目录；官网可下载不等于获得再分发许可。迁移到其他电脑时也应遵守相应厂商许可。

格式字段依据 [Ansys AGF 说明](https://optics.ansys.com/hc/en-us/articles/46891219767827-Zemax-AGF-material-catalog-file-format)。解析、名义色散计算、项目选材、光学验收和制造放行分别记录完成状态。
