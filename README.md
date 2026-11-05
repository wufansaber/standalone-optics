# standalone-optics

用于 Codex 的独立光学计算技能。用 Python 和 NumPy 对同轴顺序折射系统做光线追迹、RMS 波前、RMS 光斑、轴向和横向色差、球差、场曲、像散、光线扇形图与衍射 MTF。运行计算无需安装 Zemax。

## 安装

使用有仓库访问权限的 GitHub 账号，将本仓库克隆到 Codex 的技能目录。以下是 PowerShell 命令。若设置了 CODEX_HOME，使用其中的 skills 目录。

```powershell
git clone https://github.com/wufansaber/standalone-optics.git "${HOME}/.codex/skills/standalone-optics"
python -m pip install "numpy>=1.26,<3"
```

需要 Python 3.10 或更新版本。打开新的 Codex 对话后可调用 `$standalone-optics`。也可以在仓库根目录直接运行内置示例。

```powershell
python scripts/standalone_optics.py --demo --out demo.json
```

## 计算常用像差

```powershell
python scripts/standalone_optics.py --model lens.zmx --glass-dir Glasscat --aberrations --mtf --mtf-all-waves --out analysis.json
```

`--aberrations` 输出几何像差曲线。`--mtf` 输出像方频率（lp/mm）上的多色 MTF，追加 `--mtf-all-waves` 可同时输出各单色曲线。两类分析可独立运行。计算受支持的 ZMX 时请提供自己的 AGF 材料目录。也可使用预先导出的中性模型 JSON。仓库包含技能代码、说明和验证摘要。

- [安装与使用](references/install-and-use.md)
- [验证结果与适用范围](references/validated-scope.md)
