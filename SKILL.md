---
name: standalone-optics
description: Evaluate coaxial sequential refractive lens prescriptions with standalone NumPy ray tracing, reference-sphere wavefront, and geometric aberrations. Use for Zemax-free lens checks, same-prescription comparisons, or independent optical calculation and verification (独立光线追迹、波前、像差、同处方复核).
---

# 独立光学计算

使用 [standalone_optics.py](scripts/standalone_optics.py) 进行可复现的光线追迹和像差评价。脚本在普通 Python 环境中使用 NumPy。跨电脑安装见[安装与使用](references/install-and-use.md)，结构和指标范围见[验证记录](references/validated-scope.md)。

1. 确认任务是独立计算还是与原生软件对照。对照需要同一处方的参考结果，记录其来源、版本、配置和文件哈希。读取源处方并把输出写入独立工作目录。
2. 核对顺序、同轴、折射模型是否可由脚本表达。固定材料或每波长折射率、曲面顶点与面型、孔径光阑、像面、物方发射条件和波长。遇到坐标断点、镜面反射、非顺序部件或其他未实现的面型时，先说明对应系统超出当前验证范围。
3. 运行脚本的 `--help` 查看参数。`--model` 接受受支持的文本 ZMX 或中性处方 JSON。ZMX 需要 `--glass-dir` 指向用户自己的 AGF 目录。逐波长逐视场检查波前、光斑、焦点、失败光线面号和实际采样数。
4. 做同处方对照时，先比像面截点和方向，再比同一采样下的 RMS 波前与光斑。分别报告原始 OPD 和对齐符号、活塞、倾斜后的差值。T/S 焦面和场曲必须使用同一评价定义。
5. 默认拦截负传播段。用户要复现带符号的 Zemax 计算时，可用 `--allow-signed-segments` 并报告最短路径、目标面和 `signed_comparison_only`。把数值复现结论与实体间隙分开。衍射 PSF/MTF、偏振和公差使用各自的独立验证。
