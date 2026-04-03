# 气象能见度数据分析与可视化

[![Gitee](https://img.shields.io/badge/Gitee-vis-blue)](https://gitee.com/yinlb97/vis)
[![GitHub](https://img.shields.io/badge/GitHub-vis-black)](https://github.com/yinlb3/vis)

本项目是一个气象能见度数据分析与可视化的 Python 脚本集合，主要用于处理中国区域气象站的能见度观测与预报数据，对能见度预报进行分级检验与统计评估，并绘制论文所需的各类统计图。

## 功能特性

- **数据处理**：读取和处理气象站点能见度、降水、相对湿度等观测数据
- **分级统计**：按能见度六级分级（<50m, <200m, <500m, <1000m, <2000m, <10000m）进行统计
- **预报检验**：计算 ME、MAE、RMSE、MRE、R、TS、ETS、HSS、TSS 等多种检验指标
- **可视化**：绘制柱状图、箱线图、小提琴图、饼图、热图、空间分布图等

## 项目结构

```
D:\Project\vis\
├── draw.py          # 主绘图脚本（可视化主入口）
├── access.py        # 预报检验与 PDF 匹配订正
├── tl.py            # 时效试验分析与检验
├── ots.py           # 最优阈值选取
├── schematic.py     # 示意图绘制
├── huanghua.py      # 历史月总簿数据解析
├── vis2411.py       # 原始 CSV 数据转换
├── vis_grade.py     # 能见度等级处理
├── 图/              # 输出目录（图片、CSV、NPY）
└── AGENTS.md        # 项目详细说明
```

## 环境要求

- **Python**: 3.8+
- **核心依赖**: numpy, pandas, matplotlib, seaborn, scipy, arrow
- **领域库**: meteva (国产气象检验工具库)
- **内存**: 建议 ≥ 16GB（处理大型 numpy 数组）

## 使用方法

```powershell
# 运行主绘图脚本
python draw.py

# 运行其他分析脚本
python access.py
python tl.py
```

## 数据来源

- 输入数据: `D:\data\vis\` (需在运行前准备)
- 输出结果: `D:\Project\vis\图\`

## 作者

- **作者**: yinlb
- **创建时间**: 2024-04-18
- **最后修改**: 2026-04-03

## 许可证

本项目为科研用途开发，仅供内部使用。
