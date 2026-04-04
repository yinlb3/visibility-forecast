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
├── src/             # 公共模块（由 draw.py 拆分而来）
│   ├── vis_acc.py   # VisAcc 类与 THRES 常量
│   ├── utils.py     # 通用工具函数
│   ├── data_prep.py # 观测数据准备与预处理
│   ├── data_stats.py# 观测数据统计与输出
│   ├── plot_obs.py  # 观测数据可视化
│   ├── forecast_prep.py    # 预报数据加载与整体检验
│   ├── forecast_eval.py    # 分类型预报检验与绘图
│   ├── temporal_eval.py    # 时效分析与站点/类型检验
│   ├── plot_temporal.py    # 时效特征热图与柱状图
│   ├── plot_spatial.py     # 站点级空间分布图
│   ├── plot_geo_mre.py     # MRE 与地理要素关系图
│   ├── plot_mre_violin.py  # MRE 改善率小提琴图
│   └── case_study.py       # 2024 年独立样本个例分析
├── draw.py          # 主绘图脚本（仅剩 main() 调度入口）
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
- **最后修改**: 2026-04-04

## 最近更新

- **2026-04-04** 完成 `draw.py` 大文件拆分：将原来近 1000 行的主脚本按 9 个阶段重构为 `src/` 下的 13 个模块；公共函数 `VisAcc`、`format_time`、`plot_weather_type_eval_bw` 统一提取到 `src/`；CDF 计算优化为 `np.searchsorted`，复杂度从 O(N·M) 降至 O(M log M)。

## 已知问题

- **类型注解错误**: `VisAcc.get_mre1()` 返回类型标注为 `float`，实际返回 `np.ndarray`
- **硬编码路径**: 输入/输出路径仍写死在代码中，迁移时需批量替换

## 许可证

本项目为科研用途开发，仅供内部使用。
