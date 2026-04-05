# vis

[![Gitee](https://img.shields.io/badge/Gitee-vis-blue)](https://gitee.com/yinlb97/vis)
[![GitHub](https://img.shields.io/badge/GitHub-vis-black)](https://github.com/yinlb3/vis)

> 气象能见度数据分析与可视化的 Python 脚本集合。

## 功能特性

- **数据处理**：读取和处理气象站点能见度、降水、相对湿度等观测数据
- **分级统计**：按能见度六级分级（<50m, <200m, <500m, <1000m, <2000m, <10000m）进行统计
- **预报检验**：计算 ME、MAE、RMSE、MRE、R、TS、ETS、HSS、TSS 等多种检验指标
- **可视化**：绘制柱状图、箱线图、小提琴图、饼图、热图、空间分布图等

## 安装

无需安装。确保已安装以下依赖：

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

运行前请确认：
1. `D:\data\vis\` 目录包含所需的 `.npy`、`.csv`、`.xls` 等数据文件；
2. 系统已安装中文字体（脚本会查找 `msyh.ttc`、`simhei.ttf` 等）；
3. 内存充足（部分脚本会加载大型 numpy 数组并进行循环计算）。

## 数据来源

- 输入数据: `D:\data\vis\` (需在运行前准备)
- 输出结果: `D:\Project\vis\figures\`

## 项目里程碑

### 已完成
| 任务 | 描述 | 完成时间 |
|------|------|----------|
| 代码注释 | 为 draw.py 添加中文注释和 Google 风格文档字符串 | 2026-04-02 |
| Git 配置 | 配置 .gitignore，排除 IDE 配置和输出文件 | 2026-04-02 |
| 双平台同步 | 推送到 Gitee 和 GitHub | 2026-04-03 |
| 逻辑错误检查 | 静态检查发现 draw.py 存在变量未定义、类型注解错误等问题 | 2026-04-03 |
| 大文件拆分 | 将 `draw.py` (~936 行) 按 9 个阶段拆分为 `src/` 下的 13 个模块 | 2026-04-04 |
| 公共函数提取 | 提取 `VisAcc`、`format_time` 到 `src/`；优化 CDF 计算 | 2026-04-04 |
| 文档翻译 | 翻译 README.md 和 AGENTS.md 为英文 | 2026-04-05 |

### 进行中
| 任务 | 描述 | 备注 |
|------|------|------|
| 代码重构 | 继续消除 `access.py`、`tl.py`、`ots.py` 中的重复代码 | — |

### 待办
| 任务 | 描述 | 优先级 |
|------|------|--------|
| 添加单元测试 | 为 VisAcc 类添加 pytest 测试 | P2 |
| 路径配置化 | 将硬编码路径改为配置文件 | P2 |

## 许可证

本项目为科研用途开发，仅供内部使用。

## 作者

- **作者**: yinlb <yinlb3@foxmail.com>

---

英文版见 [README.md](README.md)。
