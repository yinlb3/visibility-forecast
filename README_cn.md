# vis

[![Gitee](https://img.shields.io/badge/Gitee-vis-blue)](https://gitee.com/yinlb97/vis)
[![GitHub](https://img.shields.io/badge/GitHub-vis-black)](https://github.com/yinlb3/vis)

> 气象能见度数据分析与可视化的 Python 脚本集合。

## 功能特性

- **数据处理**：读取和处理气象站点能见度、降水、相对湿度等观测数据
- **分级统计**：按能见度六级分级（<50m, <200m, <500m, <1000m, <2000m, <10000m）进行统计
- **预报检验**：计算 ME、MAE、RMSE、MRE、R、TS、ETS、HSS、TSS 等多种检验指标
- **可视化**：绘制柱状图、箱线图、小提琴图、饼图、热图、空间分布图等

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
| 项目基础建设与文档完善 | 初始化 Git 仓库、配置 .gitignore、搭建双平台同步（Gitee/GitHub）、补充代码注释与文档字符串、翻译所有文档为英文 | 2026-04-05 |
| draw.py 重构 | 将 `draw.py` 扁平化为 3 大部分，直接调用 `src/pX_Y` 模块；删除旧中间包装模块；提取公共工具；统一控制台输出、文件结构及命名规范（区域索引、掩码、混淆矩阵、度量指标、案例研究变量、输出标签）；创建 `docs/ABBREVIATIONS.md` | 2026-04-15 |

### 进行中
| 任务 | 描述 | 备注 |
|------|------|------|
| 代码重构 | 继续消除 `access.py`、`huanghua.py`、`ots.py`、`tl.py` 、`vis_grade.py` 、`vis2411.py` 中的重复代码 | — |

### 待办
| 任务 | 描述 | 优先级 |
|------|------|--------|
| 添加单元测试 | 为 VisAcc 类添加 pytest 测试 | P2 |
| 路径配置化 | 将硬编码路径改为配置文件 | P2 |

## 作者

- **作者**: yinlb <yinlb3@foxmail.com>

---

英文版见 [README.md](README.md)。
