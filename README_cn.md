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

```powershell
pip install -r requirements.txt
```

## 使用方法

```powershell
# 运行主绘图脚本
python draw.py

# 运行其他分析脚本
python access.py
python tl.py
python ots.py

# 业务化流程
python build_near_map.py
python preprocess.py 2026071700
python inference.py 2026071700
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
| 评估计算并行化 | 使用 `joblib.Parallel(n_jobs=-1)` 并行化 `src/p3_1_eval_calc.py` 中的站点级和起报时段指标计算，运行时间降至原来的约 1/3 | 2026-04-15 |
| 代码规范合规与路径配置 | 系统性检查并修复代码规范项，硬编码路径迁移至配置文件 | 2026-05-09 |
| 代码重构 | 重构 `access.py`、`tl.py`、`vis2411.py`、`ots.py`、`vis_grade.py`、`huanghua.py`；可复用类（`PDF`、`OTS`）迁入 `src/pdf_model.py`、`src/ots_model.py`；硬编码路径改为 `config.yaml`；统一入口与英文注释 | 2026-07-16 |

### 进行中
| 任务 | 描述 | 备注 |
|------|------|------|
| 业务化推理流程 | 实现 `preprocess.py`、`inference.py` 的实时与回算运行，由 0/1/2 个 YYYYMMDDHH 命令行参数驱动；新增 `src/logger.py`、`src/model_registry.py` 与 `config/operational.yaml` | 实现完成，服务器测试待进行 |

### 待办
| 任务 | 描述 | 优先级 |
|------|------|--------|
| 特种行业风险产品研发 | 研发面向交通、航空等特种行业的能见度风险产品 | P1 |
| 预处理并行化 | 研究内存感知的 GRIB 插值任务动态并行方案（joblib / 批量调度），加速服务器上多起报时次预处理 | P2 |
| 多模式数据应用 | 将多模式预报数据应用于能见度分析与产品服务 | P2 |

## 作者

- **作者**: yinlb <yinlb3@foxmail.com>

---

**最后更新**：2026-07-26

英文版见 [README.md](README.md)。
