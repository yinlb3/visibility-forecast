# AGENTS.md

> 本文件供 AI 编程助手阅读，用于快速了解本项目结构、技术栈与开发约定。

---

## 项目概述

本项目是一个**气象能见度数据分析与可视化**的 Python 脚本集合，主要用于：
- 处理中国区域气象站的能见度观测与预报数据；
- 对能见度预报进行分级检验与统计评估；
- 绘制论文所需的各类统计图（柱状图、箱线图、小提琴图、饼图、空间分布图等）。

主入口脚本 `draw.py` 已按职责拆分为 `src/` 下的多个模块，公共工具函数（`VisAcc`、`format_time` 等）已统一提取到 `src/` 中，避免重复代码。

---

## 目录结构

```
D:\Project\vis\
├── .idea/                  # PyCharm / IntelliJ IDEA 配置
├── src/                    # 提取出的公共模块
│   ├── vis_acc.py          # VisAcc 类与 THRES 常量
│   ├── utils.py            # 通用工具函数（format_time 等）
│   ├── data_prep.py        # 观测数据准备与预处理（阶段 1）
│   ├── data_stats.py       # 观测数据统计与 CSV 输出（阶段 2）
│   ├── plot_obs.py         # 观测数据可视化（阶段 3）
│   ├── forecast_prep.py    # 预报数据加载与整体检验（阶段 4）
│   ├── forecast_eval.py    # 分类型预报检验与 CDF/频率图（阶段 5-6）
│   ├── temporal_eval.py    # 时效分析与站点/类型检验（阶段 7）
│   ├── plot_temporal.py    # 时效特征热图与对比柱状图（阶段 8.1-8.2）
│   ├── plot_spatial.py     # 站点级 TS4+/RMSE 改善率空间分布（阶段 8.3-8.4）
│   ├── plot_geo_mre.py     # MRE 与地理要素关系图（阶段 8.5）
│   ├── plot_mre_violin.py  # MRE 改善率小提琴图/箱线图（阶段 8.6）
│   └── case_study.py       # 2024 年独立样本个例分析（阶段 9）
├── 图/                     # 输出目录：存放生成的图片、CSV、NPY 等
│   ├── 投消图/             # 论文投稿用图片子目录
│   ├── *.png / *.jpg / *.eps / *.pdf   # 各类可视化成果
│   ├── vis_*.csv           # 中间统计结果（如 vis_hour.csv、vis_sta.csv）
│   └── ...
├── access.py               # 预报检验与 PDF 匹配订正
├── draw.py                 # 主绘图脚本（仅剩 main() 入口）
├── huanghua.py             # 历史月总簿 Excel 数据解析
├── ots.py                  # 最优阈值选取（Optimal Threshold Selection）
├── schematic.py            # 对角线颜色网格示意图绘制
├── tl.py                   # 时效（Temporal Lead）试验分析与检验
├── vis2411.py              # 原始 CSV 能见度数据 → numpy 数组转换
├── vis_grade.py            # 能见度等级处理与多年统计
└── index_sta08.npy         # 站点筛选索引文件
```

> **注意**：数据输入目录固定为 `D:\data\vis\`，不在版本控制内。

---

## 技术栈

- **语言**：Python 3.8（IDE 中配置的 SDK 名称为 `meteva`）
- **核心依赖**：
  - `numpy` — 数组运算
  - `pandas` — 表格数据处理
  - `matplotlib` + `seaborn` — 绘图
  - `scipy` — 统计检验（如 `pearsonr`）
  - `arrow` — 时间解析与格式化
  - `joblib` — 模型序列化/反序列化
- **领域库**：
  - `meteva` — 国产气象检验与可视化工具库（`meteva.base as meb`）

> 本项目**没有** `requirements.txt`、`pyproject.toml`、`setup.py` 等依赖配置文件。运行前请确保上述库已安装。

---

## 如何运行

没有统一的构建或测试命令。每个脚本均为**独立可执行文件**，直接运行即可：

```powershell
python draw.py
python access.py
python tl.py
```

运行前请确认：
1. `D:\data\vis\` 目录下存在脚本所需的 `.npy`、`.csv`、`.xls` 等数据文件；
2. 系统已安装中文字体（脚本在 Windows 默认路径查找 `msyh.ttc`、`simhei.ttf` 等）；
3. 内存充足（部分脚本会加载大型 numpy 数组并进行循环计算）。

---

## 主要模块说明

| 文件 | 职责 | 关键类/函数 |
|------|------|-------------|
| `draw.py` | 主绘图入口。仅剩 `main()` 函数，按 9 个阶段调度 `src/` 各模块完成全部分析与可视化。 | `main`, `format_time` |
| `src/vis_acc.py` | 能见度分级/定量检验指标计算（VisAcc 类）。 | `VisAcc`, `THRES` |
| `src/utils.py` | 通用工具函数。 | `format_time` |
| `src/data_prep.py` | 观测数据准备：读站点、加载观测、筛选区域、分级、月份索引。 | `read_sta`, `load_obs`, `filter_region`, `grade_visibility`, `build_month_index` |
| `src/data_stats.py` | 观测数据统计：月/小时/站点三级统计字典与 CSV 输出。 | `build_month_stats`, `build_hour_stats`, `build_sta_stats`, `save_obs_stats` |
| `src/plot_obs.py` | 观测数据可视化：饼图、箱线图、小提琴图、堆叠柱状图、空间分布图。 | `plot_obs_pies`, `plot_monthly_bars`, `plot_sta_frequency_maps` 等 |
| `src/forecast_prep.py` | 预报数据加载与整体检验指标打印。 | `load_forecast_data`, `load_experiment_preds`, `print_overall_metrics` |
| `src/forecast_eval.py` | 分类型预报检验：天气类型指标、CDF、二维频率图、分级频率柱状图。 | `calc_weather_type_metrics`, `plot_vis_cdf`, `plot_weather_type_eval_bw` 等 |
| `src/temporal_eval.py` | 时效分析：起报时次/预报时效/预报时间检验，类型与站点指标。 | `calc_temporal_metrics`, `calc_type_metrics`, `calc_sta_metrics` 等 |
| `src/plot_temporal.py` | 时效可视化：hour_access 热图、TS4+ 对比柱状图。 | `plot_hour_access_heatmaps`, `plot_ts_comparison_bars` |
| `src/plot_spatial.py` | 空间分布可视化：站点 TS4+、RMSE 改善率地图。 | `plot_sta_ts4_maps`, `plot_rmse_improvement_map` |
| `src/plot_geo_mre.py` | MRE 与经纬度高程的散点图及线性拟合。 | `plot_geo_mre_relations` |
| `src/plot_mre_violin.py` | MRE 与改善率的小提琴图/箱线图。 | `plot_mre_violins` |
| `src/case_study.py` | 2024 年独立样本个例分析。 | `load_2024_preds`, `analyze_case_studies` |
| `access.py` | 预报订正与检验。实现 PDF 匹配订正（`PDF` 类）以及分级/定量检验指标计算（`VisAcc` 类）。 | `PDF`, `VisAcc` |
| `tl.py` | 时效试验分析。对不同起报时效的预报进行滑动加权平均，并计算检验指标。 | `VisAcc`, `main` |
| `ots.py` | 最优阈值选取。基于训练样本搜索使 TS 评分最大的阈值，再对预报进行分段线性映射。 | `OTS`, `VisAcc` |
| `schematic.py` | 绘制 24×24 对角线颜色网格示意图，用于展示模式循环同步方案。 | `create_diagonal_matrix`, `draw_diagonal_grid` |
| `huanghua.py` | 读取 2013–2023 年历年月总簿 Excel，提取场面气压、修正海平面气压、温度、相对湿度等要素。 | `get_n_days`, `main` |
| `vis2411.py` | 将 1951 年以来的逐月 CSV 能见度数据整理为 `vis.npy`。 | `main` |
| `vis_grade.py` | 对能见度进行六级分级统计，并生成多年站点级统计表格。 | `main` |

### 关于 `VisAcc` 类

`VisAcc` 类已统一迁移至 `src/vis_acc.py`，用于计算能见度分级检验指标：
- **定量指标**：ME、MAE、RMSE、MRE、Pearson 相关系数 R
- **分级指标**：TS、ETS、HSS、TSS、BIAS、FAR、MAR、POD、OA、Kappa
- 能见度分级阈值（米）定义为全局常量：
  ```python
  THRES = (10000., 2000., 1000., 500., 200., 50.)
  ```

### 关于 `format_time`

`format_time` 已统一迁移至 `src/utils.py`，用于将秒数格式化为可读字符串，例如 `'43.5 seconds'`、`'12.5m'`。

---

## 项目特有约定

> 注：通用编码规范（文档规范、导入规范、函数/类规范等）遵循全局 SKILL.md 2.2 节。

1. **拼音缩写命名**
   - 变量命名混用拼音缩写与英文，如 `hxjz`（混淆矩阵）、`cjzxy`（长江中下游）。

2. **内存管理习惯**
   - 绘图脚本中频繁使用 `plt.close(fig)`、`del fig, ax`、`gc.collect()` 来释放 matplotlib 占用的大量内存。

3. **硬编码路径**
   - 输入数据路径硬编码为 `D:\data\vis\...`
   - 输出路径硬编码为 `D:\Project\vis\图\...`
   - 修改或迁移项目时，需要全局替换这些路径。

4. **数据清洗约定**
   - 缺失值常以 `999990` 或 `999999` 形式出现，读取后统一替换为 `np.nan`。
   - 能见度上限常截断为 `30000` 米。

5. **Shebang 写法**
   - 文件头使用 `#!user/bin.python3`（非标准 Linux shebang）。

---

## 测试说明

- **没有单元测试框架**（无 `pytest`、`unittest`）。
- 验证方式：运行脚本后检查控制台输出的检验指标，以及 `图/` 目录下生成的图片和 CSV 文件是否符合预期。

---

## 安全与可移植性注意事项

1. **绝对路径依赖**：脚本中充斥 Windows 绝对路径。若迁移至 Linux 或更换数据盘符，必须批量替换路径前缀。
2. **字体依赖**：部分脚本（如 `schematic.py`、`draw.py`）依赖 Windows 系统自带的微软雅黑、SimHei 等中文字体。在缺少这些字体的环境中运行时，中文标签会显示为方框。
3. **大数据内存占用**：`draw.py`、`access.py`、`tl.py` 会同时加载多个大型 `.npy` 文件（如 `vis1183_ob.npy`、`vis1183_pr.npy`），建议在内存 ≥ 16GB 的环境中运行。
4. **无隔离环境**：项目未使用 `venv`、`conda env` 等虚拟环境配置文件，建议在名为 `meteva` 的 Python 环境中运行（与 `.idea/misc.xml` 中的 SDK 配置一致）。

---

## 项目里程碑

### ✅ 已完成
| 任务 | 描述 | 完成时间 |
|------|------|----------|
| 代码注释 | 为 draw.py 添加中文注释和 Google 风格文档字符串 | 2026-04-02 |
| Git 配置 | 配置 .gitignore，排除 IDE 配置和输出文件 | 2026-04-02 |
| 双平台同步 | 推送到 Gitee 和 GitHub | 2026-04-03 |
| 逻辑错误检查 | 静态检查发现 draw.py 存在变量未定义、类型注解错误等问题 | 2026-04-03 |
| **大文件拆分** | 将 `draw.py` (~936 行) 按 9 个阶段拆分为 `src/` 下的 13 个模块；`draw.py` 仅剩 `main()` 入口（~300 行） | 2026-04-04 |
| **公共函数提取** | 提取 `VisAcc`、`format_time`、`plot_weather_type_eval_bw` 到 `src/`；优化 CDF 计算为 `np.searchsorted` | 2026-04-04 |

### 🔄 进行中
| 任务 | 描述 | 备注 |
|------|------|------|
| 代码重构 | 继续消除 `access.py`、`tl.py`、`ots.py` 中的重复代码 | — |

### 📋 待办事项
| 任务 | 描述 | 优先级 |
|------|------|--------|
| 添加单元测试 | 为 VisAcc 类添加 pytest 测试 | P2 |
| 路径配置化 | 将硬编码路径改为配置文件 | P2 |
