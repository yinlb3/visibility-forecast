# AGENTS.md

> 本文件供 AI 编程助手阅读，用于快速了解本项目结构、技术栈与开发约定。

---

## 项目概述

本项目是一个**气象能见度数据分析与可视化**的 Python 脚本集合，主要用于：
- 处理中国区域气象站的能见度观测与预报数据；
- 对能见度预报进行分级检验与统计评估；
- 绘制论文所需的各类统计图（柱状图、箱线图、小提琴图、饼图、空间分布图等）。

项目没有采用包（package）结构，所有脚本平铺在根目录下，各自独立运行。脚本中存在大量互相复制的工具函数（如 `VisAcc`、`format_time`）。

---

## 目录结构

```
D:\Project\vis\
├── .idea/                  # PyCharm / IntelliJ IDEA 配置
├── 图/                     # 输出目录：存放生成的图片、CSV、NPY 等
│   ├── 投消图/             # 论文投稿用图片子目录
│   ├── *.png / *.jpg / *.eps / *.pdf   # 各类可视化成果
│   ├── vis_*.csv           # 中间统计结果（如 vis_hour.csv、vis_sta.csv）
│   └── ...
├── access.py               # 预报检验与 PDF 匹配订正
├── draw.py                 # 主绘图脚本（最大的可视化入口）
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
| `draw.py` | 主绘图入口。读取观测与多种预报试验结果，绘制月/小时频率图、箱线图、小提琴图、饼图、空间散点图，并输出到 `图/` 目录。 | `VisAcc`, `plot_weather_type_eval_bw`, `main` |
| `access.py` | 预报订正与检验。实现 PDF 匹配订正（`PDF` 类）以及分级/定量检验指标计算（`VisAcc` 类）。 | `PDF`, `VisAcc` |
| `tl.py` | 时效试验分析。对不同起报时效的预报进行滑动加权平均，并计算检验指标。 | `VisAcc`, `main` |
| `ots.py` | 最优阈值选取。基于训练样本搜索使 TS 评分最大的阈值，再对预报进行分段线性映射。 | `OTS`, `VisAcc` |
| `schematic.py` | 绘制 24×24 对角线颜色网格示意图，用于展示模式循环同步方案。 | `create_diagonal_matrix`, `draw_diagonal_grid` |
| `huanghua.py` | 读取 2013–2023 年历年月总簿 Excel，提取场面气压、修正海平面气压、温度、相对湿度等要素。 | `get_n_days`, `main` |
| `vis2411.py` | 将 1951 年以来的逐月 CSV 能见度数据整理为 `vis.npy`。 | `main` |
| `vis_grade.py` | 对能见度进行六级分级统计，并生成多年站点级统计表格。 | `main` |

### 关于 `VisAcc` 类

多个脚本中均复制了一份 `VisAcc` 类，用于计算能见度分级检验指标：
- **定量指标**：ME、MAE、RMSE、MRE、Pearson 相关系数 R
- **分级指标**：TS、ETS、HSS、TSS、BIAS、FAR、MAR、POD、OA、Kappa
- 能见度分级阈值（米）定义为全局常量：
  ```python
  THRES = (10000., 2000., 1000., 500., 200., 50.)
  ```

### 关于 `format_time`

同样被复制到多个脚本中，用于将秒数格式化为可读字符串，例如 `'43.5 seconds'`、`'12.5m'`。

---

## 代码风格与开发约定

1. **文件头格式**
   ```python
   #!user/bin.python3
   """
   Founded in YYYY-MM-DD
   Modified in YYYY-MM-DD
   @author: yinlb
   """
   ```
   - Shebang 写法为 `#!user/bin.python3`（非标准 Linux shebang）。

2. **注释与文档字符串**
   - 函数 docstring 和注释使用**中文**。
   - 变量命名混用拼音缩写（如 `hxjz` 混淆矩阵、`cjzxy` 长江中下游）与英文。

3. **内存管理习惯**
   - 绘图脚本中频繁使用 `plt.close(fig)`、`del fig, ax`、`gc.collect()` 来释放 matplotlib 占用的大量内存。

4. **硬编码路径**
   - 输入数据路径硬编码为 `D:\data\vis\...`
   - 输出路径硬编码为 `D:\Project\vis\图\...`
   - 修改或迁移项目时，需要全局替换这些路径。

5. **数据清洗约定**
   - 缺失值常以 `999990` 或 `999999` 形式出现，读取后统一替换为 `np.nan`。
   - 能见度上限常截断为 `30000` 米。

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
