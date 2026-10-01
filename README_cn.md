# visibility-forecast

[![Gitee](https://img.shields.io/badge/Gitee-visibility--forecast-blue)](https://gitee.com/yinlb97/visibility-forecast)
[![GitHub](https://img.shields.io/badge/GitHub-visibility--forecast-black)](https://github.com/yinlb3/visibility-forecast)

> 面向中国东部气象站的能见度事件分析与快速更新预报订正，实现PDFM-TLE订正、分级检验（TS/FAR/MAR/POD）、时空分布分析与GRIB到业务产品的全流程。

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
python tle_experiment.py

# 业务化流程（先一次性生成最近站映射并注册模型，之后按起报时次运行包装脚本）
python build_near_map.py
python register_model.py model/pdfm2.dat 20240101 pdfm2
bash run_pipeline.sh 2026071700        # Linux
run_pipeline.bat 2026071700            # Windows

# 黄花机场观测数据（机场自有月总簿格式）
python huanghua_airport.py
```

运行前请确认：
1. `D:\data\vis\` 目录包含所需的 `.npy`、`.csv`、`.xls` 等数据文件；
2. 系统已安装中文字体（脚本会查找 `msyh.ttc`、`simhei.ttf` 等）；
3. 内存充足（部分脚本会加载大型 numpy 数组并进行循环计算）；
4. 业务化流程需配置 `config/operational.yaml` 及平台特定的 `config.local.{windows,linux}.yaml`。

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
| 流程合并与阶段拆分 | 将 `preprocess.py`、`inference.py` 合并为 `pipeline.py`；业务化流程按数据（`forecast_prep`）、模型（`postprocess`）、产品（`product_writer`）三阶段拆分为独立模块；`PDF` 更名为 `PDFM` 并保持旧模型文件可加载；下线 `ots.py`、`vis2411.py`、`vis_grade.py`；`tl.py` 更名为 `tle_experiment.py`、`huanghua.py` 更名为 `huanghua_airport.py`；重写 `docs/OPERATIONAL_INFERENCE_PIPELINE_cn.md` 与 `docs/ABBREVIATIONS.md` | 2026-09-30 |
| 公共绘图工具与命名清理 | 将 `setup_plot_style`、`save_station_scatter` 提取至 `src/utils.py`；恢复全部阶段开关；2.1 阶段键名统一为 `stage_2_1_dist_feature`；将拼写错误的 `gainovement` 更正为 `improvement`；图形输出由 SVG 改为 EPS 以适应期刊要求 | 2026-10-01 |

### 进行中
| 任务 | 描述 | 备注 |
|------|------|------|
| 业务化推理流程 | 实现统一入口 `pipeline.py`，支持实时与回算，由 0/1/2 个 YYYYMMDDHH 命令行参数驱动；新增 `src/logger.py`、`src/model_registry.py` 与 `config/operational.yaml`；移除旧的 `preprocess.py`、`inference.py`；GRIB → PDFM 订正 → 保存 PDFM 缓存 → 加载前 24 小时缓存（缺失时由 GRIB 重新生成）→ TLE 平均 → m4 产品输出 | 实现完成，服务器测试进行中 |

### 待办
| 任务 | 描述 | 优先级 |
|------|------|--------|
| 特种行业风险产品研发 | 研发面向交通、航空等特种行业的能见度风险产品 | P1 |
| 回归测试体系 | 用合成数据为 `PDFM`、`TLE` 与 PDFM 缓存补 pytest，关键指标变动超过 0.1% 即拦截，并在 CI 中执行最小编译与静态检查 | P1 |
| 静默失败兜底 | 输出 TLE 每个时效实际参与的样本数、站点顺序错配提升为 ERROR 级、产品先写临时文件再 rename | P1 |
| 配置单一来源 | 合并 `config.yaml` 与 `operational.yaml` 中目前不一致的区域与等级阈值设置 | P2 |
| 预处理并行化 | 研究内存感知的 GRIB 插值任务动态并行方案（joblib / 批量调度），加速服务器上多起报时次预处理 | P2 |
| 多模式数据应用 | 将多模式预报数据应用于能见度分析与产品服务 | P2 |
| 运行资源自适应 | `n_jobs` 按可用内存封顶，`ops/` 日志增加保留期轮转 | P3 |
| 文档一致性 | 后续变更后使 `README.md`、`README_cn.md` 与 `docs/OPERATIONAL_INFERENCE_PIPELINE_cn.md` 保持一致 | P3 |

## 作者

- **作者**: yinlb <yinlb3@foxmail.com>

---

**最后更新**：2026-10-01

英文版见 [README.md](README.md)。
