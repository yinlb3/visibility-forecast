# Operational Inference Pipeline 流程文档

> 版本：2026-07-26  
> 适用项目：vis（气象能见度分析可视化工具集）

## 1. 概述

Operational Inference Pipeline 是 `vis` 项目的实时与回算推理流程，用于将数值预报 GRIB 数据转换为中国东部能见度 MICAPS4 产品，并基于训练好的逐站 PDF 模型对预报结果进行后处理修正。

整个流程分为两个阶段：

1. **预处理阶段（`preprocess.py`）**：读取 GRIB 预报、插值、生成 MICAPS4 产品文件，并保存中间数据。
2. **推理阶段（`inference.py`）**：读取中间数据，加载逐站 PDF 模型，按最近站点匹配逐格点修正，输出修正后的能见度产品及等级。

## 2. 总体流程

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  原始 GRIB 文件  │ --> │  preprocess.py   │ --> │  MICAPS4 产品   │
│  (raw_data_dir) │     │  读取 / 插值 / 输出 │     │  (product_dir)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               v
                        ┌──────────────┐
                        │ 中间数据目录  │
                        │ intermediate │
                        │ forecast_    │
                        │ grid.npy     │
                        │ meta.json    │
                        └──────────────┘
                               │
                               v
                        ┌──────────────────┐
                        │  inference.py    │
                        │ 最近站匹配 / 修正 │
                        └──────────────────┘
                               │
                               v
                        ┌─────────────────┐
                        │ 修正后的能见度产品 │
                        │ pred_vis.npy    │
                        │ pred_vis_grade.npy │
                        │ 修正 m4 产品     │
                        └─────────────────┘
```

## 3. 核心模块职责

### 3.1 `preprocess.py`（预处理入口）

- 加载 `config/operational.yaml` 配置。
- 通过命令行参数确定需要处理的 `init_time`（起报时间）列表。
- 对每个 `init_time` 和 `lead_hour`：
  - 定位 GRIB 文件；
  - 读取并插值到目标区域网格；
  - 生成 MICAPS4 文件，写入 `product_dir`；
  - 生成 `forecast_grid.npy`、`meta.json` 等中间数据。
- 可选：加载历史观测数据（`load_observations`），用于研究/训练流程。
- 支持 `fallback` 机制：当 GRIB 缺失时，可生成样本数据继续流程。

### 3.2 `inference.py`（推理入口）

- 加载 `config/operational.yaml` 配置。
- 解析命令行参数，得到 `init_time` 列表。
- 加载区域站点表（`inference.station_csv`），并按 registry 元数据
  `station_ids` 对齐到训练顺序（缺失时回退过滤顺序并提示）。
- 对每个 `init_time`：
  - 读取 `preprocess.py` 生成的中间数据（`forecast_grid.npy`、`meta.json`）；
  - 通过 `model_registry` 加载模型（单个 PDF 或逐站 PDF 列表）；
  - 读取最近站矩阵文件，逐格点用最近站点的模型修正，得到 `pred_vis`；
  - 修正结果封顶 `inference.visibility_cap`，计算等级 `pred_vis_grade`；
  - 保存 `npy`、`meta.json`，并逐时效写出修正 MICAPS4 产品。
- 启动时对首个起报时间预检选模，注册表无可用模型立即报错。

### 3.3 `src/logger.py`（日志模块）

- 基于 `print` 实现，同时写入 UTF-8 日志文件。
- 每个运行实例生成一个 `ops_{run_id}.log` 文件。
- 支持 `DEBUG/INFO/WARNING/ERROR` 四级日志，由配置 `log_level` 控制。
- 所有日志统一保存到 `log_dir`（默认 `./ops/logs`）。

### 3.4 `src/model_registry.py`（模型注册表）

- 从 `model_registry_dir/model_registry.json` 读取模型元数据。
- 支持三种模型选择方式：
  - 显式指定 `model_id`；
  - `latest`：选择创建时间最新的模型；
  - 按 `init_time` 选择：使用训练结束时间早于起报时间的最新模型（适合回算）。
- 当前支持 `PDF` 模型（单个 `pdf_model.PDF` 或逐站 `list[PDF]`），
  可扩展其他模型类型。
- 方案 2（逐站）模型的元数据建议包含 `station_ids`（训练站点顺序），
  inference 据此对齐站点表。

### 3.5 `config/operational.yaml`（业务配置文件）

- `preprocess`：是否加载历史观测数据。
- `paths`：输入输出目录、日志目录、模型注册表目录等。
- `runtime`：起报小时列表、预报时效、回算步长、目标区域。
- `model`：默认模型 ID。
- `inference`：站点表、最近站矩阵文件、修正封顶、修正产品模板。
- `forecast`：GRIB 文件名模板、预报时效列表、缺失文件处理策略。
- `synthetic`：虚假数据设置（`fallback` 缺失 GRIB 时生成样本数据、`fake_timestamp` 伪造文件时间戳）。
- `interpolation`：插值方法、分辨率、目标区域范围等。
- `output`：MICAPS4 文件名/标题模板、日志级别。

敏感或环境相关的路径（如 `<RAW_DATA_DIR>`、`<MODEL_REGISTRY_DIR>`、
`<STATION_CSV>`）应放在 `config.local.windows.yaml` 或
`config.local.linux.yaml` 中覆盖。

### 3.6 `src/near_map.py` 与 `build_near_map.py`（最近站矩阵）

- 矩阵与产品网格同尺寸，每个格点记录最近站点的站号（station id）。
- `build_near_map.py` 手动运行一次生成，网格规格或站点表变化时重跑：
  - `python build_near_map.py`：网格规格取 `interpolation.target_extent`；
  - `python build_near_map.py 2026071700`：网格规格取该时次 `meta.json`。
- 矩阵文件（.npz）内嵌网格规格与站点 id 列表；`inference.py` 只读取
  并校验，缺失或不匹配时报错提示重建，业务运行时不做任何计算。

## 4. 命令行参数模式

`preprocess.py` 与 `inference.py` 均使用 `src/utils.py` 中的
`parse_time_args` 解析时间参数，支持三种运行模式：

| 参数个数 | 示例命令 | 含义 |
|----------|---------|------|
| 0 个 | `python preprocess.py` | 实时模式，默认处理当前时间前 12 小时的整点起报 |
| 1 个 | `python preprocess.py 2026071700` | 单时刻处理 2026-07-17 00 时（UTC）起报 |
| 2 个 | `python preprocess.py 2026070100 2026070700` | 回算模式，按 `backfill_step_hours` 步长生成起报时间序列 |

> 注：时刻参数格式固定为 `YYYYMMDDHH`（10 位数字）。
> `build_near_map.py` 仅支持 0/1 个参数（见 7.1）。

## 5. 输入输出

### 5.1 输入

- 原始 GRIB2 预报文件：`{raw_data_dir}/{filename_template}`
- 模型注册表：`{model_registry_dir}/model_registry.json`
- 最近站矩阵：`{nearest_map_file}`（由 `build_near_map.py` 生成）
- 站点表：`{inference.station_csv}`
- 配置文件：`config/operational.yaml` + 本地覆盖文件

### 5.2 预处理输出（`preprocess.py`）

- MICAPS4 产品：`{product_dir}/{YYYYMMDD}/{init_time}.{lead:03d}`
- 中间目录：`{intermediate_dir}/{init_time}/`
  - `forecast_grid.npy`：预报网格堆栈
  - `meta.json`：元数据（起报时间、区域、网格规格、实际处理时效等）
  - 可选：`sta.csv`、`vis.npy`、`pre.npy`、`rhu.npy`、`vis_grade.npy`、`index_mlyr.npy`

### 5.3 推理输出（`inference.py`）

- 修正后能见度：`{product_dir}/{init_time}/pred_vis.npy`
- 修正后等级：`{product_dir}/{init_time}/pred_vis_grade.npy`
- 元数据：`{product_dir}/{init_time}/meta.json`
- 修正 MICAPS4 产品：`{product_dir}/{YYYYMMDD}/{init_time}_pdfm.{lead:03d}`
  （及 `display_dir` 平铺拷贝）

## 6. 目录结构示例

```
ops/
├── intermediate/                  # 中间数据
│   ├── nearest_sta_idx.npz        # 最近站矩阵（build_near_map.py 生成）
│   ├── 2026070100/
│   │   ├── forecast_grid.npy
│   │   ├── meta.json
│   │   └── ...
│   └── 2026070700/
│       └── ...
├── products/                      # 最终产品
│   ├── 20260701/
│   │   ├── 2026070100.001         # 原始 m4 产品
│   │   ├── 2026070100_pdfm.001    # 修正 m4 产品
│   │   └── ...
│   └── 2026070700/                # 推理输出（按 init_time 分子目录）
│       ├── pred_vis.npy
│       ├── pred_vis_grade.npy
│       └── meta.json
└── logs/                          # 运行日志
    └── ops_20260717003800.log
```

## 7. 运行示例

### 7.1 首次准备：生成最近站矩阵

```bash
python build_near_map.py
```

> 仅需手动运行一次；网格规格或站点表变化时需重跑。

### 7.2 实时单起报预处理

```bash
python preprocess.py 2026071700
```

### 7.3 回算一段时间

```bash
python preprocess.py 2026070100 2026070700
python inference.py 2026070100 2026070700
```

### 7.4 只运行推理（假设预处理和矩阵文件已完成）

```bash
python inference.py 2026071700
```

### 7.5 不指定参数（默认实时模式）

```bash
python preprocess.py
python inference.py
```

> 运行前请确保：
> 1. `config/operational.yaml` 已正确配置；
> 2. `<RAW_DATA_DIR>`、`<MODEL_REGISTRY_DIR>`、`<STATION_CSV>` 等敏感路径已在 `config.local.{windows,linux}.yaml` 中覆盖；
> 3. 磁盘空间充足，特别是 `product_dir` 与 `intermediate_dir` 所在分区。

## 8. 异常处理策略

### 8.1 GRIB 文件缺失

- `forecast.skip_missing = false` 且 `synthetic.fallback.enabled = false`：报错中断。
- `forecast.skip_missing = true`：跳过缺失文件，继续处理其他任务。
- `synthetic.fallback.enabled = true`：对缺失文件生成样本数据，保证流程不中断。

### 8.2 推理失败

- 单起报推理失败：记录错误，继续处理后续起报时间。
- 仅一个起报时间时：抛出异常，便于调试。
- 存在失败起报时间时：进程退出码为 1，便于调度系统感知。

### 8.3 日志记录

- 所有关键步骤（起报时间解析、任务数量、读写文件、失败信息）均写入 `ops/logs/ops_{run_id}.log`。
- 每个运行实例一个日志文件，便于事后回溯。

## 9. 注意事项

1. **时区约定**：GRIB 文件名中的时间按北京时（CST, UTC+8）解析。
2. **产品目录按日期分组**：`product_dir` 下的 MICAPS4 文件按 `YYYYMMDD` 子目录存放，便于业务系统按日期检索。
3. **假时间戳**：`synthetic.fake_timestamp.enabled` 用于生成模拟的业务到达时间戳，仅用于测试或历史数据回算；实时运行建议关闭。
4. **模型版本选择**：回算时应使用训练结束时间早于该起报时间的模型，避免信息泄露；实时运行可使用 `latest`。
5. **内存需求**：GRIB 插值和模型推理涉及大数组，建议在内存充足的服务器上运行，并关注 `parallel.n_jobs` 配置。
6. **最近站矩阵**：网格规格或站点表变化后必须重跑 `build_near_map.py`，否则 inference 校验不通过、报错退出。

## 10. 相关文件

- `preprocess.py`：预处理入口
- `inference.py`：推理入口
- `build_near_map.py`：最近站矩阵预生成入口
- `src/logger.py`：日志模块
- `src/model_registry.py`：模型注册表
- `src/grib_forecast.py`：GRIB 读取与插值实现
- `src/near_map.py`：最近站矩阵构建 / 加载 / 校验
- `config/operational.yaml`：业务配置
- `config/config.local.windows.yaml`：Windows 本地路径覆盖
- `config/config.local.linux.yaml`：Linux 本地路径覆盖

---

**Last Updated**: 2026-07-26
