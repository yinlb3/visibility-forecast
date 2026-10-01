---
name: vis
description: Meteorological visibility data analysis and visualization toolkit
---

# vis Agent Guide

> **Reference**: This guide records project-specific information only.
> For general coding standards and Git workflow, follow the conventions described in this document.

## 1. Context

### Your role
You are an AI programming assistant for this meteorological data analysis project. Your responsibilities include:
- Writing and refactoring Python code for visibility data processing and visualization
- Following the established code style (English comments, pinyin abbreviations for local variables)
- Ensuring memory efficiency when handling large numpy arrays
- Maintaining consistency with the 10-stage analysis pipeline defined in `draw.py`

### Project knowledge

**Background**: This project is a collection of Python scripts for meteorological visibility data analysis and visualization, mainly used to process visibility observation and forecast data from meteorological stations in China, perform grade-based verification and statistical evaluation of visibility forecasts, and draw various statistical charts required for papers.

**Tech Stack**:
- **Language**: Python 3.8 (SDK name in IDE is `meteva`)
- **Core Dependencies**: numpy, pandas, matplotlib, seaborn, scipy, arrow, joblib
- **Domain Library**: meteva (domestic meteorological verification toolkit)

**File Structure**:
- `src/` - Pipeline step modules (no intermediate wrappers)
- `src/p1_config_data.py` - Config + obs/forecast loading helpers
- `src/p2_1_dist_feature.py` - Distribution stats calc (month/hour/station)
- `src/p2_2_event_type.py` - Event type plots (pies, violin/box)
- `src/p2_3_temporal.py` - Temporal plots (monthly/hourly bars + violins)
- `src/p2_4_spatial.py` - Spatial plots (station frequency/mean maps)
- `src/p3_1_eval_calc.py` - Forecast eval calc (overall, station, weather-type, temporal)
- `src/p3_2_init_lead.py` - Init/lead time plots (3.2: heatmaps, bars, maps, violins)
- `src/p3_3_wt_ts4.py` - Weather type evaluation output (3.3)
- `src/p3_4_ablation.py` - Ablation plots (3.4: weather-type BW bars, CDF)
- `src/p3_5_case_study.py` - 2024 case study analysis (3.5)
- `src/utils.py`, `src/vis_acc.py` - Shared utilities and verification class
- `figures/` - Output directory for images, CSV, NPY files
- `draw.py` - Main orchestrator (flat pipeline, calls step modules directly)
- `access.py` - Forecast verification and PDFM correction schemes
- `tle_experiment.py` - Lead-by-lead TLE experiment over station observations
- `huanghua_airport.py` - Huanghua airport monthly-logbook data extraction
- `pipeline.py` - Unified operational pipeline (read, correct, write in one process)
- `build_near_map.py` - One-time nearest-station map builder
- `src/grib_forecast.py` - GRIB reading and station-to-grid interpolation
- `src/forecast_prep.py` - Data stage: GRIB reading, interpolation, intermediate storage
- `src/postprocess.py` - Model stage: PDFM class, TLE averaging, PDFM cache
- `src/product_writer.py` - Product stage: MICAPS4 output and display copying
- `src/near_map.py` - Nearest-station map build/load utilities
- `src/model_registry.py` - Model registry for operational inference
- `src/logger.py` - Print-based operational logger
- `D:\data\vis\` - Input data directory (not under version control)

The operational pipeline is split by stage, each stage in one module:
`forecast_prep` (data) -> `postprocess` (model) -> `product_writer`
(product). `pipeline.py` only orchestrates the flow and parses arguments.

## 2. Commands

```powershell
# One-click execution
python draw.py

# Operational pipeline (GRIB -> PDFM/TLE -> corrected m4)
bash run_pipeline.sh              # Linux: real-time or backfill
run_pipeline.bat                  # Windows: real-time or backfill

# Step-by-step (alternative scripts)
python pipeline.py [--utc|--bjt] [YYYYMMDDHH] [YYYYMMDDHH]  # GRIB -> corrected m4
python build_near_map.py                           # Build nearest-station map once
python register_model.py <dat_path> <training_end> [model_id] [registry_dir]
python access.py            # PDFM correction schemes and verification
python tle_experiment.py     # Lead-by-lead TLE experiment
python huanghua_airport.py   # Huanghua airport logbook extraction
```

Before running, confirm:
1. `D:\data\vis\` contains required `.npy`, `.csv`, `.xls` files
2. Chinese fonts installed (scripts look for `msyh.ttc`, `simhei.ttf`)
3. Memory >= 16GB recommended (large numpy arrays)
4. For operational pipeline: configure `config/operational.yaml` and platform-specific `config.local.{windows,linux}.yaml`

## 3. Testing

This project has no unit test framework (no `pytest`, `unittest`). Verification method: run scripts and check:
- Console output verification indicators
- Images and CSV files generated in `figures/` directory meet expectations

## 4. Code Style

See the project-specific conventions below and the Boundaries section in this document.

**Project-specific conventions**:
1. **Pinyin Abbreviation Naming**: Variable naming mixes pinyin with English, e.g., `mlyr` (Middle-Lower Yangtze River)
2. **Memory Management**: Use `plt.close(fig)`, `del fig, ax`, `gc.collect()` to release matplotlib memory
3. **Hard-coded Paths**: Input `D:\data\vis\...`, Output `D:\Project\vis\figures\...`
4. **Data Cleaning**: Missing values as `999990`/`999999` -> `np.nan`; visibility capped at 30000m

## 5. Git Workflow

See the Git guidelines below and the additional notes in this document.

**Additional notes**:
- Dual-platform sync: Push to both Gitee and GitHub
- Do not commit large data files (`.npy`, `.csv`) to git. `.gitignore`
  already excludes documents, raw data, archives and figure outputs
  (`*.docx`, `*.npy`, `*.csv`, `figures*/`, ...). Use `git add -f` only
  when a small sample must be tracked on purpose.

## 6. Boundaries

- **Always do**:
  - Use English for all code comments and output messages
  - Clean up memory after plotting (`plt.close()`, `gc.collect()`)
  - Keep variable names <= 20 characters
  - Mark file modifications with line numbers when showing changes

- **Ask first**:
  - Before modifying the 10-stage pipeline structure in `draw.py`
  - Before changing visibility grade thresholds (`THRES` constant)
  - Before adding new dependencies

- **Never do**:
  - Execute `git commit/push` without user explicit instruction
  - Modify files outside working directory
  - Use PowerShell `Set-Content` for Chinese text (use UTF-8 aware methods)

## 7. Milestones

### Completed
| Task | Description | Completion Date |
|------|-------------|-----------------|
| Project Setup & Documentation | Initialize git repository, configure .gitignore, set up dual-platform sync (Gitee/GitHub), add code comments/docstrings, and translate all documentation to English | 2026-04-05 |
| draw.py Refactoring | Flatten `draw.py` into 3 Parts with direct `src/pX_Y` module calls; remove old intermediate wrappers; extract shared utilities; standardize console output, file structure, and naming conventions (indices, masks, confusion matrix, metrics, case study vars, output labels); add `docs/ABBREVIATIONS.md` | 2026-04-15 |
| Evaluation Calculation Parallelization | Parallelize station-level and init-hour block metrics calculation in `src/p3_1_eval_calc.py` using `joblib.Parallel(n_jobs=-1)`, reducing runtime to approximately 1/3 of original | 2026-04-15 |
| Code Style Compliance & Path Configuration | Systematically check and fix code style items, and migrate hard-coded paths to the configuration file | 2026-05-09 |
| Code Refactoring | Refactor `access.py`, `tl.py`, `vis2411.py`, `ots.py`, `vis_grade.py`, `huanghua.py`; move reusable classes (`PDF`, `OTS`) into `src/pdf_model.py` and `src/ots_model.py`; replace hard-coded paths with `config.yaml`; standardize entry points and English comments | 2026-07-16 |
| Pipeline Consolidation and Stage Split | Merge `preprocess.py` and `inference.py` into `pipeline.py`; split the operational flow into three stage modules (`forecast_prep` data, `postprocess` model, `product_writer` product); rename `PDF` to `PDFM` with backward-compatible model loading; retire `ots.py`, `vis2411.py`, `vis_grade.py`; rename `tl.py` to `tle_experiment.py` and `huanghua.py` to `huanghua_airport.py` | 2026-09-30 |
| Shared Plot Helpers & Naming Cleanup | Extract `setup_plot_style` and `save_station_scatter` into `src/utils.py`; restore every stage switch; rename the 2.1 stage key to `stage_2_1_dist_feature`; correct the misspelled `gainovement` identifier to `improvement`; emit EPS instead of SVG for journal figures | 2026-10-01 |

### In Progress
| Task | Description | Notes |
|------|-------------|-------|
| Operational Inference Pipeline | Implement unified `pipeline.py` for real-time and backfill operation, driven by 0/1/2 YYYYMMDDHH command-line arguments; add `src/logger.py`, `src/model_registry.py`, `config/operational.yaml`, and remove legacy `preprocess.py`/`inference.py`; GRIB -> PDFM correction -> save PDFM cache -> load previous 24 h caches (regenerate from GRIB if missing) -> TLE averaging -> m4 product output | implementation complete, server testing in progress |

### Todo
| Task | Description | Priority |
|------|-------------|----------|
| Special Industry Risk Product Development | Develop visibility risk products for special industries such as transportation and aviation | P1 |
| Regression Test Suite | Add pytest coverage for `PDFM`, `TLE` and the PDFM cache using synthetic data, gate key metric changes within 0.1%, and run a minimal compile plus lint check in CI | P1 |
| Silent Failure Guards | Report the actual TLE sample count per lead, raise station-order mismatches to ERROR level, and write products to a temporary file before renaming | P1 |
| Single Source of Configuration | Consolidate the region and grade-threshold settings that currently differ between `config.yaml` and `operational.yaml` | P2 |
| Preprocessing Parallelization | Research memory-aware dynamic parallelization for GRIB interpolation tasks (joblib / batch scheduling) to speed up multi-init-time preprocessing on servers | P2 |
| Multi-model Data Application | Apply multi-model forecast data to visibility analysis and product services | P2 |
| Resource Self-adaptation | Cap `parallel.n_jobs` by available memory and add retention rotation for `ops/` logs | P3 |
| Documentation Consistency | Keep `README.md` and `README_cn.md` in step with the code after further changes | P3 |

---

**Last Updated**: 2026-10-01
