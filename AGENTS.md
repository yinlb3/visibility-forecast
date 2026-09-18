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
- Maintaining consistency with the 9-stage analysis pipeline defined in `draw.py`

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
- `access.py` - Forecast verification and PDF matching correction
- `tl.py` - Temporal lead experiment analysis
- `ots.py` - Optimal threshold selection
- `pipeline.py` - Unified operational pipeline (preprocess + inference in one process)
- `build_near_map.py` - One-time nearest-station map builder
- `src/grib_forecast.py` - GRIB reading and station-to-grid interpolation
- `src/near_map.py` - Nearest-station map build/load utilities
- `src/model_registry.py` - Model registry for operational inference
- `src/tle_model.py` - Time-lagged ensemble averaging for operational inference
- `src/logger.py` - Print-based operational logger
- `D:\data\vis\` - Input data directory (not under version control)

## 2. Commands

```powershell
# One-click execution
python draw.py

# Operational pipeline (unified: preprocess + inference)
bash run_pipeline.sh              # Linux: real-time or backfill
run_pipeline.bat                  # Windows: real-time or backfill

# Step-by-step (alternative scripts)
python pipeline.py [--utc|--bjt] [YYYYMMDDHH] [YYYYMMDDHH]     # Unified GRIB -> corrected m4
python build_near_map.py                           # Build nearest-station map once
python register_model.py <dat_path> <training_end> [model_id] [registry_dir]
python access.py    # Forecast correction and verification
python tl.py        # Temporal lead experiment analysis
python ots.py       # Optimal threshold selection
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
- Do not commit large data files (`.npy`, `.csv`) to git

## 6. Boundaries

- **Always do**:
  - Use English for all code comments and output messages
  - Clean up memory after plotting (`plt.close()`, `gc.collect()`)
  - Keep variable names <= 20 characters
  - Mark file modifications with line numbers when showing changes

- **Ask first**:
  - Before modifying the 9-stage pipeline structure in `draw.py`
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

### In Progress
| Task | Description | Notes |
|------|-------------|-------|
| Operational Inference Pipeline | Implement unified `pipeline.py` for real-time and backfill operation, driven by 0/1/2 YYYYMMDDHH command-line arguments; add `src/logger.py`, `src/model_registry.py`, `src/tle_model.py`, `config/operational.yaml`, and remove legacy `preprocess.py`/`inference.py`; GRIB -> PDFM correction -> save PDFM cache -> load previous 24 h caches (regenerate from GRIB if missing) -> TLE averaging -> m4 product output | implementation complete, server testing in progress |

### Todo
| Task | Description | Priority |
|------|-------------|----------|
| Special Industry Risk Product Development | Develop visibility risk products for special industries such as transportation and aviation | P1 |
| Preprocessing Parallelization | Research memory-aware dynamic parallelization for GRIB interpolation tasks (joblib / batch scheduling) to speed up multi-init-time preprocessing on servers | P2 |
| Multi-model Data Application | Apply multi-model forecast data to visibility analysis and product services | P2 |

---

**Last Updated**: 2026-08-17
