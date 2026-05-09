---
name: vis
description: Meteorological visibility data analysis and visualization toolkit
---

# vis Agent Guide

> **Reference**: See `user-preferences` skill for:
> - Coding standards (A1-A11)
> - Git workflow (B4)
> - This document only records project-specific information.

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
- `src/p3_2_init_lead.py` - Init/lead time plots (heatmaps, bars, maps, violins)
- `src/p3_3_ablation.py` - Ablation plots (weather-type BW bars, CDF)
- `src/p3_4_case_study.py` - 2024 case study analysis
- `src/utils.py`, `src/vis_acc.py` - Shared utilities and verification class
- `figures/` - Output directory for images, CSV, NPY files
- `draw.py` - Main orchestrator (flat pipeline, calls step modules directly)
- `access.py` - Forecast verification and PDF matching correction
- `tl.py` - Temporal lead experiment analysis
- `ots.py` - Optimal threshold selection
- `D:\data\vis\` - Input data directory (not under version control)

## 2. Commands

```powershell
# One-click execution
python draw.py

# Step-by-step (alternative scripts)
python access.py    # Forecast correction and verification
python tl.py        # Temporal lead experiment analysis
python ots.py       # Optimal threshold selection
```

Before running, confirm:
1. `D:\data\vis\` contains required `.npy`, `.csv`, `.xls` files
2. Chinese fonts installed (scripts look for `msyh.ttc`, `simhei.ttf`)
3. Memory >= 16GB recommended (large numpy arrays)

## 3. Testing

Refer to `user-preferences` skill Testing guidelines.

No unit test framework (no `pytest`, `unittest`). Verification method: run scripts and check:
- Console output verification indicators
- Images and CSV files generated in `figures/` directory meet expectations

## 4. Code Style

Refer to `user-preferences` skill A.1-A.9.

**Project-specific conventions**:
1. **Pinyin Abbreviation Naming**: Variable naming mixes pinyin with English, e.g., `mlyr` (Middle-Lower Yangtze River)
2. **Memory Management**: Use `plt.close(fig)`, `del fig, ax`, `gc.collect()` to release matplotlib memory
3. **Hard-coded Paths**: Input `D:\data\vis\...`, Output `D:\Project\vis\figures\...`
4. **Data Cleaning**: Missing values as `999990`/`999999` -> `np.nan`; visibility capped at 30000m

## 5. Git Workflow

Refer to `user-preferences` skill B.4.

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
| Code Style Compliance | Systematically check and fix all code style items | 2026-05-09 |

### In Progress
| Task | Description | Notes |
|------|-------------|-------|
| Code Refactoring | Continue eliminating duplicate code in `access.py`, `huanghua.py`, `ots.py`, `tl.py`, `vis_grade.py`, `vis2411.py` | — |

### Todo
| Task | Description | Priority |
|------|-------------|----------|
| Add Unit Tests | Add pytest tests for VisAcc class | P2 |
| Path Configuration | Change hard-coded paths to configuration file | P2 |

---

**Last Updated**: 2026-04-15
