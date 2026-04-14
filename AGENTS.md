---
name: vis
description: Meteorological visibility data analysis and visualization toolkit
---

# vis Agent Guide

> **Reference**: See `user-preferences` skill for:
> - Coding standards (A1-A9)
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
- `src/` - Extracted common modules (9 modules split from `draw.py`, 1 stage = 1 file)
- `figures/` - Output directory for images, CSV, NPY files
- `draw.py` - Main orchestrator (pipeline entry only)
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
| draw.py Refactoring | Split `draw.py` into 9 `src/` modules by stage (1 stage = 1 file); extract shared utilities; standardize console output and file structure | 2026-04-07 |

### In Progress
| Task | Description | Notes |
|------|-------------|-------|
| Code Refactoring | Continue eliminating duplicate code in `access.py`, `huanghua.py`, `ots.py`, `tl.py`, `vis_grade.py`, `vis2411.py` | — |

### Completed
| Task | Description | Completion Date |
|------|-------------|-----------------|
| A10 Code Style Check | Standardize naming: indices (`idx_east_china`, `idx_mlyr`, `case_idx`), masks (`mask_lv4plus_*`), confusion matrix (`get_conf_mat/_norm`), metrics (`get_*_grade` for by-grade, `get_*_ge` for >=grade), case study vars (`ts_ge4_cma_all` format), output labels (TS_GE, FAR_GE, etc.), add `docs/ABBREVIATIONS.md` | 2026-04-07 |

### Todo
| Task | Description | Priority |
|------|-------------|----------|
| Add Unit Tests | Add pytest tests for VisAcc class | P2 |
| Path Configuration | Change hard-coded paths to configuration file | P2 |

---

**Last Updated**: 2026-04-07
