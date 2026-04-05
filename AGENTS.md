---
name: vis
description: Meteorological visibility data analysis and visualization toolkit
---

# vis Agent Guide

> **Reference**: See `user-preferences` skill for:
> - Coding standards (A1-A9)
> - Git workflow (B4)
> - This document only records project-specific information.

## Context

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
- `src/` - Extracted common modules (13 modules split from draw.py)
- `figures/` - Output directory for images, CSV, NPY files
- `draw.py` - Main plotting script (main() entry only)
- `access.py` - Forecast verification and PDF matching correction
- `tl.py` - Temporal lead experiment analysis
- `ots.py` - Optimal threshold selection
- `D:\data\vis\` - Input data directory (not under version control)

## 1. Commands

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
3. Memory ≥ 16GB recommended (large numpy arrays)

## 2. Testing

No unit test framework (no `pytest`, `unittest`).

Verification method: Run scripts and check:
- Console output verification indicators
- Images and CSV files generated in `figures/` directory meet expectations

## 3. Code Style

Refer to `user-preferences` skill A.1-A.9.

**Project-specific conventions**:
1. **Pinyin Abbreviation Naming**: Variable naming mixes pinyin with English, e.g., `hxjz` (confusion matrix), `cjzxy` (middle-lower Yangtze)
2. **Memory Management**: Use `plt.close(fig)`, `del fig, ax`, `gc.collect()` to release matplotlib memory
3. **Hard-coded Paths**: Input `D:\data\vis\...`, Output `D:\Project\vis\figures\...`
4. **Data Cleaning**: Missing values as `999990`/`999999` → `np.nan`; visibility capped at 30000m

## 4. Git Workflow

Refer to `user-preferences` skill B.4.

**Additional notes**:
- Dual-platform sync: Push to both Gitee and GitHub
- Do not commit large data files (`.npy`, `.csv`) to git

## 5. Boundaries

- **Always do**:
  - Use English for all code comments and output messages
  - Clean up memory after plotting (`plt.close()`, `gc.collect()`)
  - Keep variable names ≤ 20 characters
  - Mark file modifications with line numbers when showing changes

- **Ask first**:
  - Before modifying the 9-stage pipeline structure in `draw.py`
  - Before changing visibility grade thresholds (`THRES` constant)
  - Before adding new dependencies

- **Never do**:
  - Execute `git commit/push` without user explicit instruction
  - Modify files outside working directory
  - Use PowerShell `Set-Content` for Chinese text (use UTF-8 aware methods)

## 6. Milestones

### Completed
| Task | Description | Completion Date |
|------|-------------|-----------------|
| Code Comments | Add Chinese comments and Google-style docstrings to draw.py | 2026-04-02 |
| Git Config | Configure .gitignore, exclude IDE config and output files | 2026-04-02 |
| Dual-Platform Sync | Push to Gitee and GitHub | 2026-04-03 |
| Logic Error Check | Static analysis found undefined variables, type annotation errors in draw.py | 2026-04-03 |
| Large File Refactoring | Split `draw.py` (~936 lines) into 13 modules under `src/` by 9 stages | 2026-04-04 |
| Common Function Extraction | Extract `VisAcc`, `format_time` to `src/`; Optimize CDF calculation | 2026-04-04 |
| Translation | Translate README.md and AGENTS.md to English | 2026-04-05 |

### In Progress
| Task | Description | Notes |
|------|-------------|-------|
| Code Refactoring | Continue eliminating duplicate code in `access.py`, `tl.py`, `ots.py` | — |

### Todo
| Task | Description | Priority |
|------|-------------|----------|
| Add Unit Tests | Add pytest tests for VisAcc class | P2 |
| Path Configuration | Change hard-coded paths to configuration file | P2 |

---

**Last Updated**: 2026-04-05
