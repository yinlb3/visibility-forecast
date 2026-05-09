# vis

[![Gitee](https://img.shields.io/badge/Gitee-vis-blue)](https://gitee.com/yinlb97/vis)
[![GitHub](https://img.shields.io/badge/GitHub-vis-black)](https://github.com/yinlb3/vis)

> A collection of Python scripts for meteorological visibility data analysis and visualization.

## Features

- **Data Processing**: Read and process meteorological station visibility, precipitation, relative humidity and other observation data
- **Grade-based Statistics**: Statistics by six visibility grades (<50m, <200m, <500m, <1000m, <2000m, <10000m)
- **Forecast Verification**: Calculate various verification indicators such as ME, MAE, RMSE, MRE, R, TS, ETS, HSS, TSS
- **Visualization**: Draw bar charts, box plots, violin plots, pie charts, heatmaps, spatial distribution maps, etc.

## Usage

```powershell
# Run main plotting script
python draw.py

# Run other analysis scripts
python access.py
python tl.py
```

Before running, please confirm:
1. The `D:\data\vis\` directory contains the required `.npy`, `.csv`, `.xls`, and other data files;
2. Chinese fonts are installed on the system (scripts look for `msyh.ttc`, `simhei.ttf`, etc. in Windows default paths);
3. Sufficient memory (some scripts load large numpy arrays and perform loop calculations).

## Data Sources

- Input data: `D:\data\vis\` (needs to be prepared before running)
- Output results: `D:\Project\vis\figures\`

## Milestones

### Completed
| Task | Description | Completion Date |
|------|-------------|-----------------|
| Project Setup & Documentation | Initialize git repository, configure .gitignore, set up dual-platform sync (Gitee/GitHub), add code comments/docstrings, and translate all documentation to English | 2026-04-02 ~ 2026-04-05 |
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

## Author

- **Author**: yinlb <yinlb3@foxmail.com>

---

For Chinese version, see [README_cn.md](README_cn.md).
