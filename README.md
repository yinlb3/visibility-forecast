# vis

[![Gitee](https://img.shields.io/badge/Gitee-vis-blue)](https://gitee.com/yinlb97/vis)
[![GitHub](https://img.shields.io/badge/GitHub-vis-black)](https://github.com/yinlb3/vis)

> A collection of Python scripts for meteorological visibility data analysis and visualization.

## Features

- **Data Processing**: Read and process meteorological station visibility, precipitation, relative humidity and other observation data
- **Grade-based Statistics**: Statistics by six visibility grades (<50m, <200m, <500m, <1000m, <2000m, <10000m)
- **Forecast Verification**: Calculate various verification indicators such as ME, MAE, RMSE, MRE, R, TS, ETS, HSS, TSS
- **Visualization**: Draw bar charts, box plots, violin plots, pie charts, heatmaps, spatial distribution maps, etc.

## Install

No installation required. Ensure the following dependencies are installed:

- **Python**: 3.8+
- **Core Dependencies**: numpy, pandas, matplotlib, seaborn, scipy, arrow
- **Domain Library**: meteva (domestic meteorological verification toolkit)
- **Memory**: Recommended ≥ 16GB (for processing large numpy arrays)

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
| Code Comments | Add Chinese comments and Google-style docstrings to draw.py | 2026-04-02 |
| Git Config | Configure .gitignore, exclude IDE config and output files | 2026-04-02 |
| Dual-Platform Sync | Push to Gitee and GitHub | 2026-04-03 |
| Logic Error Check | Static analysis found undefined variables, type annotation errors in draw.py | 2026-04-03 |
| Large File Refactoring | Split `draw.py` (~936 lines) into 13 modules under `src/` by 9 stages; `draw.py` only has `main()` entry (~300 lines) | 2026-04-04 |
| Common Function Extraction | Extract `VisAcc`, `format_time`, `plot_weather_type_eval_bw` to `src/`; Optimize CDF calculation to `np.searchsorted`, complexity from O(N·M) to O(M log M) | 2026-04-04 |
| Translation | Translate README.md and AGENTS.md to English; Fix Chinese path `图/` to `figures/` | 2026-04-05 |

### In Progress
| Task | Description | Notes |
|------|-------------|-------|
| Code Refactoring | Continue eliminating duplicate code in `access.py`, `tl.py`, `ots.py` | — |

### Todo
| Task | Description | Priority |
|------|-------------|----------|
| Add Unit Tests | Add pytest tests for VisAcc class | P2 |
| Path Configuration | Change hard-coded paths to configuration file | P2 |

## License

This project is developed for scientific research purposes and is for internal use only.

## Author

- **Author**: yinlb <yinlb3@foxmail.com>

---

For Chinese version, see [README_cn.md](README_cn.md).
