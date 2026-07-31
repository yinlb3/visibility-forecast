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

```powershell
pip install -r requirements.txt
```

## Usage

```powershell
# Run main plotting script
python draw.py

# Run other analysis scripts
python access.py
python tl.py
python ots.py

# Operational pipeline (run once to build the nearest-station map and register
# the model, then use the wrapper for each initialization cycle)
python build_near_map.py
python register_model.py model/pdfm2.dat 20240101 pdfm2
bash run_pipeline.sh 2026071700        # Linux
run_pipeline.bat 2026071700            # Windows
```

Before running, please confirm:
1. The `D:\data\vis\` directory contains the required `.npy`, `.csv`, `.xls`, and other data files;
2. Chinese fonts are installed on the system (scripts look for `msyh.ttc`, `simhei.ttf`, etc. in Windows default paths);
3. Sufficient memory (some scripts load large numpy arrays and perform loop calculations);
4. For the operational pipeline: configure `config/operational.yaml` and the platform-specific `config.local.{windows,linux}.yaml`.

## Data Sources

- Input data: `D:\data\vis\` (needs to be prepared before running)
- Output results: `D:\Project\vis\figures\`

## Milestones

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
| Operational Inference Pipeline | Implement `preprocess.py` and `inference.py` for real-time and backfill operation, driven by 0/1/2 YYYYMMDDHH command-line arguments; add `src/logger.py`, `src/model_registry.py`, and `config/operational.yaml` | implementation complete, server testing in progress |

### Todo
| Task | Description | Priority |
|------|-------------|----------|
| Special Industry Risk Product Development | Develop visibility risk products for special industries such as transportation and aviation | P1 |
| Preprocessing Parallelization | Research memory-aware dynamic parallelization for GRIB interpolation tasks (joblib / batch scheduling) to speed up multi-init-time preprocessing on servers | P2 |
| Multi-model Data Application | Apply multi-model forecast data to visibility analysis and product services | P2 |

## Author

- **Author**: yinlb <yinlb3@foxmail.com>

---

**Last Updated**: 2026-07-31

For Chinese version, see [README_cn.md](README_cn.md).
