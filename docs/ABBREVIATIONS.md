# Abbreviations

## Region
| Abbreviation | Full Name | Description |
|:------------:|-----------|-------------|
| MLYR | Middle-Lower Yangtze River | Middle-Lower Yangtze River region |
| - | East China | East China (use full name, no abbreviation) |

## Data & Metrics
| Abbreviation | Full Name | Description |
|:------------:|-----------|-------------|
| val | Validation | Validation set / validation period |
| qem | Quantitative Evaluation Metrics | Quantitative metrics: R, MAE, RMSE, MRE |
| cem | Categorical/Grade Evaluation Metrics | Categorical metrics: TS, FAR, MAR, POD |
| vt | Valid Time / Lead Time | Forecast lead time (hours) |
| shour | Start Hour | Initialization hour (UTC) |
| fhour | Forecast Hour | Forecast hour (UTC) |
| wt | Weather Type | Weather type: 1=precip, 2=fog, 3=haze |
| LVE | Low Visibility Event | Low visibility event (sum of fog+precip+haze frequencies) |
| LVPE | Low Visibility Precipitation Event | Precipitation-type low visibility event |
| LVFE | Low Visibility Fog Event | Fog-type low visibility event |
| LVHE | Low Visibility Haze Event | Haze-type low visibility event |

## Models & Experiments
| Abbreviation | Full Name | Description |
|:------------:|-----------|-------------|
| CMA-SH-WARR | China Meteorological Administration-Shanghai WRF ADAS rapid refresh system | China Meteorological Administration Shanghai WRF ADAS rapid refresh system |
| PDFM | probability density function matching | Probability density function matching |
| TLE | time-lagged ensemble | Time-lagged ensemble |

## Region Index
| Abbreviation | Full Name | Description |
|:------------:|-----------|-------------|
| idx_mlyr | index for Middle-Lower Yangtze River | Middle-Lower Yangtze River region index |

## Notes
- Add inline comment on first occurrence of abbreviation in code (format: `# xxx: full name and specific description`)
- MLYR follows international geographic naming conventions
- East China uses full name to avoid confusion with ECMWF (ec)
