# Abbreviations

## Region
| Abbreviation | Full Name | Description |
|:------------:|-----------|-------------|
| MLYR | Middle-Lower Yangtze River | Middle-Lower Yangtze River region |
| - | East China | East China (use full name, no abbreviation) |

## Time Axes
| Abbreviation | Full Name | Description |
|:------------:|-----------|-------------|
| vt | Valid Time / Lead Time | Forecast lead time in hours (1-24). The horizontal axis of lead-time charts. |
| ft | Forecast Time | Forecast initialization time of day in UTC (0-23). The horizontal axis of forecast-time charts. Distinct from vt. |
| shour | Start Hour | Initialization hour (UTC) |
| lve | Low Visibility Event | Low visibility event (sum of fog+precip+haze frequencies) |

## Data & Variables
| Abbreviation | Full Name | Description |
|:------------:|-----------|-------------|
| ob | Observation | Observed visibility array |
| pr | Prediction / Forecast | Forecast visibility array |
| fcst | Forecast | Forecast array, used when ob/pr naming would be ambiguous |
| sta | Station | Weather station |
| rhu | Relative Humidity | Relative humidity (%) |
| pre | Precipitation | Precipitation indicator |
| vis | Visibility | Visibility in meters |
| wt | Weather Type | Weather type: 1=precip, 2=fog, 3=haze |
| na, nb, nc | - | Standard 2x2 contingency table cells: na=hit, nb=false alarm, nc=miss |

## Metrics
| Abbreviation | Full Name | Description |
|:------------:|-----------|-------------|
| ts | Threat Score | TS = na / (na + nb + nc) |
| ts_ge | Threat Score, grade Greater-or-Equal | TS computed for grade i and above; `ts_ge4` means grade 4+ |
| ccm / corr | Pearson Correlation Coefficient | Linear correlation between obs and forecast |
| mae | Mean Absolute Error | - |
| rmse | Root Mean Square Error | - |
| mre | Mean Relative Error | - |
| ets | Equitable Threat Score | - |
| hss | Heidke Skill Score | - |
| tss | True Skill Statistic | Also known as Pierce's Skill Score |
| far | False Alarm Ratio | - |
| mar | Miss Alarm Ratio | - |
| pod | Probability of Detection | - |
| bias | Frequency Bias | - |
| gain | Gain | Relative improvement of a corrected forecast over the baseline, in percent. Not to be confused with ts_ge. |

## Models & Experiments
| Abbreviation | Full Name | Description |
|:------------:|-----------|-------------|
| CMA-SH-WARR | China Meteorological Administration-Shanghai WRF ADAS rapid refresh system | CMA Shanghai WRF-ADAS rapid refresh system |
| PDFM | probability density function matching | Maps the raw forecast distribution onto the observed one through a quantile mapping curve. Implemented by the `PDFM` class in `src/postprocess.py`. |
| TLE | time-lagged ensemble | Averages PDFM-corrected grids from earlier initialization times along the anti-diagonal of equal valid time. Implemented by `apply_equal_tle` in `src/postprocess.py`. |

## Region Index
| Abbreviation | Full Name | Description |
|:------------:|-----------|-------------|
| idx_mlyr | index for Middle-Lower Yangtze River | Middle-Lower Yangtze River region index |

## Notes
- Add inline comment on first occurrence of abbreviation in code (format: `# xxx: full name and specific description`)
- MLYR follows international geographic naming conventions
- East China uses full name to avoid confusion with ECMWF (ec)
- `PDFM` and `TLE` are distinct stages: PDFM corrects one forecast distribution, TLE then averages several corrected fields. The class formerly named `PDF` was renamed to `PDFM`; existing model files still load through `load_pdfm_file`.
- `tle_experiment.py` (station space) and `postprocess.apply_equal_tle` (grid space) are two different TLE implementations; do not use the names interchangeably.