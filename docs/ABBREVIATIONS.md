# Abbreviations (项目缩写规范)

## Region (区域)
| 缩写 | 全称 | 说明 |
|:----:|------|------|
| MLYR | Middle-Lower Yangtze River | 长江中下游区域 |
| - | East China | 中国东部（使用全称，无缩写）|

## Data & Metrics (数据与指标)
| 缩写 | 全称 | 说明 |
|:----:|------|------|
| val | Validation | 验证集/验证时段 |
| qem | Quantitative Evaluation Metrics | 定量指标：R, MAE, RMSE, MRE |
| cem | Categorical/Grade Evaluation Metrics | 等级指标：TS, FAR, MAR, POD |
| vt | Valid Time / Lead Time | 预报时效（小时）|
| shour | Start Hour | 起报时次（UTC）|
| fhour | Forecast Hour | 预报时间（UTC）|
| wt | Weather Type | 天气类型：1=precip, 2=fog, 3=haze |

## Notes
- First occurrence in code should have inline comment
- MLYR follows international geographical naming convention
- East China uses full name to avoid confusion with ECMWF (ec)
