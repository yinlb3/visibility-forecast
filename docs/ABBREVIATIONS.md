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
| LVE | Low Visibility Event | 低能见度事件（雾+降水+霾频率之和）|
| LVPE | Low Visibility Precipitation Event | 降水型低能见度事件 |
| LVFE | Low Visibility Fog Event | 雾型低能见度事件 |
| LVHE | Low Visibility Haze Event | 霾型低能见度事件 |

## Models & Experiments (模型与试验)
| 缩写 | 全称 | 说明 |
|:----:|------|------|
| CMA-SH-WARR | China meteorological administration-Shanghai WRF ADAS rapid refresh system | 中国气象局上海快速更新同化预报系统 |
| PDFM | probability density function matching | 概率密度匹配 |
| TLE | time-lagged ensemble | 时间滞后集合 |

## Region Index (区域索引)
| 缩写 | 全称 | 说明 |
|:----:|------|------|
| idx_mlyr | index for Middle-Lower Yangtze River | 长江中下游区域索引 |

## Notes (说明)
- 代码中首次出现缩写时应添加行内注释（格式：`# xxx: 全称 和 具体说明`）
- MLYR 遵循国际地理命名惯例
- East China 使用全称以避免与 ECMWF（ec）混淆
