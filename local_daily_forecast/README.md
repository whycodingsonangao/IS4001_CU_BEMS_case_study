# Question B: next-day AC zone priorities

## Facility decision

At 00:00 on day D, rank the 24 C1 AC zones by forecast total AC kWh; show the difference from that zone's historical typical value for context. A manager can review schedules and equipment before the operating day. The dashboard replays historical 2019 forecast dates; it is not connected to a live feed. A daily forecast cannot identify the hour of a peak.

## Target and inputs

The target is total AC kWh for one zone on one complete calendar day. All 24 AC hours must pass the 57/60-minute rule; the strict sensitivity run requires 60/60 minutes. Missing days are not zero. Inputs are zone, day of week, weekend and holiday flags, complete AC totals 1 and 7 days earlier, a prior seven-day mean, and prior-day ERA5 weather summaries. Missing lag indicators are retained. No target-day readings, target-day weather or Question A labels enter the forecast. The model uses only 2019 dates from 8 January onward.

## Evaluation

Chronological expanding windows refit monthly. May–August is development; September–December is the final retrospective test. The benchmark is the zone's complete daily total seven days earlier, falling back to training-only zone/day-type and zone means when unavailable. Boosting is selected only if it has lower development MAE than Ridge and |NMBE| is at most 5%; otherwise Ridge is selected. The selected model is Ridge. Primary metrics are zone-day MAE, MAE skill against the benchmark on identical complete days, and average overlap with the actual five highest-use zones each day. RMSE, CV(RMSE), NMBE and building-day MAE are secondary. CV(RMSE) is descriptive, not a pass/fail threshold for zone prioritization.

| Rule | Final scored zone-days | Ridge MAE | Benchmark MAE | MAE skill | Ridge CV(RMSE) | Ridge NMBE |
|---|---:|---:|---:|---:|---:|---:|
| 57/60 | 2,553 | 22.79 kWh | 36.36 kWh | 37.3% | 42.6% | +3.1% |
| 60/60 | 2,229 | 22.53 kWh | 35.39 kWh | 36.3% | 46.1% | +5.2% |

Final-test Ridge MAE beats the benchmark in all 24 AC zones under the main rule. Floor 6 has no data after 22 October; the dashboard also reports a result excluding it. This is a retrospective evaluation on a fixed dataset, not prospective operational validation.

The daily model and the seven-day benchmark have almost identical top-five zone overlap on the final test: 514/600 (85.7%) and 516/600 (86.0%) zone selections, respectively. Top-five overlap is recall@5 for the five actual highest-use zones; with exactly five predicted zones, precision@5 has the same value. The model's clear gain is in estimated kWh, not in identifying a different set of high-use zones. Ranking by forecast above a zone's typical value performs worse than the benchmark, so the dashboard ranks absolute forecast kWh and shows above-typical use only as context.

## Daily review flags

For each zone and day type, the development residual 5th and 95th percentiles define an expected range around the daily forecast. A complete final-test day outside that range is a candidate for human review after actual use is known, not a confirmed fault. The main run flags 231 zone-days. The exception page reports per-zone rates and a downloadable list.

## Reproduce

From the repository root after the project dependencies and data are available:

```sh
python scripts/run_question_b.py
streamlit run app/Home.py
```

On macOS, XGBoost also requires an OpenMP runtime such as `libomp`.
