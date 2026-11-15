# Where to Look First: Energy and Next-Day AC Priorities

**IS4001 Group 7 · CU-BEMS case study.** We show facilities managers where the building's electricity goes
(Question A), which AC zones merit attention tomorrow (Question B), and which completed days
behaved unexpectedly (the retrospective review layer).

| | A: descriptive | B: predictive |
|---|---|---|
| Asks | Which daily electricity patterns recur, and how much kWh does each account for? | Which AC zones should a facility manager review tomorrow? |
| Unit | Zone-day | Complete AC zone-day |
| Method | k-means on 24 hourly kWh per end use (C1: 72, C2: 48 features), 95th-percentile scaling; 24/7 zone F1Z2 reported separately | Same-zone daily total seven days earlier vs Poisson gradient boosting vs Ridge |
| Evaluation | k in 3–8: size rule → half-year ARI ≥ 0.8 → silhouette → interpretability; Ward cross-check | Chronological expanding window, monthly refit: dev May–Aug, final test Sep–Dec |
| KPIs | kWh per cluster, zone, floor and end use | Zone-day MAE, skill vs benchmark and recall@5 for the five highest-use zones (primary); RMSE, CV(RMSE), NMBE (secondary); daily exception rate |

A's cluster labels are never inputs to B.

The complete Question B method, evaluation and facility use case are documented in
[Question B: next-day AC zone priorities](local_daily_forecast/README.md).

## Stack

Python 3.11+ end to end: **pandas/pyarrow** for the pipeline, **scikit-learn** (k-means, Ridge) and **XGBoost** for
modelling, and **Streamlit + Plotly** for the localhost dashboard. Analysis code lives in an importable package
(`src/cubems`), so scripts, the methods notebook and the dashboard share one implementation of every rule.

## Setup (Windows, PowerShell)

```powershell
git clone <repo-url>; cd IS4001_CU_BEMS_case_study
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt     # also installs the local `cubems` package in editable mode
```

macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

## Build and open everything (one command)

Pick either. Both run the same pipeline (fetch data → processed tables → Question A → Question B; ~2 min, plus a
one-time ~150 MB download):

```powershell
streamlit run app/Home.py              # on a fresh clone the dashboard shows a "Build everything now" button
python scripts/run_all.py --serve      # or build in the terminal, then open the dashboard automatically
```

Use the Rebuild panel at the bottom of **Data quality** after pulling code changes. The terminal equivalent is
`python scripts/run_all.py`. `run_all.py` stops at the first failure and is safe to re-run, because valid raw files
are skipped. Options:

- `--serve` opens the dashboard when the build finishes. `--skip-build` opens it without rebuilding.
- `--from "C:\path\to\folder"` copies the raw CSVs from a folder you already have instead of downloading them.
- `--quick` runs the main 57/60 rule only, skipping the 60/60 sensitivity run and the diagnostic (about half the time).
- `--no-weather` works offline, but skips Question B, which needs the weather features.

## Data

The raw data (14 CSVs, ~700 MB) is **not** in this repo, and neither is anything derived from it. `run_all.py`
handles it, or you can run the two data steps yourself (venv active):

```powershell
python scripts/get_data.py        # downloads the 14 CSVs from the authors' site into data/raw/ and checks them
python scripts/build_dataset.py   # ~30 s: builds data/processed/ (+ one Open-Meteo call; --no-weather if offline)
```

`get_data.py` pulls the zips from the dataset's official page (<https://sgrudata.github.io/>), extracts only
the CSVs, and checks each one against a SHA-256 checksum, so everyone runs on byte-identical data. It skips files
that are already valid, so re-running it is safe.

- **Already have the files** (e.g. a Kaggle download)? `python scripts/get_data.py --from "C:\path\to\folder"`
  copies and checks them instead of downloading.
- **Keep the data elsewhere?** Set `CUBEMS_RAW_DIR` to that folder before running either script.
- **A file fails its checksum?** It's deleted and reported. Re-run, or use `--force` to re-download everything.

| Output | Grain | Contents |
|---|---|---|
| `data/processed/zone_hourly.parquet` | zone × hour | kWh per end use (mean kW × 1 h), valid-minute counts, indoor sensor means |
| `data/processed/zone_day.parquet` | zone × day | kWh over valid hours, valid hours per end use, `usable` (57/60) and `usable_strict` (60/60), calendar (official holidays, day type, season), weather `wx_*` |
| `data/processed/zones.parquet` | zone | Group (C1 / C2 / Lighting only), AC unit count, sensors |
| `data/processed/weather_daily.parquet` | day | Mean/max/min temperature, RH, dew point, radiation, cooling degree-hours |
| `data/processed/qc_events.parquet` | flagged episode | Spike / flatline episodes per channel: start, end, minutes, peak value, threshold |
| `data/processed/qc_channels.parquet` | channel × year | Screening thresholds, flagged minutes, longest identical run |

## Run the analysis (individual steps)

```powershell
python scripts/run_question_a.py   # ~30 s: k-means for C1 and C2, main + strict 60/60 run -> data/processed/qa*/
python scripts/run_question_b.py   # ~1-3 min: dev folds, model decision, final test, exceptions -> data/processed/qb*/
```

Both scripts run the main 57/60 rule, then the strict 60/60 sensitivity run (`--main-only` skips it).
Question B also runs a weekly-block cross-validation diagnostic once (`--skip-diagnostic` skips it).

## Run the dashboard

```powershell
streamlit run app/Home.py     # opens http://localhost:8501 (light/dark follows your OS; switch in ⋮ > Settings)
```

To inspect the methods and results without the dashboard, run
`jupyter lab notebooks/01_quickstart.ipynb` after `python scripts/run_all.py`. The notebook walks through
screening, completeness, cluster selection, forecast folds and model choice, and the strict 60/60 check.
It reads the pipeline's outputs and points to the implementation; it does not duplicate the methods.

| Page | What it shows |
|---|---|
| Energy composition (A) | Floor × end-use stack (1 Jan–21 Oct), Pareto zone ranking, hour × weekday load, after-hours load |
| Daily patterns (A) | Cluster profiles per end use, kWh per cluster / zone / floor / end use (CSV), calendar and weather labels, k-selection table, scaling screen, atypical days |
| Weather and AC (A) | Open-Meteo timeline, AC kWh vs weather, per-zone sensitivity (correlation only) |
| Next-day priorities (B) | Zone priority list, model decision on dev folds, final test with and without Floor 6, per-zone skill, actual vs predicted daily totals |
| Daily deviations (B) | Complete days outside each zone's locked expected daily range, exception rate and list (CSV) |
| Data quality | Complete zone-days (57/60 and 60/60), days lost per zone and month, cited holiday list checked against building load, spike and stuck-sensor screening |
| Sensitivity (60/60) | Every result re-run with the strict rule |

## Methodology rules (defined once in `src/cubems/config.py`)

- **Screening (`src/cubems/qc.py`):** each channel is compared with its own behaviour, and flagged minutes are
  set to missing before aggregation.
  - *Spike:* above 5× the channel's 99th percentile of active (> 0.05 kW) readings and above 1 kW. Using active
    readings keeps rarely-used AC units' normal running level from being flagged. 2019: 260 minutes removed,
    241 of them a Floor 3 glitch in February that hit several lighting and plug channels at once.
  - *Flatline (stuck sensor):* the same value for ≥ 60 min (AC), ≥ 360 min (lighting, plugs) or ≥ 120 min
    (temperature, RH). Power channels are checked only at ≥ 0.5 kW, because idle readings are naturally flat.
    Lighting can be legitimately constant, so its limit is longest. 2019: none found.
  - Impact: 3 complete zone-days lost. Question A's clusters are unchanged (99.99% same labels) and Question B
    results are identical. Thresholds use each file's full year (a cleaning step, not a model input).
- **Minute:** valid for an end use only if every circuit of that end use has a reading.
- **Hour:** valid with at least 57 of 60 valid minutes; kWh = mean kW × 1 h. Otherwise missing, never zero.
  Zeros are kept. Every result is re-run with the strict 60/60 rule.
- **Complete zone-day (Question A):** all 24 hours valid for every end use of the zone's group.
- **Complete AC zone-day (Question B):** all 24 AC hours valid; the target is their summed kWh. Missing days are
  excluded from scoring and never treated as zero.
- **Groups:** C1 = AC + lighting + plugs (24 zones, incl. F1z2, which runs 24/7 and is reported separately
  instead of clustered); C2 = lighting + plugs (8: F1z1, F1z3, F2z3,
  z3 on Floors 3–7); F1z4 = lighting only (kWh totals only).
- **Period:** all of 2019. Floor 6 stops 22 Oct 13:27 (the source file is truncated mid-row). Floor comparisons
  use 1 Jan–21 Oct, and the final test is reported with and without Floor 6.
- **Holidays:** the cited official 2019 list in `references/thai_holidays_2019.csv`. Labour Day and the
  Bangkok-only ASEAN Summit days are listed but not flagged, because the building ran normally.
- **Timestamps:** treated as Bangkok local time (assumption; AC switches on 08:00, off 12–13, off 17:00).
- **Indoor temperature and RH:** never model inputs (0% coverage Jan–Feb; no columns on Floor 1).
- **Weather:** Open-Meteo (ERA5) daily summaries label Question A clusters and enter Question B only with a
  one-day lag. Day-D observed weather is never a next-day forecast input.
- **Question B inputs:** zone, day of week, holiday and weekend flags, complete AC daily totals 1 and 7 days
  earlier, a previous-seven-day mean (with missing indicators), and prior-day weather. No day-D readings or A labels.
- **Decision rule:** boosting stays primary only if it beats Ridge on dev MAE **and** its |NMBE| ≤ 5%. Only one
  model goes to the final test. Daily exception ranges (zone × day type residual P5–P95) are locked on
  development residuals and used only after actual daily use is known.
- **Leakage checks (automated):** training dates precede test dates, encoders, scalers and fallback means are fit
  on training rows only, every model is scored on identical targets, and predictions are finite and non-negative.

## Repo layout

```
app/                 Streamlit dashboard (Home.py router + views/, common.py helpers)
src/cubems/          config, loader, aggregate, calendar, weather, clustering (A), forecast (B), anomaly
scripts/             run_all.py (everything), get_data.py, build_dataset.py, run_question_a.py, run_question_b.py
references/          thai_holidays_2019.csv (cited official holiday list)
notebooks/           read-only methods walkthrough using generated results
tests/               pytest
data/{raw,processed,external}/   gitignored source data and generated tables
```

## Open items

1. **Resolved: F1Z2 is reported separately.** It formed its own C1 cluster at every k, so no k passed rule 1
   (≥ 3 zones per cluster). It runs 24/7, and on workdays its AC is low in office hours and high 17:00–06:00
   (the inverse of the office schedule), which suggests a round-the-clock facility. It is now
   excluded from clustering (`config.QA_SEPARATE_ZONES`) and shown in its own card on the Daily patterns page.
   Without it, k = 3 passes rules 1–2 under both the 57/60 and 60/60 rules.
2. **Next-day priority is not an automatic fault diagnosis.** A high forecast may reflect normal operations; use
   the priority list to decide what to review. Exception flags are available only after the target day ends.
3. **Institution-specific closures:** compare widespread daily deviations with the university calendar.
4. **Timestamp time zone** is inferred, not confirmed. The cause of the ERA5 hourly jumps is unknown.
5. **Monotone constraints** on past AC totals are an assumption. Document them in the report.

## Checks

```powershell
pytest -q
ruff check src app scripts tests
```

## Data sources and citation

**Building data: CU-BEMS** (Chamchuri 5 building, Chulalongkorn University, Bangkok; 33 zones, 1-minute
resolution, 1 Jul 2018 – 31 Dec 2019). Licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
so cite it in the report and slides:

> Pipattanasomporn, M., Chitalia, G., Songsiri, J., Aswakul, C., Pora, W., Suwankawin, S., Audomvongseree, K., &
> Hoonchareon, N. (2020). CU-BEMS, smart building electricity consumption and indoor environmental sensor datasets.
> *Scientific Data*, 7, 241. <https://doi.org/10.1038/s41597-020-00582-3>

Files: <https://sgrudata.github.io/>, used unmodified. The truncated last row of `2019Floor6.csv` is part of the
published file; the loader drops it.

**Weather: Open-Meteo Historical Weather API** (ERA5-based reanalysis), [CC BY 4.0](https://open-meteo.com/en/terms),
free for non-commercial use. Credit: *"Weather data by Open-Meteo.com"*.

```bibtex
@article{pipattanasomporn2020cubems,
  title   = {{CU-BEMS}, smart building electricity consumption and indoor environmental sensor datasets},
  author  = {Pipattanasomporn, Manisa and Chitalia, Gopal and Songsiri, Jitkomut and Aswakul, Chaodit and
             Pora, Wanchalerm and Suwankawin, Surapong and Audomvongseree, Kulyos and Hoonchareon, Naebboon},
  journal = {Scientific Data},
  volume  = {7},
  pages   = {241},
  year    = {2020},
  doi     = {10.1038/s41597-020-00582-3}
}
```
