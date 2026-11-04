# Where to Look First: Energy Priorities and Zone Exceptions

**IS4001 Group 7 · CU-BEMS case study.** We show facilities managers where the building's
electricity goes (Question A) and which zones are drifting from their normal behaviour
(Question B), so they know where to look first.

| | Question A (descriptive) | Question B (anomaly detection) |
|---|---|---|
| Asks | What drives consumption, and how does it vary across floors, end uses and operating patterns? | Which zones show persistent, unexpected changes in AC operation or indoor conditions? |
| Window | Jan–Dec 2019 | Baseline Jan–Aug 2019 → holdout Sep–Dec 2019, plus within-season month folds |
| KPIs | Normalised kWh baseline, % energy change | Event rate, duration burden (per valid monitored hour) |

## Stack

Python 3.11+ end to end: **pandas/pyarrow** for the pipeline, **scikit-learn** (K-means) and
**XGBoost** for modelling, and **Streamlit + Plotly** for the localhost dashboard. Analysis code
lives in an importable package (`src/cubems`), so notebooks and the dashboard share one
implementation of every rule.

## Setup (Windows, PowerShell)

```powershell
git clone <repo-url>; cd IS4001_CU_BEMS_case_study
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt     # also installs the local `cubems` package in editable mode
```

macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

## Data

The raw data (14 CSVs, ~700 MB) is **not** in this repo, and neither is anything derived from it.
Every teammate fetches and builds it locally with two commands (venv active):

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
| `data/processed/zone_hourly.parquet` | zone × hour | kWh per end use (AC / Light / Plug), valid-minute counts, indoor sensor means |
| `data/processed/zone_day.parquet` | zone × day | kWh, joint-meter coverage, `usable` flag, calendar (season, Thai holidays, day type, window), weather features `wx_*` |
| `data/processed/zones.parquet` | zone | Model group, AC unit count, sensors |
| `data/processed/weather_daily.parquet` | day | Mean/max/min temperature, RH, dew point, radiation, cooling degree-hours |

## Run the dashboard

```powershell
streamlit run app/Home.py     # opens http://localhost:8501 (light/dark follows your OS; switch in ⋮ > Settings)
```

| Page | What it shows |
|---|---|
| Data quality | Recomputed proposal scope table, zone × month usable heatmap, coverage per day, extreme-reading register |
| Energy composition (Q-A) | Floor × end-use stack, Pareto zone ranking (kWh per usable day), hour × weekday load, after-hours load |
| Weather and AC (Q-A) | Open-Meteo timeline, AC kWh vs weather feature, per-zone sensitivity (kWh per °C what-if) |
| Daily profiles | K-means on normalised 24-h profiles (Model 1 / Model 2), silhouette, pattern by day type / zone / month |
| Seasonal folds (Q-B) | Leave-one-month-out within Hot / Rainy / Cool, fold coverage, optional 2018 months |
| Anomaly explorer (Q-B) | P5–P95 baseline band, sustained-episode detection, event rate and duration burden KPIs, exception list (CSV), band chart |

## Methodology rules (defined once in `src/cubems/config.py`)

- **Usable zone-day:** at least 95% of the day's 1,440 minutes have *every* power channel of the zone recorded.
  Gap days are dropped and never filled with zeros.
- **Energy:** hourly kWh = Σ valid minute kW / 60. A valid hour has ≥ 57 minutes.
- **Scope:** Model 1 = 23 AC + lighting + plug zones (Floor 2 z1, z2, z4; Floors 3–7 z1, z2, z4, z5).
  Model 2 = 8 lighting + plug zones (Floor 1 z1, z3; zone 3 on Floors 2–7). Floor 1 z2 and z4 are out of scope.
- **Extreme readings:** any minute above 500 kW is flagged and kept. Test results with and without these days.
- **Indoor temperature and RH:** context only, never used as clustering input.
- **Weather:** Open-Meteo reanalysis for the site, joined on date and analysed against Model 1 zones only.
  Treat any link as correlation, not cause. Cite *"Weather data by Open-Meteo.com"* (CC BY 4.0).
- **Seasons:** whole months, Hot = Mar–May, Rainy = Jun–Oct, Cool = Nov–Feb.
- **Floor 6** data ends on 22 Oct 2019 (the source file is truncated mid-row). Report it separately.

The coverage numbers reproduce the proposal's scope table to within 0.2 percentage points.

## Repo layout

```
app/                 Streamlit dashboard (Home.py router + views/, common.py helpers)
src/cubems/          config, loader, aggregate, weather, folds, anomaly
scripts/             get_data.py (fetch + verify raw CSVs), build_dataset.py (raw -> processed)
notebooks/           exploration (import from cubems; never re-implement rules)
tests/               pytest
data/{raw,interim,processed,external}/   gitignored
models/, reports/figures/                gitignored outputs
```

## Open decisions / next steps

1. **2018 Jul–Dec as extra folds:** a methodology change, so the team needs to agree before using it (toggle on the Seasonal folds page).
2. **Weather-aware expected-consumption model (X factor):** fit XGBoost per zone (or pooled with zone features) on
   hour, day type and `wx_*` features using the within-season folds, then feed its prediction interval into
   `cubems.anomaly.detect_episodes` in place of the empirical band.
3. **K-means:** fix k per model using silhouette and seed stability, then lock the cluster-conditioned baseline.
4. **Lock thresholds** (band percentiles, margin, minimum hours) before scoring the holdout, to avoid tuning on test data.

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
