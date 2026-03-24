# ValueArc

ValueArc is a systematic equity screening pipeline that ranks ideas by conservative margin of safety. It combines a Finviz ruleset, long-horizon price CAGR, and a local bear/base/bull intrinsic value model built from normalized EPS, quality haircuts, and balance-sheet floor values.

The repo publishes two things:

- Fresh CSV outputs in [`outputs/`](./outputs)
- A static dashboard in [`docs/index.html`](./docs/index.html) for GitHub Pages

## Live dashboard

https://aglucaci.github.io/valuearc/

The dashboard now reads a published metadata file and shows the timestamp of the last completed screener run, not the viewer's current clock.

## Repo structure

```text
.
|-- .github/workflows/          # scheduled screener automation
|-- docs/
|   |-- archive/                # older dashboard variants
|   |-- data/latest-run.json    # published run metadata for the site
|   `-- index.html              # GitHub Pages dashboard
|-- logo/                       # brand assets
|-- outputs/                    # generated CSVs and latest_run.json
|-- scripts/
|   |-- margin_of_safety.py     # standalone MOS analysis helper
|   `-- run_screener.py         # main screener entrypoint
|-- longarc_screener_mos_recommended.py  # compatibility wrapper
`-- requirements.txt
```

## How the screener works

1. Pull the Finviz screen defined in [`scripts/run_screener.py`](./scripts/run_screener.py).
2. Calculate 10-year price CAGR per ticker.
3. Filter for CAGR above 15%.
4. Compute local bear/base/bull intrinsic values from:
   - normalized EPS
   - scenario growth assumptions
   - predictability haircuts
   - balance-sheet floor values
5. Write timestamped outputs plus `longarc_filtered_TODAY.csv`.
6. Write `latest_run.json` so the dashboard can display the last completed run timestamp.

## Local usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full screener:

```bash
python scripts/run_screener.py
```

The legacy wrapper still works:

```bash
python longarc_screener_mos_recommended.py
```

Run the standalone MOS helper for one or more tickers:

```bash
python scripts/margin_of_safety.py QLYS AAPL MSFT
```

## Generated files

Each successful screener run writes:

- `outputs/longarc_full_<timestamp>.csv`
- `outputs/longarc_filtered_<timestamp>.csv`
- `outputs/longarc_filtered_TODAY.csv`
- `outputs/latest_run.json`
- `docs/data/latest-run.json`

## GitHub Actions

Two workflows are included:

- `daily_py.yaml` for a weekday scheduled run
- `hourly_market_hours_update.yml` for hourly updates during US market hours

The workflows now:

- run the organized `scripts/run_screener.py` entrypoint
- validate required output files before upload/commit
- publish run metadata for the dashboard
- use guarded commits and rebasing to reduce push failures
- skip hourly runs outside market hours without failing the job

## Notes

- The model is intentionally conservative. Very negative MOS values should be read as overvaluation flags, not literal downside forecasts.
- Data quality depends on upstream Yahoo Finance and Finviz responses.
- Outputs in this repo are research artifacts, not investment advice.
