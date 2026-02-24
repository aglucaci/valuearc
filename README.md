# ValueArc

ValueArc is a systematic equity research pipeline designed to identify asymmetric opportunities using a conservative Margin of Safety (MOS) model, automated screening, and a live dark-mode quant dashboard.

The platform integrates:

* Rule-based screening
* Scenario-based earnings valuation
* Predictability and balance-sheet risk adjustments
* Static dashboards powered by GitHub-hosted data

---

## View the Dashboard

Open:

https://aglucaci.github.io/longarc/

or deploy through static hosting platforms such as GitHub Pages or Netlify.

---

# Core Philosophy

LongArc emphasizes:

* Downside protection
* Repeatable valuation logic
* Transparent assumptions
* Efficient visual scanning of opportunities

Rather than relying on a single fair-value estimate, LongArc calculates:

Bear / Base / Bull intrinsic value scenarios

Securities are ranked primarily using:

MOS_Bear_% (conservative margin of safety)

---

# Margin of Safety Model

The MOS engine runs locally and does not depend on external valuation sites.

## Intrinsic Value Framework

Normalized EPS
→ Two-stage discounted earnings stream
→ Predictability multiplier (quality haircut)
→ Balance-sheet floor valuation

### Stage Structure

* Growth Stage: N years
* Terminal Stage: M years
* No perpetuity assumption (intentionally conservative)

### Margin of Safety Definition

MOS = (Intrinsic Value − Price) / Intrinsic Value

Large negative MOS values indicate that price significantly exceeds the model’s intrinsic estimate and should be interpreted as an overvaluation flag rather than a literal downside percentage.

---

# Screener Pipeline

Finviz Filters
→ 10-year Price CAGR
→ Recommended MOS Calculation
→ Filtered Output CSV

Generated files:

```
outputs/
 ├── longarc_full_<timestamp>.csv
 ├── longarc_filtered_<timestamp>.csv
 └── longarc_filtered_TODAY.csv
```

---

## Features

* Dark professional interface
* MOS heat shading
* Company and ticker display
* Live sorting and filtering
* Search across ticker, sector, industry, and company
* Median and top-decile MOS metrics

No backend infrastructure is required.

---

# Usage

## Run the Screener

```
python longarc_screener.py
```

Dependencies:

```
pip install yfinance pandas numpy finvizfinance
```

---

## Publish Data

Commit the updated CSV:

```
outputs/longarc_filtered_TODAY.csv
```

GitHub Raw acts as the live data source.

---

# Output Columns

| Column               | Description                   |
| -------------------- | ----------------------------- |
| MOS_Bear_%           | Conservative margin of safety |
| MOS_Base_%           | Neutral scenario MOS          |
| MOS_Bull_%           | Upside scenario MOS           |
| Value_Bear/Base/Bull | Intrinsic values              |
| EPS_Norm             | Normalized earnings           |
| Quality_Mult         | Predictability adjustment     |
| Floor_Value          | Downside floor valuation      |
| 10yr_CAGR            | Historical price CAGR         |

---

# Design Principles

* Conservative valuation bias
* Minimal external dependencies
* Static-site deployment
* Institutional-style interface

---

# Notes

* Extremely negative MOS values are expected under conservative assumptions.
* Yahoo Finance fields may occasionally be incomplete or delayed.
* This tool is intended for research and educational purposes only.

---

# License

Internal research tooling. Adapt freely for personal or organizational use.
