# LONGARC

**LONGARC** is an automated equity screening engine focused on identifying **high-quality compounders** and **mispriced value opportunities** using fundamentals, long-term growth, and intrinsic value signals.

The system runs end-to-end in **GitHub Actions**, producing timestamped CSV outputs that can be visualized, archived, or published automatically.

---

## Live Dashboard

https://aglucaci.github.io/longarc/

## What LONGARC Does

LONGARC combines multiple layers of analysis:

1. **Fundamental Screening (Finviz)**
   - Balance sheet quality
   - Profitability (ROE / ROI / ROA)
   - Cash flow discipline
   - Reasonable valuation constraints

2. **Long-Term Growth**
   - Computes **10-year CAGR** from historical price data
   - Filters for durable compounders (default: CAGR > 15%)

3. **Intrinsic Value Check (GuruFocus)**
   - Extracts **DCF Margin of Safety (%)**
   - Supports **negative MOS** (overvaluation detection)
   - Uses browser-based fallback to handle anti-bot protections

4. **Automated Output**
   - Timestamped CSVs committed back to the repo
   - Artifacts uploaded for each run
   - Ready for dashboards, Substack posts, or further analysis

---

## Repository Structure

```text
.
├── longarc_screener.py               # Main screener script
├── requirements.txt                  # Python dependencies
├── scripts/
│   └── get_gurufocus_margin_of_safety.py
│                                     # MOS extraction logic (requests + Playwright)
├── outputs/
│   ├── longarc_full_<timestamp>.csv  # Full universe
│   └── longarc_filtered_<timestamp>.csv
│                                     # CAGR-filtered universe with MOS
├── .github/
│   └── workflows/
│       └── longarc_screener.yml      # GitHub Actions automation
└── README.md








