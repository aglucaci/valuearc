# 📊 Margin of Safety Screener --- Recommended Model

A **robust, conservative valuation framework** for calculating Margin of
Safety (MOS) using normalized earnings, scenario analysis, quality
adjustments, and downside floor protection.

This script is designed for **systematic equity research workflows**
where consistency and defensibility matter more than single-point
valuations.

------------------------------------------------------------------------

# 🧠 Overview

Traditional MOS calculations often rely on:

-   one optimistic DCF
-   noisy TTM metrics
-   subjective fair values

This model improves reliability by combining:

✅ Normalized earnings power\
✅ Bear / Base / Bull growth scenarios\
✅ Rule-based predictability haircuts\
✅ Balance-sheet downside floor valuation

------------------------------------------------------------------------

# 🧮 Margin of Safety Formula

MOS = (Value − Price) / Value\
MOS = 1 − (Price / Value)

------------------------------------------------------------------------

# ⚙️ How It Works

## 1️⃣ Data Collection (yfinance)

The script pulls:

-   Current price
-   Shares outstanding
-   Market cap
-   Trailing EPS (proxy)
-   Annual diluted EPS history
-   Cash and debt
-   Tangible book value components

No external scraping or proprietary data sources are used.

------------------------------------------------------------------------

## 2️⃣ Normalized Earnings

Normalized EPS = median(last N annual EPS)

Fallback: trailingEps (TTM)

------------------------------------------------------------------------

## 3️⃣ Growth Scenarios

  Scenario   Growth Rule
  ---------- --------------------------------
  Bear       0.5 × historical CAGR (capped)
  Base       1.0 × historical CAGR (capped)
  Bull       1.5 × historical CAGR (capped)

Fallback defaults: bear = 5%\
base = 10%\
bull = 15%

------------------------------------------------------------------------

## 4️⃣ Two-Stage Earnings DCF

Stage 1 --- Growth Phase\
Stage 2 --- Terminal Phase

Each year's EPS is discounted:

PV = EPS_t / (1 + r)\^t

Value = Growth Value + Terminal Value

(No perpetuity --- intentionally conservative.)

------------------------------------------------------------------------

## 5️⃣ Predictability Multiplier

Quality haircut based on:

-   EPS volatility
-   Negative EPS frequency
-   Leverage proxy (debt / market cap)

Range: 0.60 → low predictability\
1.00 → high predictability

------------------------------------------------------------------------

## 6️⃣ Downside Floor Value

Floor = max( 0.8 × Tangible Book Value per share, 0.8 × Net Cash per
share, 0 )

Final intrinsic value:

Value_final = max(Earnings_Value × Quality_Multiplier, Floor)

------------------------------------------------------------------------

## 7️⃣ Margin of Safety

MOS_bear\
MOS_base\
MOS_bull

⭐ Recommended ranking metric: MOS_bear

------------------------------------------------------------------------

# 🚀 Usage

Basic run:

python scripts/mos_recommended.py QLYS

Multiple tickers:

python scripts/mos_recommended.py QLYS AAPL MSFT

Export CSV:

python scripts/mos_recommended.py QLYS AAPL --csv results.csv

------------------------------------------------------------------------

# 📈 Output Fields

  Column                 Description
  ---------------------- ------------------------
  price                  current market price
  eps_norm               normalized EPS
  quality_mult           predictability haircut
  floor_value            downside floor
  value_bear/base/bull   intrinsic values
  mos_bear/base/bull     margin of safety

------------------------------------------------------------------------

# 🔬 Design Principles

-   Conservative by default
-   Transparent assumptions
-   Scenario-driven
-   Suitable for large-scale screening

------------------------------------------------------------------------

# ⚠️ Notes

-   trailingEps from yfinance is a proxy.
-   Two-stage EPS DCF avoids inflated perpetuity assumptions.
-   MOS is a valuation risk metric, not a trading signal.

------------------------------------------------------------------------

# 📜 License

Internal research tool. Modify freely.
