# -*- coding: utf-8 -*-
"""
LONGARC Screener (updated)

Updates:
- Cleaned notebook magics (`!pip install ...`) out of the script.
- Adds DCF Margin of Safety (MOS %) from GuruFocus via scripts/get_gurufocus_margin_of_safety.py
  into the dataframe (column: "DCF_Margin_of_Safety_%") instead of storing a GuruFocus link.

Notes:
- GuruFocus may block simple HTTP requests (403). The helper will fall back to Playwright.
- If you run this in GitHub Actions, you likely need:
    pip install playwright && playwright install chromium

Outputs:
- outputs/longarc_full_<timestamp>.csv
- outputs/longarc_filtered_<timestamp>.csv

Run:
  python longarc_screener.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from finvizfinance.screener.overview import Overview

# Import MOS helper (expects repo path scripts/get_gurufocus_margin_of_safety.py)
from scripts.get_gurufocus_margin_of_safety import get_margin_of_safety


# -----------------------
# Screener filters (Finviz)
# -----------------------
filters_dict = {
    "Debt/Equity": "Under 1",
    "EPS growthpast 5 years": "Over 15%",
    "Price/Free Cash Flow": "Under 50",
    "Return on Assets": "Positive (>0%)",
    "Return on Equity": "Over +15%",
    "Return on Investment": "Over +15%",
    "52-Week High/Low": "0-10% above Low",
}


# -----------------------
# CAGR helper (simple; you can swap in your batched version if desired)
# -----------------------
def calculate_cagr(ticker: str, years: int = 10) -> float | None:
    try:
        stock = yf.Ticker(ticker)
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365 * years)

        hist = stock.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        if hist is None or hist.empty:
            return None

        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])

        if start_price <= 0:
            return None

        cagr = (end_price / start_price) ** (1 / years) - 1
        return round(cagr * 100, 2)
    except Exception as e:
        print(f"[cagr] Error with {ticker}: {e}")
        return None


def main() -> int:
    os.makedirs("outputs", exist_ok=True)

    foverview = Overview()
    foverview.set_filter(filters_dict=filters_dict)
    df = foverview.screener_view()

    if df is None or df.empty:
        raise SystemExit("Finviz returned no rows. Check your filters or connectivity.")

    # Ensure column exists up front
    df["10yr_CAGR"] = pd.NA

    # CAGR Analysis
    for index, row in df.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        print(ticker, "calculating CAGR...")
        cagr = calculate_cagr(ticker, years=10)
        if cagr is not None:
            df.at[index, "10yr_CAGR"] = cagr
        time.sleep(1)

    # Filter by CAGR
    df_cagr = pd.to_numeric(df["10yr_CAGR"], errors="coerce")
    df_flt = df[df_cagr > 15].copy()

    # --- NEW: Fetch GuruFocus MOS and store it in the dataframe ---
    df_flt["DCF_Margin_of_Safety_%"] = pd.NA

    for index, row in df_flt.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        print(ticker, "fetching GuruFocus MOS...")
        try:
            mos = get_margin_of_safety(ticker, prefer_requests=True, headless=True)
        except TypeError:
            # If their helper signature differs, fall back to calling with just ticker
            mos = get_margin_of_safety(ticker)

        if mos is not None:
            df_flt.at[index, "DCF_Margin_of_Safety_%"] = float(mos)

        time.sleep(1)

    # Write outputs (timestamped)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = f"outputs/longarc_full_{stamp}.csv"
    flt_path = f"outputs/longarc_filtered_{stamp}.csv"
    flt_path_today = f"outputs/longarc_filtered_TODAY.csv"
  
    df.to_csv(full_path, index=False)
    df_flt.to_csv(flt_path, index=False)
    df_flt.to_csv(flt_path_today, index=False)
  
    print(f"✅ Wrote: {full_path}")
    print(f"✅ Wrote: {flt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
