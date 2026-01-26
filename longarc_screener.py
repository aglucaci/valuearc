# -*- coding: utf-8 -*-
"""
LONGARC Screener (clean script)

What it does
- Runs a Finviz fundamentals screener (finvizfinance Overview)
- Computes 10-year CAGR for each ticker using BATCHED yfinance downloads (rate-limit friendlier)
- Adds a GuruFocus DCF link per ticker
- Writes outputs to timestamped CSVs under an output directory

Why this version exists
- Your previous script was exported from a notebook and included `!pip install ...` lines and
  per-ticker yfinance calls that can trigger rate limits. fileciteturn2file0L12-L20
- This script removes notebook magics and uses chunked/batched downloads.

Usage
  python longarc_screener_clean.py
  python longarc_screener_clean.py --cagr-threshold 15 --years 10 --outdir outputs

Dependencies
  pip install finvizfinance yfinance pandas requests beautifulsoup4
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yfinance as yf
from finvizfinance.screener.overview import Overview


# -----------------------
# Defaults
# -----------------------
DEFAULT_FILTERS: Dict[str, str] = {
    "Debt/Equity": "Under 1",
    "EPS growthpast 5 years": "Over 15%",
    "Price/Free Cash Flow": "Under 50",
    "Return on Assets": "Positive (>0%)",
    "Return on Equity": "Over +15%",
    "Return on Investment": "Over +15%",
    "52-Week High/Low": "0-10% above Low",
}


# -----------------------
# Utilities
# -----------------------
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def chunked(items: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def gurufocus_dcf_url(ticker: str) -> str:
    return f"https://www.gurufocus.com/stock/{ticker}/dcf"


def compute_cagr_from_close(close: pd.Series, years: int) -> Optional[float]:
    """
    Compute CAGR (%) from a close-price Series using first/last valid values.
    """
    if close is None or close.empty:
        return None
    s = close.dropna()
    if s.empty:
        return None
    start_price = float(s.iloc[0])
    end_price = float(s.iloc[-1])
    if start_price <= 0:
        return None
    cagr = (end_price / start_price) ** (1.0 / years) - 1.0
    return round(cagr * 100.0, 2)


def yf_download_with_retry(
    tickers: List[str],
    period: str,
    interval: str = "1d",
    max_retries: int = 6,
    base_sleep_s: float = 2.0,
) -> pd.DataFrame:
    """
    Download batched prices with exponential backoff (helps for transient issues / 429s).
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                threads=False,   # less bursty
                progress=False,
            )
        except Exception as e:
            last_err = e
            sleep_s = base_sleep_s * (2 ** (attempt - 1))
            sleep_s = sleep_s * 1.25  # small jitter
            log(f"[yfinance] error on attempt {attempt}/{max_retries}: {e}")
            log(f"[yfinance] sleeping {sleep_s:.1f}s then retrying...")
            time.sleep(sleep_s)

    raise RuntimeError(f"yfinance download failed after {max_retries} attempts: {last_err}")


def add_cagr_batched(
    df: pd.DataFrame,
    years: int,
    chunk_size: int,
    sleep_between_chunks_s: float,
) -> pd.DataFrame:
    """
    Adds/overwrites df['10yr_CAGR'] using batched downloads.

    Notes:
    - Ensures the column ALWAYS exists to prevent KeyError downstream.
    - Uses 'period={years}y' which is the most stable approach for yfinance.
    """
    out = df.copy()
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()

    # Create the column up front so it always exists
    out["10yr_CAGR"] = pd.NA

    tickers = out["Ticker"].tolist()
    cagr_map: Dict[str, Optional[float]] = {}

    period = f"{years}y"

    for block in chunked(tickers, chunk_size):
        log(f"[prices] downloading {len(block)} tickers (period={period})...")
        prices = yf_download_with_retry(block, period=period, interval="1d")

        for t in block:
            try:
                close = None
                if isinstance(prices.columns, pd.MultiIndex):
                    # MultiIndex: (Ticker, OHLCV)
                    if t in prices.columns.get_level_values(0):
                        close = prices[t]["Close"]
                else:
                    # Single ticker case
                    if "Close" in prices.columns:
                        close = prices["Close"]

                cagr_map[t] = compute_cagr_from_close(close, years=years)
            except Exception as e:
                log(f"[cagr] failed for {t}: {e}")
                cagr_map[t] = None

        time.sleep(sleep_between_chunks_s)

    out["10yr_CAGR"] = out["Ticker"].map(cagr_map)
    return out


def normalize_percent_to_fraction(x) -> Optional[float]:
    """
    Finviz dividend is often a string like '1.23%'. Convert to fraction 0.0123.
    Leaves None/NA as None.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v / 100.0 if v > 1.0 else v
    s = str(x).strip()
    if not s:
        return None
    if s.endswith("%"):
        s = s[:-1].strip()
    try:
        v = float(s.replace(",", ""))
        return v / 100.0 if v > 1.0 else v
    except Exception:
        return None


# -----------------------
# Main
# -----------------------
def run_screener(filters: Dict[str, str]) -> pd.DataFrame:
    foverview = Overview()
    foverview.set_filter(filters_dict=filters)
    df = foverview.screener_view()
    if df is None or df.empty:
        raise RuntimeError("Finviz returned 0 rows. Check filters and connectivity.")
    if "Ticker" not in df.columns:
        raise RuntimeError("Finviz output missing 'Ticker' column.")
    return df


def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_csv(df: pd.DataFrame, outdir: str, name: str, stamp: str) -> str:
    path = os.path.join(outdir, f"{name}_{stamp}.csv")
    df.to_csv(path, index=False)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LONGARC screener (Finviz + 10y CAGR + GuruFocus links)")
    p.add_argument("--outdir", default="outputs", help="Directory to write CSV outputs")
    p.add_argument("--years", type=int, default=10, help="Years for CAGR calculation (default: 10)")
    p.add_argument("--cagr-threshold", type=float, default=15.0, help="Filter threshold for CAGR percent (default: 15)")
    p.add_argument("--chunk-size", type=int, default=40, help="Tickers per yfinance batch (default: 40)")
    p.add_argument("--sleep-between-chunks", type=float, default=3.0, help="Seconds to sleep between yfinance batches (default: 3)")
    p.add_argument("--no-filter", action="store_true", help="Do not apply CAGR filter; still write full output")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    outdir = ensure_outdir(args.outdir)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log("Running Finviz screener...")
    df = run_screener(DEFAULT_FILTERS)
    log(f"Finviz rows: {len(df)}")

    # Prefer Finviz dividend column to avoid yfinance .info calls.
    if "Dividend" in df.columns and "Dividend Yield" not in df.columns:
        df["Dividend Yield"] = df["Dividend"]

    if "Dividend Yield" in df.columns:
        df["Dividend_Yield_Fraction"] = df["Dividend Yield"].apply(normalize_percent_to_fraction)

    log(f"Computing {args.years}y CAGR (batched yfinance)...")
    df = add_cagr_batched(df, years=args.years, chunk_size=args.chunk_size, sleep_between_chunks_s=args.sleep_between_chunks)

    # Always add GuruFocus DCF link for convenience
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["GuruFocus_DCF_URL"] = df["Ticker"].apply(gurufocus_dcf_url)

    full_path = write_csv(df, outdir, "longarc_full", stamp)
    log(f"Wrote full CSV: {full_path}")

    if args.no_filter:
        log("Skipping CAGR filter (--no-filter). Done.")
        return 0

    df_cagr = pd.to_numeric(df["10yr_CAGR"], errors="coerce")
    df_flt = df[df_cagr > args.cagr_threshold].copy()
    df_flt = df_flt.sort_values("10yr_CAGR", ascending=False)

    flt_path = write_csv(df_flt, outdir, "longarc_filtered", stamp)
  
    # Write today
    flt_path_today = write_csv(df_flt, outdir, "longarc_filtered_TODAY", "")
  
    log(f"Wrote filtered CSV (CAGR > {args.cagr_threshold}%): {flt_path} ({len(df_flt)} rows)")

    if not df_flt.empty:
        log("GuruFocus DCF links (filtered):")
        for t in df_flt["Ticker"].tolist():
            print(f"{t}\t{gurufocus_dcf_url(t)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

