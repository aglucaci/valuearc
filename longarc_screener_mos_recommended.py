# -*- coding: utf-8 -*-
"""
LONGARC Screener (updated for recommended MOS model)

What changed vs your previous version:
- Removed GuruFocus fetching (no 403 / no Playwright).
- Adds a *local* "recommended" Margin of Safety (MOS) calculation based on:
    - Normalized EPS (median of recent annual EPS; fallback to trailing EPS)
    - Two-stage discounted EPS stream (growth stage + terminal stage; no perpetuity)
    - Bear/Base/Bull scenarios for growth (rule-based from EPS CAGR; fallback defaults)
    - Rule-based predictability haircut (EPS volatility/negatives + leverage proxy)
    - Downside floor (max of haircutted TBV/share and net-cash/share)

New output columns (in the filtered df):
- MOS_Bear_%
- MOS_Base_%
- MOS_Bull_%
- Value_Bear
- Value_Base
- Value_Bull
- EPS_Norm
- EPS_CAGR_Proxy
- Quality_Mult
- Floor_Value

Outputs:
- outputs/longarc_full_<timestamp>.csv
- outputs/longarc_filtered_<timestamp>.csv
- outputs/longarc_filtered_TODAY.csv

Run:
  python longarc_screener.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from finvizfinance.screener.overview import Overview


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
# CAGR helper (price CAGR; keeps your original behavior)
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


# -----------------------
# Recommended MOS model (local; no GuruFocus)
# -----------------------
def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        x = float(x)
        if np.isnan(x):
            return None
        return x
    except Exception:
        return None


def _pct(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(100.0 * x, 2)


def _mos(price: Optional[float], value: Optional[float]) -> Optional[float]:
    # MOS = (V - P) / V
    if price is None or value is None or value == 0:
        return None
    return (value - price) / value


def _fetch_price(t: yf.Ticker) -> Tuple[Optional[float], str]:
    # Prefer history close (most reliable)
    try:
        hist = t.history(period="5d", interval="1d", auto_adjust=False)
        if hist is not None and not hist.empty:
            close = _safe_float(hist["Close"].dropna().iloc[-1])
            if close is not None:
                return close, "history.Close(last)"
    except Exception:
        pass

    # fast_info
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            p = _safe_float(fi.get("lastPrice")) or _safe_float(fi.get("last_price"))
            if p is not None:
                return p, "fast_info.lastPrice"
    except Exception:
        pass

    # info
    try:
        info = t.get_info() or {}
        p = _safe_float(info.get("currentPrice")) or _safe_float(info.get("regularMarketPrice"))
        if p is not None:
            return p, "info.currentPrice/regularMarketPrice"
    except Exception:
        pass

    return None, "NA"


def _fetch_shares_mcap_trailing_eps(t: yf.Ticker) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    shares = None
    mcap = None
    trailing_eps = None

    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            shares = _safe_float(fi.get("shares"))
            mcap = _safe_float(fi.get("marketCap")) or _safe_float(fi.get("market_cap"))
    except Exception:
        pass

    try:
        info = t.get_info() or {}
        if shares is None:
            shares = _safe_float(info.get("sharesOutstanding"))
        if mcap is None:
            mcap = _safe_float(info.get("marketCap"))
        trailing_eps = _safe_float(info.get("trailingEps"))
    except Exception:
        pass

    return shares, mcap, trailing_eps


def _fetch_eps_history_annual(t: yf.Ticker, years: int = 5) -> List[float]:
    """Return a list of annual EPS values (most recent first) if available."""
    rows = ["Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS"]
    try:
        is_df = t.income_stmt  # annual
        if is_df is None or is_df.empty:
            return []
        cols = list(is_df.columns)[: max(1, years)]
        for r in rows:
            if r in is_df.index:
                vals = [_safe_float(is_df.loc[r, c]) for c in cols]
                vals = [v for v in vals if v is not None]
                if len(vals) >= 2:
                    return vals
    except Exception:
        return []
    return []


def _fetch_balance_sheet_floors(t: yf.Ticker, shares: Optional[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """cash, debt, tbv_ps, net_cash_ps (best effort)."""
    cash = debt = tbv_ps = net_cash_ps = None

    try:
        bs = t.balance_sheet
    except Exception:
        bs = None

    def bs_get(*keys) -> Optional[float]:
        if bs is None or getattr(bs, "empty", True):
            return None
        col = bs.columns[0]
        for k in keys:
            if k in bs.index:
                return _safe_float(bs.loc[k, col])
        return None

    cash = bs_get(
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Short Term Investments",
        "Cash",
    )
    debt = bs_get(
        "Total Debt",
        "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation",
    )

    equity = bs_get("Total Stockholder Equity", "Stockholders Equity")
    goodwill = bs_get("Goodwill")
    intang = bs_get("Intangible Assets", "Other Intangible Assets")

    if equity is not None and shares not in (None, 0):
        tangible_equity = equity - (goodwill or 0.0) - (intang or 0.0)
        tbv_ps = tangible_equity / shares

    if shares not in (None, 0) and cash is not None:
        net_cash_ps = (cash - (debt or 0.0)) / shares

    return cash, debt, tbv_ps, net_cash_ps


def _normalized_eps(eps_hist: List[float], trailing_eps: Optional[float]) -> Tuple[Optional[float], str]:
    if eps_hist and len(eps_hist) >= 2:
        return float(np.median(eps_hist)), f"median({len(eps_hist)}y EPS)"
    if trailing_eps is not None:
        return float(trailing_eps), "trailingEps (proxy)"
    return None, "NA"


def _eps_cagr(eps_hist: List[float]) -> Optional[float]:
    """CAGR from oldest to newest; eps_hist is most recent first."""
    if not eps_hist or len(eps_hist) < 2:
        return None
    newest = eps_hist[0]
    oldest = eps_hist[-1]
    if newest is None or oldest is None or oldest <= 0 or newest <= 0:
        return None
    n = len(eps_hist) - 1
    return (newest / oldest) ** (1.0 / n) - 1.0


def _scenario_growth_rates(hist_cagr: Optional[float]) -> Dict[str, float]:
    if hist_cagr is None:
        return {"bear": 0.05, "base": 0.10, "bull": 0.15}
    h = max(0.0, float(hist_cagr))
    # conservative caps for screeners
    return {
        "bear": min(0.10, 0.5 * h),
        "base": min(0.20, 1.0 * h),
        "bull": min(0.30, 1.5 * h),
    }


def _predictability_multiplier(eps_hist: List[float], debt: Optional[float], mcap: Optional[float]) -> float:
    mult = 1.0
    eps = [e for e in eps_hist if e is not None]
    if len(eps) >= 3:
        mean = float(np.mean(eps))
        std = float(np.std(eps, ddof=0))
        cv = std / (abs(mean) + 1e-9)
        neg_frac = float(np.mean([1.0 if x < 0 else 0.0 for x in eps]))

        if cv > 0.75:
            mult *= 0.75
        elif cv > 0.40:
            mult *= 0.85
        elif cv > 0.25:
            mult *= 0.92

        if neg_frac >= 0.34:
            mult *= 0.80
        elif neg_frac > 0.0:
            mult *= 0.90

    if debt is not None and mcap is not None and mcap > 0:
        lev = float(debt / mcap)
        if lev > 0.60:
            mult *= 0.80
        elif lev > 0.30:
            mult *= 0.90
        elif lev > 0.15:
            mult *= 0.95

    return float(max(0.60, min(1.00, mult)))


def _two_stage_eps_value(eps0: float, discount: float, stage1_years: int, g1: float, stage2_years: int, g2: float) -> float:
    """Discounted EPS stream; no perpetuity beyond stage2."""
    r = discount
    eps = eps0
    pv = 0.0

    for t in range(1, stage1_years + 1):
        eps *= (1.0 + g1)
        pv += eps / ((1.0 + r) ** t)

    for t in range(stage1_years + 1, stage1_years + stage2_years + 1):
        eps *= (1.0 + g2)
        pv += eps / ((1.0 + r) ** t)

    return float(pv)


def _floor_value(tbv_ps: Optional[float], net_cash_ps: Optional[float], tbv_haircut: float = 0.80, net_cash_haircut: float = 0.80) -> float:
    candidates = [0.0]
    if tbv_ps is not None:
        candidates.append(tbv_haircut * tbv_ps)
    if net_cash_ps is not None:
        candidates.append(net_cash_haircut * net_cash_ps)
    return float(max(candidates))


def compute_recommended_mos(
    ticker: str,
    discount: float = 0.11,
    stage1_years: int = 10,
    stage2_years: int = 10,
    terminal_rate: float = 0.04,
) -> Dict[str, Optional[float]]:
    """Compute bear/base/bull intrinsic values and MOS using the recommended model."""
    tk = ticker.strip().upper()
    t = yf.Ticker(tk)

    price, _ = _fetch_price(t)
    shares, mcap, trailing_eps = _fetch_shares_mcap_trailing_eps(t)
    if mcap is None and price is not None and shares is not None:
        mcap = price * shares

    eps_hist = _fetch_eps_history_annual(t, years=5)
    cash, debt, tbv_ps, net_cash_ps = _fetch_balance_sheet_floors(t, shares)

    eps_norm, _eps_src = _normalized_eps(eps_hist, trailing_eps)
    eps_cagr = _eps_cagr(eps_hist)
    g = _scenario_growth_rates(eps_cagr)
    qual = _predictability_multiplier(eps_hist if eps_hist else ([trailing_eps] if trailing_eps is not None else []), debt, mcap)
    floor_v = _floor_value(tbv_ps, net_cash_ps)

    out: Dict[str, Optional[float]] = {
        "Price": price,
        "EPS_Norm": eps_norm,
        "EPS_CAGR_Proxy": eps_cagr,
        "Quality_Mult": qual,
        "Floor_Value": floor_v,
    }

    # If we can't value, return what we have
    if price is None or eps_norm is None:
        out.update({
            "Value_Bear": None, "Value_Base": None, "Value_Bull": None,
            "MOS_Bear_%": None, "MOS_Base_%": None, "MOS_Bull_%": None,
        })
        return out

    for name in ("bear", "base", "bull"):
        v_raw = _two_stage_eps_value(
            eps0=float(eps_norm),
            discount=float(discount),
            stage1_years=int(stage1_years),
            g1=float(g[name]),
            stage2_years=int(stage2_years),
            g2=float(terminal_rate),
        )
        v_hair = v_raw * float(qual)
        v_final = max(v_hair, float(floor_v))

        mos_val = _mos(float(price), float(v_final))

        out[f"Value_{name.capitalize()}"] = v_final
        out[f"MOS_{name.capitalize()}_%"] = _pct(mos_val)

    return out


def main() -> int:
    os.makedirs("outputs", exist_ok=True)

    foverview = Overview()
    foverview.set_filter(filters_dict=filters_dict)
    df = foverview.screener_view()

    if df is None or df.empty:
        raise SystemExit("Finviz returned no rows. Check your filters or connectivity.")

    # Ensure column exists up front
    df["10yr_CAGR"] = pd.NA
    print()
    print("[INFO] Calculating Compound Annual Growth Rate (CAGR)")

    # CAGR Analysis (price CAGR)
    for index, row in df.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        print(ticker, "calculating CAGR...")
        cagr = calculate_cagr(ticker, years=10)
        if cagr is not None:
            df.at[index, "10yr_CAGR"] = cagr
        time.sleep(5)

    # Filter by CAGR
    df_cagr = pd.to_numeric(df["10yr_CAGR"], errors="coerce")
    df_flt = df[df_cagr > 15].copy()

    # --- NEW: Compute recommended MOS locally and store in dataframe ---
    for col in [
        "MOS_Bear_%", "MOS_Base_%", "MOS_Bull_%",
        "Value_Bear", "Value_Base", "Value_Bull",
        "EPS_Norm", "EPS_CAGR_Proxy", "Quality_Mult", "Floor_Value",
    ]:
        df_flt[col] = pd.NA

    print()
    print("[INFO] Computing recommended Margin of Safety (MOS) locally")

    for index, row in df_flt.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        print(ticker, "computing MOS...")
        try:
            res = compute_recommended_mos(
                ticker=ticker,
                discount=0.11,
                stage1_years=10,
                stage2_years=10,
                terminal_rate=0.04,
            )
        except Exception as e:
            print(f"[mos] Error with {ticker}: {e}")
            res = {}

        for k, v in res.items():
            if k in df_flt.columns and v is not None:
                df_flt.at[index, k] = v

        time.sleep(3)

    # Write outputs (timestamped)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = f"outputs/longarc_full_{stamp}.csv"
    flt_path = f"outputs/longarc_filtered_{stamp}.csv"
    flt_path_today = f"outputs/longarc_filtered_TODAY.csv"

    df.to_csv(full_path, index=False)
    df_flt.to_csv(flt_path, index=False)
    df_flt.to_csv(flt_path_today, index=False)

    print(f"Wrote: {full_path}")
    print(f"Wrote: {flt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
