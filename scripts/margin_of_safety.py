#!/usr/bin/env python3
"""
mos_recommended.py

A "recommended" Margin of Safety workflow for a screener:
- Compute an earnings-power intrinsic value per share using a *two-stage EPS stream DCF*
- Use *normalized EPS* (median of last N annual Diluted EPS, fallback to trailing EPS)
- Build *bear/base/bull* scenarios for growth assumptions (rule-based, conservative caps)
- Apply a *rule-based predictability haircut* (quality multiplier) based on EPS volatility,
  negative EPS frequency, and leverage proxy
- Add a *floor value* (asset-ish downside anchor): conservative tangible book/share and net cash/share
- Final value per scenario = max( haircutted EPS-DCF value, floor value )
- MOS per scenario = (V - P)/V ; rank on MOS_bear

No GuruFocus fetching. yfinance only.

Usage:
  python scripts/mos_recommended.py QLYS
  python scripts/mos_recommended.py QLYS AAPL MSFT --csv out.csv
  python scripts/mos_recommended.py QLYS --discount 0.11 --stage1_years 10 --stage2_years 10
"""

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# Utils
# ----------------------------
def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None


def pct(x: Optional[float]) -> str:
    return "NA" if x is None else f"{100.0 * x:.2f}%"


def money(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    ax = abs(x)
    if ax >= 1e12:
        return f"{x/1e12:.3f}T"
    if ax >= 1e9:
        return f"{x/1e9:.3f}B"
    if ax >= 1e6:
        return f"{x/1e6:.3f}M"
    if ax >= 1e3:
        return f"{x/1e3:.3f}K"
    return f"{x:.2f}"


def mos(price: Optional[float], value: Optional[float]) -> Optional[float]:
    """MOS = (V - P)/V"""
    if price is None or value is None or value == 0:
        return None
    return (value - price) / value


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ----------------------------
# Data snapshot
# ----------------------------
@dataclass
class Snapshot:
    ticker: str
    price: Optional[float]
    shares: Optional[float]
    market_cap: Optional[float]
    trailing_eps: Optional[float]

    # EPS history for normalization (annual diluted EPS if available)
    eps_hist: List[float]

    cash: Optional[float]
    debt: Optional[float]
    tangible_book_ps: Optional[float]
    net_cash_ps: Optional[float]

    src: Dict[str, str]


def fetch_price(t: yf.Ticker) -> Tuple[Optional[float], str]:
    # history close tends to be most reliable
    try:
        hist = t.history(period="5d", interval="1d", auto_adjust=False)
        if hist is not None and not hist.empty:
            close = safe_float(hist["Close"].dropna().iloc[-1])
            if close is not None:
                return close, "history.Close(last)"
    except Exception:
        pass

    # fast_info
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            p = safe_float(fi.get("lastPrice")) or safe_float(fi.get("last_price"))
            if p is not None:
                return p, "fast_info.lastPrice"
    except Exception:
        pass

    # info
    try:
        info = t.get_info() or {}
        p = safe_float(info.get("currentPrice")) or safe_float(info.get("regularMarketPrice"))
        if p is not None:
            return p, "info.currentPrice/regularMarketPrice"
    except Exception:
        pass

    return None, "NA"


def fetch_shares_mcap_eps(t: yf.Ticker) -> Tuple[Optional[float], Optional[float], Optional[float], Dict[str, str]]:
    src = {}
    shares = None
    mcap = None
    trailing_eps = None

    # fast_info first
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            shares = safe_float(fi.get("shares"))
            mcap = safe_float(fi.get("marketCap")) or safe_float(fi.get("market_cap"))
            if shares is not None:
                src["shares"] = "fast_info.shares"
            if mcap is not None:
                src["market_cap"] = "fast_info.marketCap"
    except Exception:
        pass

    # info fallback
    try:
        info = t.get_info() or {}
        if shares is None:
            shares = safe_float(info.get("sharesOutstanding"))
            if shares is not None:
                src["shares"] = "info.sharesOutstanding"
        if mcap is None:
            mcap = safe_float(info.get("marketCap"))
            if mcap is not None:
                src["market_cap"] = "info.marketCap"
        trailing_eps = safe_float(info.get("trailingEps"))
        if trailing_eps is not None:
            src["trailing_eps"] = "info.trailingEps"
        else:
            src["trailing_eps"] = "NA"
    except Exception:
        src.setdefault("trailing_eps", "NA")

    return shares, mcap, trailing_eps, src


def fetch_eps_history_annual_diluted(t: yf.Ticker, years: int = 5) -> Tuple[List[float], str]:
    """
    Best-effort EPS history for normalization.
    We try:
      - ticker.income_stmt (annual) with a row like 'Diluted EPS'
    """
    # yfinance index labels vary; keep a list
    eps_rows = [
        "Diluted EPS",
        "DilutedEPS",
        "Basic EPS",
        "BasicEPS",
    ]

    try:
        is_df = t.income_stmt  # annual income statement
        if is_df is not None and not is_df.empty:
            # columns are periods (most recent first)
            cols = list(is_df.columns)[: max(1, years)]
            for r in eps_rows:
                if r in is_df.index:
                    vals = [safe_float(is_df.loc[r, c]) for c in cols]
                    vals = [v for v in vals if v is not None]
                    if len(vals) >= 2:
                        return vals, f"income_stmt.{r}"
    except Exception:
        pass

    return [], "NA"


def fetch_balance_sheet_floors(t: yf.Ticker, shares: Optional[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Dict[str, str]]:
    """
    Fetch cash, debt, tangible book/share, net cash/share.
    """
    src = {"cash": "NA", "debt": "NA", "tbv_ps": "NA", "net_cash_ps": "NA"}
    cash = None
    debt = None
    tbv_ps = None
    net_cash_ps = None

    try:
        bs = t.balance_sheet
    except Exception:
        bs = None

    def bs_get(*keys) -> Optional[float]:
        if bs is None or getattr(bs, "empty", True):
            return None
        col = bs.columns[0]  # most recent
        for k in keys:
            if k in bs.index:
                return safe_float(bs.loc[k, col])
        return None

    # Cash
    cash = bs_get(
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Short Term Investments",
        "Cash",
    )
    if cash is not None:
        src["cash"] = "balance_sheet.cash"

    # Debt
    debt = bs_get(
        "Total Debt",
        "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation",
    )
    if debt is not None:
        src["debt"] = "balance_sheet.total_debt"

    # Tangible book value per share (best-effort):
    # tangible equity = stockholders equity - goodwill - intangibles
    equity = bs_get("Total Stockholder Equity", "Stockholders Equity")
    goodwill = bs_get("Goodwill")
    intang = bs_get("Intangible Assets", "Other Intangible Assets")

    if equity is not None and shares not in (None, 0):
        tangible_equity = equity - (goodwill or 0.0) - (intang or 0.0)
        tbv_ps = tangible_equity / shares
        src["tbv_ps"] = "computed(tangible_equity/shares)"

    if shares not in (None, 0) and cash is not None:
        net_cash_ps = (cash - (debt or 0.0)) / shares
        src["net_cash_ps"] = "computed((cash-debt)/shares)"

    return cash, debt, tbv_ps, net_cash_ps, src


def build_snapshot(ticker: str, eps_norm_years: int = 5) -> Snapshot:
    tk = ticker.upper()
    t = yf.Ticker(tk)

    src: Dict[str, str] = {}

    price, psrc = fetch_price(t)
    src["price"] = psrc

    shares, mcap, trailing_eps, src2 = fetch_shares_mcap_eps(t)
    src.update(src2)

    # reconstruct mcap if missing
    if mcap is None and price is not None and shares is not None:
        mcap = price * shares
        src["market_cap"] = "computed(price*shares)"
    else:
        src.setdefault("market_cap", "NA")

    eps_hist, epsh_src = fetch_eps_history_annual_diluted(t, years=eps_norm_years)
    src["eps_hist"] = epsh_src

    cash, debt, tbv_ps, net_cash_ps, src_bs = fetch_balance_sheet_floors(t, shares)
    src.update(src_bs)

    return Snapshot(
        ticker=tk,
        price=price,
        shares=shares,
        market_cap=mcap,
        trailing_eps=trailing_eps,
        eps_hist=eps_hist,
        cash=cash,
        debt=debt,
        tangible_book_ps=tbv_ps,
        net_cash_ps=net_cash_ps,
        src=src,
    )


# ----------------------------
# Recommended valuation + MOS
# ----------------------------
def normalized_eps(snap: Snapshot, fallback_years_min: int = 2) -> Tuple[Optional[float], str]:
    """
    Normalize EPS as median of last N annual diluted EPS, if available.
    Fallback: trailing EPS (TTM).
    """
    if snap.eps_hist and len(snap.eps_hist) >= fallback_years_min:
        return float(np.median(snap.eps_hist)), f"median({len(snap.eps_hist)}y EPS)"
    if snap.trailing_eps is not None:
        return float(snap.trailing_eps), "trailingEps (proxy)"
    return None, "NA"


def eps_cagr(eps_series: List[float]) -> Optional[float]:
    """
    CAGR from oldest to newest (assuming series is most-recent-first from yfinance).
    Returns None if can't compute (non-positive endpoints).
    """
    if not eps_series or len(eps_series) < 2:
        return None
    newest = eps_series[0]
    oldest = eps_series[-1]
    if oldest is None or newest is None:
        return None
    if oldest <= 0 or newest <= 0:
        return None
    n = len(eps_series) - 1
    return (newest / oldest) ** (1.0 / n) - 1.0


def scenario_growth_rates(
    hist_cagr: Optional[float],
    cap_bear: float = 0.10,
    cap_base: float = 0.20,
    cap_bull: float = 0.30,
) -> Dict[str, float]:
    """
    Rule-based growth scenarios:
      bear = min(0.5*hist, cap_bear) but not below 0
      base = min(hist, cap_base) but not below 0
      bull = min(1.5*hist, cap_bull) but not below 0
    If hist is missing: default bear/base/bull = 5% / 10% / 15%.
    """
    if hist_cagr is None:
        return {"bear": 0.05, "base": 0.10, "bull": 0.15}

    h = max(0.0, hist_cagr)
    return {
        "bear": clamp(0.5 * h, 0.0, cap_bear),
        "base": clamp(1.0 * h, 0.0, cap_base),
        "bull": clamp(1.5 * h, 0.0, cap_bull),
    }


def predictability_multiplier(
    eps_series: List[float],
    debt: Optional[float],
    market_cap: Optional[float],
) -> Tuple[float, Dict[str, float]]:
    """
    Simple, rule-based quality haircut (0.60 .. 1.00):
      - EPS volatility: cv = std/|mean|
      - negative EPS fraction
      - leverage proxy: debt / market_cap

    This is deliberately conservative and easy to reason about.
    """
    details: Dict[str, float] = {}
    mult = 1.0

    # EPS stats
    eps = [e for e in eps_series if e is not None]
    if len(eps) >= 3:
        mean = float(np.mean(eps))
        std = float(np.std(eps, ddof=0))
        cv = std / (abs(mean) + 1e-9)
        neg_frac = float(np.mean([1.0 if x < 0 else 0.0 for x in eps]))
        details["eps_cv"] = cv
        details["eps_neg_frac"] = neg_frac

        # Volatility haircut
        if cv > 0.75:
            mult *= 0.75
        elif cv > 0.40:
            mult *= 0.85
        elif cv > 0.25:
            mult *= 0.92

        # Negative EPS haircut
        if neg_frac >= 0.34:
            mult *= 0.80
        elif neg_frac > 0.0:
            mult *= 0.90

    # Leverage proxy
    lev = None
    if debt is not None and market_cap is not None and market_cap > 0:
        lev = float(debt / market_cap)
        details["debt_to_mcap"] = lev
        if lev > 0.60:
            mult *= 0.80
        elif lev > 0.30:
            mult *= 0.90
        elif lev > 0.15:
            mult *= 0.95

    mult = float(clamp(mult, 0.60, 1.00))
    details["multiplier"] = mult
    return mult, details


def two_stage_eps_stream_value(
    eps0: float,
    discount: float,
    stage1_years: int,
    g1: float,
    stage2_years: int,
    g2: float,
) -> Tuple[float, float, float]:
    """
    Two-stage EPS stream PV:
      growth_value  = PV of EPS years 1..stage1_years with g1
      terminal_value= PV of EPS years stage1_years+1 .. stage1_years+stage2_years with g2
      total = growth_value + terminal_value

    NOTE: No perpetuity beyond stage2_years. This matches the structure you used.
    """
    r = discount
    if r <= -0.999:
        raise ValueError("discount rate invalid")

    eps = eps0
    gv = 0.0
    for t in range(1, stage1_years + 1):
        eps *= (1.0 + g1)
        gv += eps / ((1.0 + r) ** t)

    tv = 0.0
    for t in range(stage1_years + 1, stage1_years + stage2_years + 1):
        eps *= (1.0 + g2)
        tv += eps / ((1.0 + r) ** t)

    return float(gv), float(tv), float(gv + tv)


def floor_value_per_share(
    tbv_ps: Optional[float],
    net_cash_ps: Optional[float],
    tbv_haircut: float = 0.80,
    net_cash_haircut: float = 0.80,
) -> float:
    """
    Conservative downside floor anchor:
      floor = max( 0.8*TBV/share, 0.8*NetCash/share, 0 )
    """
    candidates = [0.0]
    if tbv_ps is not None:
        candidates.append(tbv_haircut * tbv_ps)
    if net_cash_ps is not None:
        candidates.append(net_cash_haircut * net_cash_ps)
    return float(max(candidates))


# ----------------------------
# Main
# ----------------------------
def analyze_one(
    ticker: str,
    discount: float,
    stage1_years: int,
    stage2_years: int,
    terminal_rate: float,
    eps_norm_years: int,
    show_sources: bool,
) -> Dict[str, object]:
    snap = build_snapshot(ticker, eps_norm_years=eps_norm_years)

    eps_norm, eps_norm_src = normalized_eps(snap)
    hist_cagr = eps_cagr(snap.eps_hist) if snap.eps_hist else None
    g_scen = scenario_growth_rates(hist_cagr)

    qual_mult, qual_details = predictability_multiplier(
        eps_series=snap.eps_hist if snap.eps_hist else ([] if snap.trailing_eps is None else [snap.trailing_eps]),
        debt=snap.debt,
        market_cap=snap.market_cap,
    )

    floor_v = floor_value_per_share(snap.tangible_book_ps, snap.net_cash_ps)

    out: Dict[str, object] = {
        "ticker": snap.ticker,
        "price": snap.price,
        "eps_norm": eps_norm,
        "eps_norm_src": eps_norm_src,
        "eps_hist_cagr": hist_cagr,
        "quality_mult": qual_mult,
        "floor_value": floor_v,
        "tbv_ps": snap.tangible_book_ps,
        "net_cash_ps": snap.net_cash_ps,
        "shares": snap.shares,
        "market_cap": snap.market_cap,
        "discount": discount,
        "stage1_years": stage1_years,
        "stage2_years": stage2_years,
        "terminal_rate": terminal_rate,
    }

    # scenarios
    for name, g1 in g_scen.items():
        if snap.price is None or eps_norm is None:
            out[f"value_{name}"] = None
            out[f"mos_{name}"] = None
            continue

        gv, tv, v_raw = two_stage_eps_stream_value(
            eps0=float(eps_norm),
            discount=float(discount),
            stage1_years=int(stage1_years),
            g1=float(g1),
            stage2_years=int(stage2_years),
            g2=float(terminal_rate),
        )
        v_hair = v_raw * qual_mult
        v_final = max(v_hair, floor_v)

        out[f"g1_{name}"] = g1
        out[f"growth_value_{name}"] = gv
        out[f"terminal_value_{name}"] = tv
        out[f"value_{name}"] = v_final
        out[f"mos_{name}"] = mos(float(snap.price), float(v_final))

    if show_sources:
        out["sources"] = snap.src
        out["quality_details"] = qual_details

    return out


def main():
    ap = argparse.ArgumentParser(description="Recommended Margin of Safety screener (scenario + normalization + haircuts + floor).")
    ap.add_argument("tickers", nargs="+", help="One or more tickers")
    ap.add_argument("--discount", type=float, default=0.11, help="Discount rate (default 0.11)")
    ap.add_argument("--stage1_years", type=int, default=10, help="Growth stage years (default 10)")
    ap.add_argument("--stage2_years", type=int, default=10, help="Terminal stage years (default 10)")
    ap.add_argument("--terminal_rate", type=float, default=0.04, help="Terminal stage growth rate g2 (default 0.04)")
    ap.add_argument("--eps_norm_years", type=int, default=5, help="Years of annual EPS to use for normalization (default 5)")
    ap.add_argument("--show_sources", action="store_true", help="Include field provenance and quality details")
    ap.add_argument("--csv", default=None, help="Write results to CSV path")
    args = ap.parse_args()

    rows = []
    for tk in args.tickers:
        try:
            rows.append(
                analyze_one(
                    ticker=tk,
                    discount=args.discount,
                    stage1_years=args.stage1_years,
                    stage2_years=args.stage2_years,
                    terminal_rate=args.terminal_rate,
                    eps_norm_years=args.eps_norm_years,
                    show_sources=args.show_sources,
                )
            )
        except Exception as e:
            rows.append({"ticker": tk.upper(), "error": str(e)})

    df = pd.DataFrame(rows)

    # Rank by MOS_bear (higher is better). Missing -> very low.
    if "mos_bear" in df.columns:
        df["_rank_key"] = df["mos_bear"].fillna(-9999.0)
        df = df.sort_values("_rank_key", ascending=False).drop(columns=["_rank_key"])

    # Pretty console view
    show_cols = [
        "ticker", "price",
        "eps_norm", "eps_norm_src",
        "quality_mult", "floor_value",
        "g1_bear", "value_bear", "mos_bear",
        "g1_base", "value_base", "mos_base",
        "g1_bull", "value_bull", "mos_bull",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    pretty = df[show_cols].copy()

    # Format a bit
    for c in ["price", "eps_norm", "quality_mult", "floor_value", "value_bear", "value_base", "value_bull"]:
        if c in pretty.columns:
            pretty[c] = pretty[c].apply(lambda x: None if pd.isna(x) else float(x))

    for c in ["mos_bear", "mos_base", "mos_bull"]:
        if c in pretty.columns:
            pretty[c] = pretty[c].apply(lambda x: pct(None if pd.isna(x) else float(x)))

    for c in ["g1_bear", "g1_base", "g1_bull"]:
        if c in pretty.columns:
            pretty[c] = pretty[c].apply(lambda x: pct(None if pd.isna(x) else float(x)))

    # Print
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 200):
        print("\n" + pretty.to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nWrote: {args.csv}")

    # If show_sources, print per-ticker source blobs (compact)
    if args.show_sources:
        for r in rows:
            if "sources" in r:
                print(f"\n--- {r['ticker']} sources ---")
                for k, v in r["sources"].items():
                    print(f"{k:16s} -> {v}")
            if "quality_details" in r:
                print(f"--- {r['ticker']} quality details ---")
                for k, v in r["quality_details"].items():
                    try:
                        print(f"{k:16s} -> {float(v):.4f}")
                    except Exception:
                        print(f"{k:16s} -> {v}")


if __name__ == "__main__":
    main()
