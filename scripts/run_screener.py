from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from finvizfinance.screener.overview import Overview


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
DOCS_DATA_DIR = ROOT / "docs" / "data"


filters_dict = {
    "Debt/Equity": "Under 1",
    "EPS growthpast 5 years": "Over 15%",
    "Price/Free Cash Flow": "Under 50",
    "Return on Assets": "Positive (>0%)",
    "Return on Equity": "Over +15%",
    "Return on Investment": "Over +15%",
    "52-Week High/Low": "0-10% above Low",
}


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
    except Exception as exc:
        print(f"[cagr] Error with {ticker}: {exc}")
        return None


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
    if price is None or value is None or value == 0:
        return None
    return (value - price) / value


def _fetch_price(ticker: yf.Ticker) -> Tuple[Optional[float], str]:
    try:
        hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
        if hist is not None and not hist.empty:
            close = _safe_float(hist["Close"].dropna().iloc[-1])
            if close is not None:
                return close, "history.Close(last)"
    except Exception:
        pass

    try:
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info:
            price = _safe_float(fast_info.get("lastPrice")) or _safe_float(fast_info.get("last_price"))
            if price is not None:
                return price, "fast_info.lastPrice"
    except Exception:
        pass

    try:
        info = ticker.get_info() or {}
        price = _safe_float(info.get("currentPrice")) or _safe_float(info.get("regularMarketPrice"))
        if price is not None:
            return price, "info.currentPrice/regularMarketPrice"
    except Exception:
        pass

    return None, "NA"


def _fetch_shares_mcap_trailing_eps(ticker: yf.Ticker) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    shares = None
    market_cap = None
    trailing_eps = None

    try:
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info:
            shares = _safe_float(fast_info.get("shares"))
            market_cap = _safe_float(fast_info.get("marketCap")) or _safe_float(fast_info.get("market_cap"))
    except Exception:
        pass

    try:
        info = ticker.get_info() or {}
        if shares is None:
            shares = _safe_float(info.get("sharesOutstanding"))
        if market_cap is None:
            market_cap = _safe_float(info.get("marketCap"))
        trailing_eps = _safe_float(info.get("trailingEps"))
    except Exception:
        pass

    return shares, market_cap, trailing_eps


def _fetch_eps_history_annual(ticker: yf.Ticker, years: int = 5) -> List[float]:
    rows = ["Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS"]
    try:
        income_stmt = ticker.income_stmt
        if income_stmt is None or income_stmt.empty:
            return []
        cols = list(income_stmt.columns)[: max(1, years)]
        for row in rows:
            if row in income_stmt.index:
                values = [_safe_float(income_stmt.loc[row, col]) for col in cols]
                values = [value for value in values if value is not None]
                if len(values) >= 2:
                    return values
    except Exception:
        return []
    return []


def _fetch_balance_sheet_floors(
    ticker: yf.Ticker, shares: Optional[float]
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    cash = debt = tangible_book_value_per_share = net_cash_per_share = None

    try:
        balance_sheet = ticker.balance_sheet
    except Exception:
        balance_sheet = None

    def bs_get(*keys) -> Optional[float]:
        if balance_sheet is None or getattr(balance_sheet, "empty", True):
            return None
        column = balance_sheet.columns[0]
        for key in keys:
            if key in balance_sheet.index:
                return _safe_float(balance_sheet.loc[key, column])
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
    intangible_assets = bs_get("Intangible Assets", "Other Intangible Assets")

    if equity is not None and shares not in (None, 0):
        tangible_equity = equity - (goodwill or 0.0) - (intangible_assets or 0.0)
        tangible_book_value_per_share = tangible_equity / shares

    if shares not in (None, 0) and cash is not None:
        net_cash_per_share = (cash - (debt or 0.0)) / shares

    return cash, debt, tangible_book_value_per_share, net_cash_per_share


def _normalized_eps(eps_hist: List[float], trailing_eps: Optional[float]) -> Tuple[Optional[float], str]:
    if eps_hist and len(eps_hist) >= 2:
        return float(np.median(eps_hist)), f"median({len(eps_hist)}y EPS)"
    if trailing_eps is not None:
        return float(trailing_eps), "trailingEps (proxy)"
    return None, "NA"


def _eps_cagr(eps_hist: List[float]) -> Optional[float]:
    if not eps_hist or len(eps_hist) < 2:
        return None
    newest = eps_hist[0]
    oldest = eps_hist[-1]
    if newest is None or oldest is None or oldest <= 0 or newest <= 0:
        return None
    periods = len(eps_hist) - 1
    return (newest / oldest) ** (1.0 / periods) - 1.0


def _scenario_growth_rates(hist_cagr: Optional[float]) -> Dict[str, float]:
    if hist_cagr is None:
        return {"bear": 0.05, "base": 0.10, "bull": 0.15}
    hist_cagr = max(0.0, float(hist_cagr))
    return {
        "bear": min(0.10, 0.5 * hist_cagr),
        "base": min(0.20, hist_cagr),
        "bull": min(0.30, 1.5 * hist_cagr),
    }


def _predictability_multiplier(eps_hist: List[float], debt: Optional[float], market_cap: Optional[float]) -> float:
    multiplier = 1.0
    eps_values = [value for value in eps_hist if value is not None]
    if len(eps_values) >= 3:
        mean = float(np.mean(eps_values))
        std = float(np.std(eps_values, ddof=0))
        cv = std / (abs(mean) + 1e-9)
        negative_fraction = float(np.mean([1.0 if value < 0 else 0.0 for value in eps_values]))

        if cv > 0.75:
            multiplier *= 0.75
        elif cv > 0.40:
            multiplier *= 0.85
        elif cv > 0.25:
            multiplier *= 0.92

        if negative_fraction >= 0.34:
            multiplier *= 0.80
        elif negative_fraction > 0.0:
            multiplier *= 0.90

    if debt is not None and market_cap is not None and market_cap > 0:
        leverage = float(debt / market_cap)
        if leverage > 0.60:
            multiplier *= 0.80
        elif leverage > 0.30:
            multiplier *= 0.90
        elif leverage > 0.15:
            multiplier *= 0.95

    return float(max(0.60, min(1.00, multiplier)))


def _two_stage_eps_value(
    eps0: float, discount: float, stage1_years: int, g1: float, stage2_years: int, g2: float
) -> float:
    eps = eps0
    present_value = 0.0

    for year in range(1, stage1_years + 1):
        eps *= 1.0 + g1
        present_value += eps / ((1.0 + discount) ** year)

    for year in range(stage1_years + 1, stage1_years + stage2_years + 1):
        eps *= 1.0 + g2
        present_value += eps / ((1.0 + discount) ** year)

    return float(present_value)


def _floor_value(
    tangible_book_value_per_share: Optional[float],
    net_cash_per_share: Optional[float],
    tbv_haircut: float = 0.80,
    net_cash_haircut: float = 0.80,
) -> float:
    candidates = [0.0]
    if tangible_book_value_per_share is not None:
        candidates.append(tbv_haircut * tangible_book_value_per_share)
    if net_cash_per_share is not None:
        candidates.append(net_cash_haircut * net_cash_per_share)
    return float(max(candidates))


def compute_recommended_mos(
    ticker: str,
    discount: float = 0.11,
    stage1_years: int = 10,
    stage2_years: int = 10,
    terminal_rate: float = 0.04,
) -> Dict[str, Optional[float]]:
    symbol = ticker.strip().upper()
    ticker_obj = yf.Ticker(symbol)

    price, _ = _fetch_price(ticker_obj)
    shares, market_cap, trailing_eps = _fetch_shares_mcap_trailing_eps(ticker_obj)
    if market_cap is None and price is not None and shares is not None:
        market_cap = price * shares

    eps_hist = _fetch_eps_history_annual(ticker_obj, years=5)
    _, debt, tbv_ps, net_cash_ps = _fetch_balance_sheet_floors(ticker_obj, shares)

    eps_norm, _ = _normalized_eps(eps_hist, trailing_eps)
    eps_cagr = _eps_cagr(eps_hist)
    growth = _scenario_growth_rates(eps_cagr)
    quality_multiplier = _predictability_multiplier(
        eps_hist if eps_hist else ([trailing_eps] if trailing_eps is not None else []),
        debt,
        market_cap,
    )
    floor_value = _floor_value(tbv_ps, net_cash_ps)

    result: Dict[str, Optional[float]] = {
        "Price": price,
        "EPS_Norm": eps_norm,
        "EPS_CAGR_Proxy": eps_cagr,
        "Quality_Mult": quality_multiplier,
        "Floor_Value": floor_value,
    }

    if price is None or eps_norm is None:
        result.update(
            {
                "Value_Bear": None,
                "Value_Base": None,
                "Value_Bull": None,
                "MOS_Bear_%": None,
                "MOS_Base_%": None,
                "MOS_Bull_%": None,
            }
        )
        return result

    for scenario in ("bear", "base", "bull"):
        value_raw = _two_stage_eps_value(
            eps0=float(eps_norm),
            discount=float(discount),
            stage1_years=int(stage1_years),
            g1=float(growth[scenario]),
            stage2_years=int(stage2_years),
            g2=float(terminal_rate),
        )
        value_haircut = value_raw * float(quality_multiplier)
        value_final = max(value_haircut, float(floor_value))
        mos_value = _mos(float(price), float(value_final))

        result[f"Value_{scenario.capitalize()}"] = value_final
        result[f"MOS_{scenario.capitalize()}_%"] = _pct(mos_value)

    return result


def write_run_metadata(
    *,
    completed_at: datetime,
    full_path: Path,
    filtered_path: Path,
    filtered_today_path: Path,
    full_rows: int,
    filtered_rows: int,
) -> None:
    ny_tz = ZoneInfo("America/New_York")
    metadata = {
        "status": "success",
        "completed_at_utc": completed_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "completed_at_new_york": completed_at.astimezone(ny_tz).isoformat(),
        "completed_at_display": completed_at.astimezone(ny_tz).strftime("%b %d, %Y %I:%M:%S %p %Z"),
        "full_rows": full_rows,
        "filtered_rows": filtered_rows,
        "artifacts": {
            "full": full_path.name,
            "filtered": filtered_path.name,
            "today": filtered_today_path.name,
        },
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for path in (OUTPUTS_DIR / "latest_run.json", DOCS_DATA_DIR / "latest-run.json"):
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    overview = Overview()
    overview.set_filter(filters_dict=filters_dict)
    df = overview.screener_view()

    if df is None or df.empty:
        raise SystemExit("Finviz returned no rows. Check your filters or connectivity.")

    df["10yr_CAGR"] = pd.NA
    print()
    print("[INFO] Calculating Compound Annual Growth Rate (CAGR)")

    for index, row in df.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        print(ticker, "calculating CAGR...")
        cagr = calculate_cagr(ticker, years=10)
        if cagr is not None:
            df.at[index, "10yr_CAGR"] = cagr
        time.sleep(5)

    df_cagr = pd.to_numeric(df["10yr_CAGR"], errors="coerce")
    df_flt = df[df_cagr > 15].copy()

    for col in [
        "MOS_Bear_%",
        "MOS_Base_%",
        "MOS_Bull_%",
        "Value_Bear",
        "Value_Base",
        "Value_Bull",
        "EPS_Norm",
        "EPS_CAGR_Proxy",
        "Quality_Mult",
        "Floor_Value",
    ]:
        df_flt[col] = pd.NA

    print()
    print("[INFO] Computing recommended Margin of Safety (MOS) locally")

    for index, row in df_flt.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        print(ticker, "computing MOS...")
        try:
            result = compute_recommended_mos(
                ticker=ticker,
                discount=0.11,
                stage1_years=10,
                stage2_years=10,
                terminal_rate=0.04,
            )
        except Exception as exc:
            print(f"[mos] Error with {ticker}: {exc}")
            result = {}

        for key, value in result.items():
            if key in df_flt.columns and value is not None:
                df_flt.at[index, key] = value

        time.sleep(3)

    completed_at = datetime.now(timezone.utc)
    stamp = completed_at.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d_%H-%M-%S")
    full_path = OUTPUTS_DIR / f"longarc_full_{stamp}.csv"
    filtered_path = OUTPUTS_DIR / f"longarc_filtered_{stamp}.csv"
    filtered_today_path = OUTPUTS_DIR / "longarc_filtered_TODAY.csv"

    df.to_csv(full_path, index=False)
    df_flt.to_csv(filtered_path, index=False)
    df_flt.to_csv(filtered_today_path, index=False)

    write_run_metadata(
        completed_at=completed_at,
        full_path=full_path,
        filtered_path=filtered_path,
        filtered_today_path=filtered_today_path,
        full_rows=len(df.index),
        filtered_rows=len(df_flt.index),
    )

    print(f"Wrote: {full_path}")
    print(f"Wrote: {filtered_path}")
    print(f"Wrote: {filtered_today_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
