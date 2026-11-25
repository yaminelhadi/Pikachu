import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import yfinance as yf

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
STATE_FILE = "pika_bot_state.json"
CASH_BALANCE = float(os.getenv("IBKR_CASH_BALANCE", "20000.0"))

RSI_PERIOD = 2
TREND_SMA = 150
FAST_SMA = 50
EXIT_SMA = 5
MOMENTUM_LOOKBACK = 63
TRAILING_STOP_PCT = 0.045
MAX_HOLD_DAYS = 8

ALLOC_HIGH = 0.70
ALLOC_MED = 0.50
ALLOC_LOW = 0.30

SHARIAH_50_UNIVERSE = [
    "NVDA", "AAPL", "MSFT", "AVGO", "TSM", "ASML", "AMD", "QCOM", "TXN", "MU",
    "ARM", "MRVL", "ADBE", "CRM", "NOW", "INTU", "PANW", "CRWD", "FTNT", "CSCO",
    "ANET", "ACN",
    "LLY", "JNJ", "MRK", "ABBV", "ABT", "MDT", "ISRG", "GILD", "REGN", "SYK",
    "BMY",
    "COST", "PG", "PEP", "KO", "MNST", "CL", "HSY", "KVUE",
    "XOM", "CVX", "LIN", "NUE", "SLB", "GRMN", "FAST", "CPRT", "ODFL",
]


def send_discord_alert(content: str):
    print("\n" + content + "\n")
    if not DISCORD_WEBHOOK_URL:
        print("[Info] DISCORD_WEBHOOK_URL not set; printed alert locally only.")
        return
    chunks = [content[i:i + 1900] for i in range(0, len(content), 1900)]
    for chunk in chunks:
        resp = requests.post(DISCORD_WEBHOOK_URL, json={"content": chunk}, timeout=10)
        resp.raise_for_status()


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"positions": {}, "cash": CASH_BALANCE}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def compute_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(100.0)


def score_confidence(price: float, sma50: float, sma150: float, rsi2: float, mom3m: float, bull_regime: bool) -> float:
    strong_trend = (price > sma150) and (sma50 > sma150)
    deep_dip = rsi2 < 8.0
    strong_mom = mom3m > 0.12

    if bull_regime and strong_trend and (deep_dip or strong_mom):
        return ALLOC_HIGH
    if strong_trend and (rsi2 < 15.0 or mom3m > 0.05):
        return ALLOC_MED
    return ALLOC_LOW


def run_scanner():
    state = load_state()
    positions = state["positions"]
    cash = float(state.get("cash", CASH_BALANCE))

    tickers = sorted(set(SHARIAH_50_UNIVERSE + ["SPUS"]))
    raw = yf.download(tickers, period="1y", interval="1d", auto_adjust=True, progress=False)
    close_df = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

    spus_series = close_df["SPUS"].dropna() if "SPUS" in close_df.columns else pd.Series(dtype=float)
    bull_regime = True
    if len(spus_series) >= TREND_SMA:
        bull_regime = float(spus_series.iloc[-1]) > float(spus_series.rolling(TREND_SMA).mean().iloc[-1])

    today_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"**50-Stock Daily Scanner** — `{today_str}`",
        f"Regime (`SPUS > 150-SMA`): **{'BULL / FAVORABLE' if bull_regime else 'CAUTIOUS / DEFENSIVE'}** | Cash: **${cash:,.2f}**",
    ]

    for sym in list(positions.keys()):
        if sym not in close_df.columns:
            continue
        s = close_df[sym].dropna()
        if len(s) < EXIT_SMA + 2:
            continue

        price = float(s.iloc[-1])
        sma5 = float(s.rolling(EXIT_SMA).mean().iloc[-1])
        rsi2 = float(compute_rsi(s, RSI_PERIOD).iloc[-1])

        pos = positions[sym]
        pos["highest_close"] = max(float(pos["highest_close"]), price)
        trail_stop = pos["highest_close"] * (1.0 - TRAILING_STOP_PCT)
        days_held = (datetime.strptime(today_str, "%Y-%m-%d") - datetime.strptime(pos["entry_date"], "%Y-%m-%d")).days

        exit_reason = None
        if price <= trail_stop:
            exit_reason = f"Trailing Stop hit (`${price:.2f} <= ${trail_stop:.2f}`)"
        elif price > sma5 and rsi2 > 50:
            exit_reason = f"Mean Reversion bounce (`${price:.2f} > 5-SMA ${sma5:.2f}`, RSI-2 `{rsi2:.1f}`)"
        elif days_held >= MAX_HOLD_DAYS:
            exit_reason = f"Max hold limit (`{days_held}` days)"

        if exit_reason:
            proceeds = pos["shares"] * price
            pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100.0
            cash += proceeds
            lines.append(
                f"🟥 **SELL {sym}** — `{pos['shares']}` shares @ ~`${price:.2f}` (`${proceeds:,.2f}`)\n"
                f"   • Reason: {exit_reason} | Est. Return: `{pnl_pct:+.2f}%`"
            )
            del positions[sym]
        else:
            lines.append(
                f"🟨 **HOLD {sym}** — `{pos['shares']}` shares @ `${price:.2f}` | Trail Stop: `${trail_stop:.2f}` | 5-SMA Exit: `${sma5:.2f}`"
            )

    candidates = []
    if cash > 100:
        for sym in SHARIAH_50_UNIVERSE:
            if sym in positions or sym not in close_df.columns:
                continue
            s = close_df[sym].dropna()
            if len(s) < max(TREND_SMA, MOMENTUM_LOOKBACK) + 5:
                continue

            price = float(s.iloc[-1])
            sma50 = float(s.rolling(FAST_SMA).mean().iloc[-1])
            sma150 = float(s.rolling(TREND_SMA).mean().iloc[-1])
            sma5 = float(s.rolling(EXIT_SMA).mean().iloc[-1])
            rsi2 = float(compute_rsi(s, RSI_PERIOD).iloc[-1])
            mom3m = (price / float(s.iloc[-MOMENTUM_LOOKBACK])) - 1.0

            in_uptrend = price > sma150
            oversold_dip = (rsi2 < 15.0) and (price < sma5)
            allow_weak_regime = bull_regime or (rsi2 < 6.0)

            if in_uptrend and oversold_dip and allow_weak_regime:
                alloc = score_confidence(price, sma50, sma150, rsi2, mom3m, bull_regime)
                priority = (15.0 - rsi2) + (mom3m * 100.0)
                candidates.append((priority, sym, price, rsi2, mom3m, alloc))

        candidates.sort(key=lambda x: x[0], reverse=True)

        for _, sym, price, rsi2, mom3m, alloc in candidates[:2]:
            if cash <= 50:
                break
            dollars_to_use = cash * alloc
            shares = int(dollars_to_use // price)
            if shares < 1:
                continue

            cost = shares * price
            cash -= cost
            init_stop = price * (1.0 - TRAILING_STOP_PCT)
            positions[sym] = {
                "entry_price": round(price, 4),
                "highest_close": round(price, 4),
                "entry_date": today_str,
                "shares": shares,
                "alloc_pct": int(alloc * 100),
            }
            lines.append(
                f"🟩 **BUY {sym}** — `{shares}` shares @ Limit ~`${price:.2f}` (**`${cost:,.2f}`** | `{int(alloc*100)}%` of cash)\n"
                f"   • RSI(2): `{rsi2:.1f}` | 3M Mom: `{mom3m*100:+.1f}%` | Trail Stop: `${init_stop:.2f}`"
            )

    if len(lines) == 2:
        lines.append("⬜ **NO NEW TRADES TODAY** — Waiting for clean uptrend pullbacks; staying in cash.")

    state["cash"] = round(cash, 2)
    save_state(state)
    send_discord_alert("\n".join(lines))


if __name__ == "__main__":
    run_scanner()
