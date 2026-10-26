"""
=====================================================================
 NEW YORK FIBONACCI PULLBACK STRATEGY - Python Backtester
=====================================================================

Runs the exact same rule set as ny_fib_pullback_bot.py (the live MT5
bot) and ny_fib_pullback_strategy.pine (the TradingView version) over
historical CSV data, so you can validate performance before ever
connecting a broker.

It reuses the core decision-logic functions directly from
ny_fib_pullback_bot.py (structure detection, leg-finding, Fibonacci,
BOS) so the three versions can't quietly drift apart. Only the parts
that differ between "live polling" and "replaying history" are
reimplemented here (trade simulation, and the BOS/candle confirmation
checks, which need slightly different bar-indexing semantics when
replaying closed historical bars vs. watching a still-forming live
bar - see the comments on bt_check_bos / bt_check_strong_candle below).

--------------------------------------------------------------------
USAGE
--------------------------------------------------------------------
1) Export 5-minute XAUUSD history from MT5 (or any broker/data
   source) to CSV with columns: time, open, high, low, close
   (volume optional). In MT5: right-click a chart -> "Export chart
   history" or use the History Center.

2) Run:
       python ny_fib_pullback_backtest.py --csv your_data.csv

   Or, to sanity-check the pipeline itself without real data:
       python ny_fib_pullback_backtest.py --demo

--------------------------------------------------------------------
ASSUMPTIONS / LIMITATIONS (read before trusting the numbers)
--------------------------------------------------------------------
- Timestamps in the CSV are assumed to already be UTC unless you pass
  --input-tz to say otherwise (e.g. --input-tz "Etc/GMT-2" for a
  common MT5 broker server offset). Getting this wrong shifts the NY
  session window and will silently produce wrong results.
- H4 bars are derived by resampling your M5 data (open=first,
  high=max, low=min, close=last), not pulled independently - make
  sure your M5 data has no large gaps.
- Trade simulation checks stop-loss before take-profit whenever a
  single bar's range could have hit both (a standard conservative
  assumption in backtesting, since 5-minute bars don't tell us the
  intrabar path) - this may understate performance slightly, never
  overstate it.
- Entry is simulated at the confirmation candle's close, with no
  slippage or spread modeled. Real fills will be worse, not better.
- This is a backtest of the RULES, not a guarantee those rules will
  keep working. Past performance on historical data does not predict
  future results.
=====================================================================
"""

import argparse
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

import ny_fib_pullback_bot as bot  # reuses structure/leg/fib logic


# =====================================================================
# BACKTEST-SPECIFIC CONFIRMATION CHECKS
# =====================================================================
# bot.check_bos / bot.check_strong_candle assume the LAST row of the
# dataframe passed in is a still-forming live bar and exclude it.
# When replaying history, the slice we pass ends exactly at the bar
# we want to evaluate (it's already closed) - so here we evaluate
# using the last row directly instead of excluding it.

def bt_check_bos(df_m5: pd.DataFrame, bias: str, lookback: int) -> bool:
    if len(df_m5) < lookback + 1:
        return False
    last = df_m5.iloc[-1]
    prior = df_m5.iloc[-(lookback + 1):-1]
    if len(prior) < lookback:
        return False
    if bias == "bullish":
        return bool(last["close"] > prior["high"].max())
    else:
        return bool(last["close"] < prior["low"].min())


def bt_check_strong_candle(df_m5: pd.DataFrame, bias: str, body_ratio: float) -> bool:
    if len(df_m5) < 1:
        return False
    c = df_m5.iloc[-1]
    rng = c["high"] - c["low"]
    if rng <= 0:
        return False
    body = abs(c["close"] - c["open"])
    if (body / rng) < body_ratio:
        return False
    return bool(c["close"] > c["open"]) if bias == "bullish" else bool(c["close"] < c["open"])


def bt_confirmation_candle(df_m5: pd.DataFrame) -> pd.Series:
    return df_m5.iloc[-1]


# =====================================================================
# DATA LOADING
# =====================================================================

def load_m5_csv(path: str, input_tz: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    time_col = None
    for candidate in ("time", "datetime", "date"):
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        raise ValueError(f"Could not find a time/date column in {path}. Columns found: {list(df.columns)}")

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Columns found: {list(df.columns)}")

    df["time"] = pd.to_datetime(df[time_col])
    if df["time"].dt.tz is None:
        if input_tz:
            df["time"] = df["time"].dt.tz_localize(input_tz).dt.tz_convert("UTC")
        else:
            df["time"] = df["time"].dt.tz_localize("UTC")
    else:
        df["time"] = df["time"].dt.tz_convert("UTC")

    df = df[["time", "open", "high", "low", "close"]].dropna()
    df = df.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)
    if len(df) < 500:
        raise ValueError(f"Only {len(df)} M5 bars loaded - need substantially more history for a meaningful backtest.")
    return df


def derive_h4(df_m5: pd.DataFrame) -> pd.DataFrame:
    s = df_m5.set_index("time")
    h4 = s.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    h4 = h4.dropna().reset_index()
    return h4


def generate_demo_data(n_days: int = 120, seed: int = 7) -> pd.DataFrame:
    """Synthetic XAUUSD-like M5 data for pipeline testing only - NOT real
    market data, do not draw any performance conclusions from it."""
    rng = np.random.default_rng(seed)
    bars_per_day = 288  # 24h * 12 (5-min bars)
    n = n_days * bars_per_day
    start = pd.Timestamp("2025-01-06 00:00:00", tz="UTC")
    times = pd.date_range(start, periods=n, freq="5min")

    # regime-switching drift to create trending legs a fib-pullback
    # strategy could plausibly interact with
    drift = np.zeros(n)
    regime_len = bars_per_day * 3
    for start_idx in range(0, n, regime_len):
        d = rng.choice([-1, 1]) * rng.uniform(0.01, 0.04)
        end_idx = min(start_idx + regime_len, n)
        drift[start_idx:end_idx] = d

    noise = rng.normal(0, 1.2, n)
    price = 2000 + np.cumsum(drift + noise)
    price = np.maximum(price, 500)

    highs = price + np.abs(rng.normal(0.6, 0.4, n))
    lows = price - np.abs(rng.normal(0.6, 0.4, n))
    opens = np.roll(price, 1)
    opens[0] = price[0]
    closes = price

    df = pd.DataFrame({
        "time": times, "open": opens, "high": highs, "low": lows, "close": closes,
    })
    return df


# =====================================================================
# TRADE SIMULATION
# =====================================================================

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    sl_initial: float
    tp: float
    risk_dist: float
    units: float
    pnl: float
    r_multiple: float
    exit_reason: str
    equity_after: float


def simulate_trade(df_m5: pd.DataFrame, entry_idx: int, direction: str, entry_price: float,
                    sl: float, tp: float, risk_dist: float, be_at_r: float,
                    units: float) -> Trade:
    current_sl = sl
    moved_be = False
    n = len(df_m5)

    for j in range(entry_idx + 1, n):
        bar = df_m5.iloc[j]

        if direction == "long":
            if not moved_be:
                gain = bar["high"] - entry_price
                if gain >= risk_dist * be_at_r:
                    current_sl = entry_price
                    moved_be = True
            # conservative: check stop before target if both are in range this bar
            if bar["low"] <= current_sl:
                exit_price = current_sl
                reason = "breakeven" if moved_be and abs(current_sl - entry_price) < 1e-9 else "stop_loss"
                pnl = units * (exit_price - entry_price)
                r_mult = (exit_price - entry_price) / risk_dist if risk_dist > 0 else np.nan
                return Trade(df_m5["time"].iloc[entry_idx], bar["time"], direction, entry_price,
                             exit_price, sl, tp, risk_dist, units, pnl, r_mult, reason, np.nan)
            if bar["high"] >= tp:
                exit_price = tp
                pnl = units * (exit_price - entry_price)
                r_mult = (exit_price - entry_price) / risk_dist if risk_dist > 0 else np.nan
                return Trade(df_m5["time"].iloc[entry_idx], bar["time"], direction, entry_price,
                             exit_price, sl, tp, risk_dist, units, pnl, r_mult, "take_profit", np.nan)
        else:  # short
            if not moved_be:
                gain = entry_price - bar["low"]
                if gain >= risk_dist * be_at_r:
                    current_sl = entry_price
                    moved_be = True
            if bar["high"] >= current_sl:
                exit_price = current_sl
                reason = "breakeven" if moved_be and abs(current_sl - entry_price) < 1e-9 else "stop_loss"
                pnl = units * (entry_price - exit_price)
                r_mult = (entry_price - exit_price) / risk_dist if risk_dist > 0 else np.nan
                return Trade(df_m5["time"].iloc[entry_idx], bar["time"], direction, entry_price,
                             exit_price, sl, tp, risk_dist, units, pnl, r_mult, reason, np.nan)
            if bar["low"] <= tp:
                exit_price = tp
                pnl = units * (entry_price - exit_price)
                r_mult = (entry_price - exit_price) / risk_dist if risk_dist > 0 else np.nan
                return Trade(df_m5["time"].iloc[entry_idx], bar["time"], direction, entry_price,
                             exit_price, sl, tp, risk_dist, units, pnl, r_mult, "take_profit", np.nan)

    # ran out of data before hitting SL or TP - close at last known price
    last = df_m5.iloc[-1]
    exit_price = last["close"]
    if direction == "long":
        pnl = units * (exit_price - entry_price)
        r_mult = (exit_price - entry_price) / risk_dist if risk_dist > 0 else np.nan
    else:
        pnl = units * (entry_price - exit_price)
        r_mult = (entry_price - exit_price) / risk_dist if risk_dist > 0 else np.nan
    return Trade(df_m5["time"].iloc[entry_idx], last["time"], direction, entry_price,
                 exit_price, sl, tp, risk_dist, units, pnl, r_mult, "end_of_data", np.nan)


# =====================================================================
# MAIN BACKTEST LOOP
# =====================================================================

def run_backtest(df_m5_full: pd.DataFrame, df_h4_full: pd.DataFrame,
                  initial_capital: float = 10000.0,
                  risk_percent: float = bot.Config.RISK_PERCENT,
                  rr_target: float = bot.Config.RR_TARGET,
                  be_at_r: float = bot.Config.MOVE_TO_BE_AT_R,
                  m5_window: int = 400, h4_window: int = 60) -> List[Trade]:

    equity = initial_capital
    trades: List[Trade] = []
    state: Optional[bot.DayState] = None
    n = len(df_m5_full)
    i = max(50, m5_window // 4)  # small warmup so ATR/structure have data

    while i < n:
        now_utc = df_m5_full["time"].iloc[i]
        session_open_utc, entry_end_utc, session_close_utc = bot.get_session_bounds_utc(now_utc)
        date_key = bot.ny_now(now_utc).strftime("%Y-%m-%d")

        if state is None or state.date_key != date_key:
            state = bot.DayState(date_key=date_key)
            state.session_open_time = session_open_utc
            state.entry_window_end = entry_end_utc

        if state.trade_taken or state.day_skipped:
            i += 1
            continue

        if now_utc < session_open_utc or now_utc > session_close_utc:
            i += 1
            continue

        df_m5_win = df_m5_full.iloc[max(0, i + 1 - m5_window):i + 1].reset_index(drop=True)
        df_h4_win = df_h4_full[df_h4_full["time"] <= now_utc].tail(h4_window).reset_index(drop=True)

        bias = bot.detect_htf_bias(df_h4_win)
        if bias is None:
            state.day_skipped = True
            i += 1
            continue
        state.bias = bias

        atr_series = bot.compute_atr(df_m5_win, bot.Config.ATR_PERIOD)
        if not bot.volatility_ok(df_m5_win, atr_series):
            state.day_skipped = True
            i += 1
            continue
        current_atr = atr_series.iloc[-1]

        if not bot.is_within_entry_window(now_utc, state):
            if state.leg is None:
                state.day_skipped = True
            i += 1
            continue

        df_session = df_m5_win[df_m5_win["time"] >= state.session_open_time].reset_index(drop=True)

        if state.leg is None:
            leg = bot.find_first_leg(df_session, bias, current_atr)
            if leg is None:
                i += 1
                continue
            state.leg = leg
            state.fib = bot.compute_fib(leg, bias)

        current_price = float(df_m5_win["close"].iloc[-1])

        if bot.price_invalidated_leg(current_price, state.fib, bias):
            state.day_skipped = True
            i += 1
            continue

        if not state.zone_tagged:
            if bot.price_in_golden_pocket(current_price, state.fib, bias):
                state.zone_tagged = True
                state.zone_tagged_bar_index = i
            i += 1
            continue

        bars_waited = i - state.zone_tagged_bar_index
        if bars_waited > bot.Config.CONFIRMATION_MAX_BARS:
            state.day_skipped = True
            i += 1
            continue

        confirmed = (bt_check_bos(df_m5_win, bias, bot.Config.BOS_LOOKBACK) or
                     bt_check_strong_candle(df_m5_win, bias, bot.Config.STRONG_CANDLE_BODY_RATIO))
        if not confirmed:
            i += 1
            continue

        # --- stop loss ---
        conf_candle = bt_confirmation_candle(df_m5_win)
        fib_618 = state.fib[0.618]
        if bias == "bullish":
            sl = min(fib_618, conf_candle["low"])
        else:
            sl = max(fib_618, conf_candle["high"])

        entry_price = current_price
        risk_distance = abs(entry_price - sl)
        if risk_distance <= 0:
            i += 1
            continue

        tp = entry_price + risk_distance * rr_target if bias == "bullish" else entry_price - risk_distance * rr_target

        risk_amount = equity * (risk_percent / 100.0)
        units = risk_amount / risk_distance
        if units <= 0 or not math.isfinite(units):
            i += 1
            continue

        direction = "long" if bias == "bullish" else "short"
        trade = simulate_trade(df_m5_full, i, direction, entry_price, sl, tp, risk_distance, be_at_r, units)
        equity += trade.pnl
        trade.equity_after = equity
        trades.append(trade)
        state.trade_taken = True

        # jump the outer loop to the exit bar so we don't evaluate new
        # setups while "in" this trade
        exit_i = df_m5_full.index[df_m5_full["time"] == trade.exit_time]
        i = int(exit_i[0]) + 1 if len(exit_i) else i + 1

    return trades


# =====================================================================
# STATISTICS
# =====================================================================

def compute_stats(trades: List[Trade], initial_capital: float) -> Dict:
    if not trades:
        return {"total_trades": 0}

    df = pd.DataFrame([t.__dict__ for t in trades])
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    win_rate = len(wins) / len(df) * 100
    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_r = df["r_multiple"].mean()
    expectancy_dollars = df["pnl"].mean()
    expectancy_r = df["r_multiple"].mean()
    holding_time = (df["exit_time"] - df["entry_time"]).mean()

    # equity curve -> daily returns -> Sharpe
    equity_series = df.set_index("exit_time")["equity_after"]
    equity_series = pd.concat([pd.Series([initial_capital], index=[df["entry_time"].iloc[0]]), equity_series])
    daily_equity = equity_series.resample("1D").last().ffill()
    daily_returns = daily_equity.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * math.sqrt(252) if daily_returns.std() > 0 else float("nan")

    running_max = daily_equity.cummax()
    drawdown = (daily_equity - running_max) / running_max
    max_dd_pct = drawdown.min() * 100

    final_equity = df["equity_after"].iloc[-1]
    total_return_pct = (final_equity - initial_capital) / initial_capital * 100

    return {
        "total_trades": len(df),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3) if math.isfinite(profit_factor) else float("inf"),
        "avg_r_multiple": round(avg_r, 3),
        "expectancy_per_trade_usd": round(expectancy_dollars, 2),
        "expectancy_per_trade_r": round(expectancy_r, 3),
        "avg_holding_time": str(holding_time),
        "sharpe_ratio_annualized": round(sharpe, 3) if math.isfinite(sharpe) else float("nan"),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "final_equity_usd": round(final_equity, 2),
        "total_return_pct": round(total_return_pct, 2),
    }


def day_of_week_breakdown(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([t.__dict__ for t in trades])
    df["ny_entry_time"] = df["entry_time"].apply(bot.ny_now)
    df["day_name"] = df["ny_entry_time"].dt.day_name()

    rows = []
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        sub = df[df["day_name"] == day]
        if len(sub) == 0:
            continue
        wins = sub[sub["pnl"] > 0]
        rows.append({
            "day": day,
            "trades": len(sub),
            "win_rate_pct": round(len(wins) / len(sub) * 100, 1),
            "avg_r": round(sub["r_multiple"].mean(), 3),
            "total_pnl_usd": round(sub["pnl"].sum(), 2),
        })
    return pd.DataFrame(rows)


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Backtest the NY Fibonacci Pullback strategy on historical data.")
    parser.add_argument("--csv", type=str, help="Path to M5 OHLC CSV (time,open,high,low,close)")
    parser.add_argument("--input-tz", type=str, default=None,
                         help="Timezone of the CSV timestamps if not already UTC, e.g. 'Etc/GMT-2'")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--risk-pct", type=float, default=bot.Config.RISK_PERCENT)
    parser.add_argument("--rr", type=float, default=bot.Config.RR_TARGET)
    parser.add_argument("--demo", action="store_true", help="Run on synthetic demo data instead of a CSV")
    args = parser.parse_args()

    if args.demo:
        print("Running on SYNTHETIC demo data - for pipeline testing only, not real market behavior.\n")
        df_m5 = generate_demo_data()
    elif args.csv:
        df_m5 = load_m5_csv(args.csv, input_tz=args.input_tz)
    else:
        parser.error("Provide --csv path/to/data.csv or use --demo")
        return

    df_h4 = derive_h4(df_m5)
    print(f"Loaded {len(df_m5)} M5 bars ({df_m5['time'].iloc[0]} -> {df_m5['time'].iloc[-1]})")
    print(f"Derived {len(df_h4)} H4 bars\n")

    trades = run_backtest(df_m5, df_h4, initial_capital=args.capital,
                           risk_percent=args.risk_pct, rr_target=args.rr)

    stats = compute_stats(trades, args.capital)
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    for k, v in stats.items():
        print(f"{k:30s}: {v}")

    print("\n" + "=" * 60)
    print("DAY-OF-WEEK BREAKDOWN")
    print("=" * 60)
    dow = day_of_week_breakdown(trades)
    if len(dow):
        print(dow.to_string(index=False))
    else:
        print("No trades taken.")

    if trades:
        trades_df = pd.DataFrame([t.__dict__ for t in trades])
        out_path = "backtest_trades.csv"
        trades_df.to_csv(out_path, index=False)
        print(f"\nFull trade log written to {out_path}")


if __name__ == "__main__":
    main()