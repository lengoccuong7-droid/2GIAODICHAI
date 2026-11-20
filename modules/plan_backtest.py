"""Replay of the app's OWN trade plan (trigger/stop/target) with VN-market frictions.

Rules: signal at close t -> entry only if price reaches trigger within `valid_sessions`
(no chasing above the entry zone); T+2: cannot sell before entry_day+2; gaps exit at open;
if stop and target are both touched in one bar, stop wins (conservative); +/-limit% price band;
fees/tax/slippage. Educational validation, not a profit guarantee.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .indicators import add_indicators
from .swing_trading_expert import analyze_swing_setup
from .market_flow_trap import analyze_flow_and_traps


def backtest_plan(frame, warmup=60, capital=100_000_000, risk_pct=1.0, max_alloc_pct=20.0,
                  fee_pct=0.15, tax_pct=0.10, slip_pct=0.10, limit_pct=7.0,
                  valid_sessions=3, max_hold=20, risk_style="CÂN BẰNG"):
    x = add_indicators(frame.sort_values("date").copy()).reset_index(drop=True)
    n = len(x)
    if n < warmup + 30:
        return {"error": f"Cần ít nhất {warmup + 30} phiên để kiểm chứng."}
    fee, slip, tax, lim = fee_pct / 100, slip_pct / 100, tax_pct / 100, limit_pct / 100
    trades, signals, i = [], 0, warmup
    while i < n - 1:
        try:
            rep = analyze_swing_setup(x.iloc[: i + 1], capital_vnd=capital, risk_pct=risk_pct,
                                      max_allocation_pct=max_alloc_pct, risk_style=risk_style)
        except Exception:
            i += 1; continue
        if rep.get("status") != "TRADE_PLAN_READY":
            i += 1; continue
        signals += 1
        p = rep["plan"]; trig, stop, tgt, zhi = p["reference_trigger"], p["invalidation"], p["target_1"], p["entry_zone_high"]
        ent = None
        for j in range(i + 1, min(i + 1 + valid_sessions, n)):
            b, pc = x.iloc[j], float(x.close.iloc[j - 1])
            if b.high >= trig and b.open <= zhi:
                f = min(max(float(b.open), trig) * (1 + slip), pc * (1 + lim), float(b.high))
                ent = (j, f); break
        if ent is None:
            i += 1; continue
        j, ef = ent
        xi, xp, reason = None, None, None
        for k in range(j + 2, min(j + max_hold, n - 1) + 1):  # T+2
            b, pc = x.iloc[k], float(x.close.iloc[k - 1])
            if b.open <= stop: xi, xp, reason = k, float(b.open), "STOP_GAP"
            elif b.low <= stop: xi, xp, reason = k, stop, "STOP"
            elif b.open >= tgt: xi, xp, reason = k, float(b.open), "TARGET_GAP"
            elif b.high >= tgt: xi, xp, reason = k, tgt, "TARGET"
            if xi is not None: break
        open_pos = False
        if xi is None:
            xi = min(j + max_hold, n - 1); xp = float(x.close.iloc[xi]); reason = "TIME_EXIT"
            open_pos = (xi == n - 1 and j + max_hold > n - 1)
        pc = float(x.close.iloc[xi - 1]); xp = max(xp * (1 - slip), pc * (1 - lim))
        cost_in = ef * (1 + fee); net_out = xp * (1 - fee - tax)
        ret = net_out / cost_in - 1; risk_unit = max(ef - stop, 1e-9)
        trades.append({"signal_date": str(x.date.iloc[i].date()), "entry_date": str(x.date.iloc[j].date()),
                       "exit_date": str(x.date.iloc[xi].date()), "entry": round(ef, 0), "exit": round(xp, 0),
                       "stop": round(stop, 0), "target": round(tgt, 0), "reason": reason,
                       "return_pct": ret * 100, "r_multiple": (net_out - cost_in) / risk_unit,
                       "setup": rep.get("setup"), "open_at_end": open_pos})
        i = xi + 1
    eq, curve = 1.0, [1.0]
    for t in trades:
        frac = min(max_alloc_pct / 100, (risk_pct / 100) / max((t["entry"] - t["stop"]) / t["entry"], 1e-9))
        eq *= 1 + frac * t["return_pct"] / 100; curve.append(eq)
    c = pd.Series(curve); r = [t["return_pct"] for t in trades]
    wins = sum(v for v in r if v > 0); loss = -sum(v for v in r if v < 0)
    bh = (float(x.close.iloc[-1]) / float(x.close.iloc[warmup]) - 1) * 100
    return {"signals": signals, "trades": trades, "closed_trades": len(trades),
            "win_rate_pct": (sum(v > 0 for v in r) / len(r) * 100) if r else None,
            "avg_r": float(np.mean([t["r_multiple"] for t in trades])) if trades else None,
            "expectancy_pct": float(np.mean(r)) if r else None,
            "profit_factor": (wins / loss) if loss else None,
            "total_return_pct": (eq - 1) * 100, "max_drawdown_pct": float((c / c.cummax() - 1).min() * 100),
            "buy_hold_pct": bh, "sessions_tested": n - warmup,
            "enough_sample": len(trades) >= 30,
            "note": "Mẫu dưới 30 lệnh thì kết quả chưa đủ ý nghĩa thống kê. Quá khứ không đảm bảo tương lai."}


def trap_forward_stats(frame, warmup=60, horizon=5):
    """Forward return after each trap_risk / flow_bias state (all bars, no lookahead)."""
    x = frame.sort_values("date").reset_index(drop=True); rows = []
    for i in range(warmup, len(x) - horizon):
        ft = analyze_flow_and_traps(x.iloc[: i + 1])
        if ft.get("status") != "OK": continue
        rows.append({"trap_risk": ft["trap_risk"], "flow_bias": ft["flow_bias"],
                     "fwd": (float(x.close.iloc[i + horizon]) / float(x.close.iloc[i]) - 1) * 100})
    if not rows: return {}
    d = pd.DataFrame(rows)
    def agg(col):
        return {k: {"count": int(len(g)), "avg_fwd_pct": float(g.fwd.mean()), "pct_positive": float((g.fwd > 0).mean() * 100)}
                for k, g in d.groupby(col)}
    return {"horizon_sessions": horizon, "by_trap_risk": agg("trap_risk"), "by_flow_bias": agg("flow_bias"),
            "baseline_avg_pct": float(d.fwd.mean())}
