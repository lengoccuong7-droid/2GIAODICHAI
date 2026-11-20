"""Multi-session price/volume flow and Trap Detection 2.0.

All OHLCV flow readings are proxies. They do not identify participants,
aggressor side, institutions, or prove manipulation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED = ("date", "open", "high", "low", "close", "volume")


def _empty(reason: str) -> dict:
    return {
        "status": "INSUFFICIENT_DATA", "flow_label": "CHƯA ĐỦ DỮ LIỆU",
        "flow_score": None, "flow_bias": "UNKNOWN", "trap_risk": "UNKNOWN",
        "trap_label": "CHƯA ĐỦ DỮ LIỆU", "signals": [], "warnings": [],
        "metrics": {"bull_trap_events_5": [], "bear_trap_events_5": [],
                    "breakout_confirmations": [], "trap_events": []},
        "trap_analysis": {"bull_trap": [], "bear_trap": [],
                          "breakout": [], "summary": "Chưa đủ dữ liệu"},
        "cautions": [reason, "OHLCV không xác định danh tính hoặc ý đồ của người giao dịch."],
    }


def _classify_follow_through(x: pd.DataFrame, event_i: int, level: float,
                             event_high: float, event_low: float, kind: str) -> dict:
    """Classify a suspected failed break using up to the next 3 sessions."""
    future = x.iloc[event_i + 1:event_i + 4]
    n = len(future)
    if kind == "BULL_TRAP":
        closes_below = int((future.close < level).sum())
        closes_above = int((future.close > level).sum())
        confirmed = closes_below >= 2 or (n >= 1 and float(future.close.iloc[0]) < event_low)
        invalidated = (n >= 1 and float(future.close.max()) > event_high) or closes_above >= 2
        if confirmed:
            status, label = "CONFIRMED", "BẪY TĂNG ĐƯỢC XÁC NHẬN SAU PHIÊN"
            reason = "Giá tiếp tục đóng cửa dưới vùng breakout trong các phiên sau."
        elif invalidated:
            status, label = "INVALIDATED", "NGHI VẤN BẪY TĂNG BỊ VÔ HIỆU"
            reason = "Giá lấy lại vùng breakout và/hoặc vượt đỉnh phiên nghi vấn."
        elif n < 3:
            status, label = "PENDING", "CHỜ XÁC NHẬN 1–3 PHIÊN"
            reason = "Chưa đủ số phiên theo dõi để kết luận."
        else:
            status, label = "UNCONFIRMED", "CHƯA ĐỦ BẰNG CHỨNG"
            reason = "Diễn biến sau breakout chưa xác nhận rõ bẫy tăng."
    else:
        closes_above = int((future.close > level).sum())
        closes_below = int((future.close < level).sum())
        confirmed = closes_above >= 2 or (n >= 1 and float(future.close.iloc[0]) > event_high)
        invalidated = (n >= 1 and float(future.close.min()) < event_low) or closes_below >= 2
        if confirmed:
            status, label = "CONFIRMED", "BẪY GIẢM ĐƯỢC XÁC NHẬN SAU PHIÊN"
            reason = "Giá lấy lại vùng breakdown trong các phiên tiếp theo."
        elif invalidated:
            status, label = "INVALIDATED", "NGHI VẤN BẪY GIẢM BỊ VÔ HIỆU"
            reason = "Giá tiếp tục nằm dưới vùng breakdown và/hoặc phá đáy phiên nghi vấn."
        elif n < 3:
            status, label = "PENDING", "CHỜ XÁC NHẬN 1–3 PHIÊN"
            reason = "Chưa đủ số phiên theo dõi để kết luận."
        else:
            status, label = "UNCONFIRMED", "CHƯA ĐỦ BẰNG CHỨNG"
            reason = "Diễn biến sau breakdown chưa xác nhận rõ bẫy giảm."
    return {
        "status": status, "label": label, "followup_sessions": n,
        "closes_below_level": int((future.close < level).sum()),
        "closes_above_level": int((future.close > level).sum()),
        "reason": reason,
    }


def _scan_traps(x: pd.DataFrame, scan_sessions: int = 8) -> tuple[list, list, list]:
    """Find false breaks and successful breakouts, then inspect follow-through."""
    bull, bear, breakouts = [], [], []
    start = max(20, len(x) - scan_sessions)
    for i in range(start, len(x)):
        row = x.iloc[i]
        prior = x.iloc[i-20:i]
        if len(prior) < 20:
            continue
        prior_high, prior_low = float(prior.high.max()), float(prior.low.min())
        avg_vol = float(prior.volume.mean())
        rvol = float(row.volume / avg_vol) if avg_vol > 0 else 0.0
        date = row.date.strftime("%Y-%m-%d")

        # Failed upside probe: intraday break above resistance but close back below.
        if float(row.high) > prior_high and float(row.close) < prior_high and rvol >= 1.15:
            follow = _classify_follow_through(x, i, prior_high, float(row.high),
                                              float(row.low), "BULL_TRAP")
            bull.append({
                "type": "BULL_TRAP", "date": date, "level": prior_high,
                "event_high": float(row.high), "event_low": float(row.low),
                "relative_volume": rvol, **follow,
            })

        # Failed downside probe: intraday break below support but close back above.
        if float(row.low) < prior_low and float(row.close) > prior_low and rvol >= 1.15:
            follow = _classify_follow_through(x, i, prior_low, float(row.high),
                                              float(row.low), "BEAR_TRAP")
            bear.append({
                "type": "BEAR_TRAP", "date": date, "level": prior_low,
                "event_high": float(row.high), "event_low": float(row.low),
                "relative_volume": rvol, **follow,
            })

        # Breakout candidate: close above prior resistance with volume expansion.
        if float(row.close) > prior_high and rvol >= 1.10:
            future = x.iloc[i+1:i+4]
            n = len(future)
            held = int((future.close >= prior_high).sum())
            retest = bool(n and ((future.low <= prior_high * 1.01) &
                                 (future.close >= prior_high)).any())
            failed = bool(n >= 2 and (future.close < prior_high).sum() >= 2)
            if failed:
                status, label = "FAILED", "BREAKOUT THẤT BẠI"
                reason = "Có ít nhất 2 phiên đóng cửa trở lại dưới vùng breakout."
            elif n >= 2 and held >= 2 and not (future.close < prior_high * 0.985).any():
                status = "CONFIRMED_RETEST" if retest else "CONFIRMED_HOLD"
                label = "BREAKOUT ĐƯỢC XÁC NHẬN" if retest else "GIỮ ĐƯỢC VÙNG BREAKOUT"
                reason = "Giá duy trì phía trên vùng breakout trong các phiên sau."
            elif n < 3:
                status, label = "PENDING", "BREAKOUT CHỜ XÁC NHẬN"
                reason = "Cần theo dõi thêm tối đa 3 phiên sau điểm vượt cản."
            else:
                status, label = "UNCONFIRMED", "BREAKOUT CHƯA ĐƯỢC XÁC NHẬN"
                reason = "Giá chưa giữ vùng breakout đủ rõ."
            breakouts.append({
                "type": "BREAKOUT", "date": date, "level": prior_high,
                "event_high": float(row.high), "event_low": float(row.low),
                "relative_volume": rvol, "followup_sessions": n,
                "sessions_held_above": held, "retest": retest,
                "status": status, "label": label, "reason": reason,
            })
    return bull, bear, breakouts


def analyze_flow_and_traps(frame: pd.DataFrame, lookback: int = 120) -> dict:
    """Assess multi-session price/volume pressure, traps and breakout follow-through."""
    if frame is None or len(frame) < 30:
        return _empty("Cần tối thiểu 30 phiên OHLCV hợp lệ; khuyến nghị 60–120 phiên.")
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        return _empty("Thiếu cột dữ liệu: " + ", ".join(missing))
    x = frame.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    for c in REQUIRED[1:]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=list(REQUIRED)).sort_values("date").tail(max(lookback, 30)).reset_index(drop=True)
    x = x[(x.close > 0) & (x.volume >= 0) & (x.high >= x.low) &
          (x.low > 0) & (x.high >= x.close) & (x.low <= x.close) &
          (x.high >= x.open) & (x.low <= x.open)]
    if len(x) < 30:
        return _empty("Sau khi loại dữ liệu lỗi còn dưới 30 phiên OHLCV hợp lệ.")

    spread = (x.high - x.low).replace(0, np.nan)
    x["clv"] = (((x.close - x.low) - (x.high - x.close)) / spread).fillna(0).clip(-1, 1)
    x["signed_volume_proxy"] = x.clv * x.volume
    x["obv_proxy"] = (np.sign(x.close.diff()).fillna(0) * x.volume).cumsum()
    x["value_proxy"] = x.close * x.volume
    last = x.iloc[-1]
    vol_avg20 = x.volume.iloc[-21:-1].mean()
    rvol = float(last.volume / vol_avg20) if pd.notna(vol_avg20) and vol_avg20 > 0 else None
    flow_5 = float(x.signed_volume_proxy.tail(5).sum())
    flow_20 = float(x.signed_volume_proxy.tail(20).sum())
    flow_scale = float(x.volume.tail(20).sum()) or 1.0
    flow_ratio = flow_20 / flow_scale
    obv_change_20 = float(x.obv_proxy.iloc[-1] - x.obv_proxy.iloc[-21])
    price_change_20 = float(last.close / x.close.iloc[-21] - 1)
    value_avg20 = float(x.value_proxy.iloc[-20:].mean())
    value_last = float(last.value_proxy)

    signals: list[str] = []
    warnings: list[str] = []
    score = 0
    if flow_ratio >= 0.12 and obv_change_20 > 0:
        score += 2
        signals.append("Dòng tiền OHLCV nghiêng về phía cầu trong 20 phiên: CLV-volume và OBV proxy cùng dương.")
    elif flow_ratio <= -0.12 and obv_change_20 < 0:
        score -= 2
        signals.append("Dòng tiền OHLCV nghiêng về phía cung trong 20 phiên: CLV-volume và OBV proxy cùng âm.")
    else:
        signals.append("Dòng tiền OHLCV 20 phiên chưa có hướng đủ rõ hoặc các proxy chưa đồng thuận.")

    if price_change_20 > 0.03 and obv_change_20 <= 0:
        score -= 1
        warnings.append("Phân kỳ tiêu cực: giá tăng nhưng OBV proxy không xác nhận.")
    elif price_change_20 < -0.03 and obv_change_20 > 0:
        score += 1
        signals.append("Phân kỳ tích cực tiềm năng: giá giảm nhưng OBV proxy cải thiện; chưa phải tín hiệu đảo chiều.")

    recent = x.tail(5)
    weak_high_vol = 0
    for _, row in recent.iterrows():
        prior = x[x.date < row.date].tail(20)
        avg = prior.volume.mean() if len(prior) else np.nan
        rng = row.high - row.low
        clv = ((row.close - row.low) - (row.high - row.close)) / rng if rng > 0 else 0
        if pd.notna(avg) and avg > 0 and row.volume / avg >= 1.4 and clv <= -0.35:
            weak_high_vol += 1
    if weak_high_vol >= 2:
        score -= 2
        warnings.append(f"Có {weak_high_vol} phiên trong 5 phiên gần nhất: khối lượng cao nhưng đóng cửa yếu; cảnh báo cung/phân phối.")

    bull_traps, bear_traps, breakouts = _scan_traps(x)
    confirmed_bull = [e for e in bull_traps if e["status"] == "CONFIRMED"]
    pending_bull = [e for e in bull_traps if e["status"] == "PENDING"]
    confirmed_bear = [e for e in bear_traps if e["status"] == "CONFIRMED"]
    failed_breakouts = [e for e in breakouts if e["status"] == "FAILED"]
    confirmed_breakouts = [e for e in breakouts if e["status"] in ("CONFIRMED_RETEST", "CONFIRMED_HOLD")]

    if confirmed_bull:
        score -= 3
        warnings.append(f"Có {len(confirmed_bull)} nghi vấn Bull Trap được xác nhận bằng diễn biến 1–3 phiên sau.")
    elif pending_bull:
        score -= 1
        warnings.append(f"Có {len(pending_bull)} mẫu hình nghi vấn Bull Trap đang chờ xác nhận; không mua đuổi.")

    if confirmed_bear:
        signals.append(f"Có {len(confirmed_bear)} nghi vấn Bear Trap được xác nhận; cần kiểm tra thêm xu hướng và lực cầu.")
    if failed_breakouts:
        score -= 2
        warnings.append(f"Có {len(failed_breakouts)} breakout thất bại sau khi theo dõi các phiên tiếp theo.")
    if confirmed_breakouts:
        signals.append(f"Có {len(confirmed_breakouts)} breakout giữ được vùng vượt cản sau xác nhận 1–3 phiên.")

    last_range_pct = float((last.high - last.low) / last.close) if last.close else 0
    if rvol is not None and rvol >= 1.7 and last_range_pct <= 0.025 and float(last.clv) >= 0.25:
        signals.append("Có dấu hiệu hấp thụ cung tiềm năng: khối lượng cao, biên độ hẹp và đóng cửa tương đối vững; chưa đủ kết luận tích lũy.")

    bias = "ACCUMULATION_PROXY" if score >= 2 else "DISTRIBUTION_PROXY" if score <= -2 else "MIXED"
    flow_label = {"ACCUMULATION_PROXY": "DÒNG TIỀN NGHIÊNG VỀ CẦU", "DISTRIBUTION_PROXY": "ÁP LỰC CUNG/PHÂN PHỐI", "MIXED": "DÒNG TIỀN CHƯA RÕ"}[bias]
    if confirmed_bull or failed_breakouts:
        trap_risk = "HIGH"
        trap_label = "RỦI RO BẪY TĂNG/BREAKOUT THẤT BẠI CAO"
    elif pending_bull or weak_high_vol >= 2:
        trap_risk = "MODERATE"
        trap_label = "CÓ NGHI VẤN BẪY TĂNG – CHỜ XÁC NHẬN"
    elif confirmed_bear:
        trap_risk = "MODERATE"
        trap_label = "CÓ MẪU HÌNH BẪY GIẢM – CẦN XÁC MINH"
    else:
        trap_risk = "LOW"
        trap_label = "CHƯA PHÁT HIỆN MẪU HÌNH BẪY RÕ"

    trap_events = sorted(bull_traps + bear_traps + breakouts, key=lambda e: e["date"], reverse=True)
    if not signals:
        signals.append("Chưa có tín hiệu dòng tiền nổi bật theo các proxy đang sử dụng.")
    cautions = [
        "Dòng tiền OHLCV là proxy; không xác định danh tính tổ chức/cá nhân hoặc bên chủ động mua/bán.",
        "Khối ngoại, tự doanh và lệnh lớn chỉ được xác nhận khi có dữ liệu nguồn tương ứng, không suy diễn từ OHLCV.",
        "Bẫy giá là đánh giá xác suất theo cấu trúc giá và diễn biến sau sự kiện; không chứng minh thao túng.",
        "Mẫu hình mới xuất hiện có thể ở trạng thái CHỜ XÁC NHẬN; không coi tín hiệu đang chờ là bẫy đã xác nhận.",
        "Giá trị giao dịch xấp xỉ = giá đóng cửa × khối lượng, không thay thế giá trị khớp lệnh thực tế.",
    ]
    return {
        "status": "OK", "flow_label": flow_label, "flow_score": int(score),
        "flow_bias": bias, "trap_risk": trap_risk, "trap_label": trap_label,
        "signals": signals + warnings, "warnings": warnings,
        "trap_analysis": {
            "bull_trap": bull_traps, "bear_trap": bear_traps,
            "breakout": breakouts, "summary": trap_label,
        },
        "metrics": {
            "relative_volume_20": rvol, "flow_ratio_20": flow_ratio,
            "flow_proxy_5": flow_5, "flow_proxy_20": flow_20,
            "obv_proxy_change_20": obv_change_20, "price_change_20_pct": price_change_20 * 100,
            "average_value_proxy_20": value_avg20, "latest_value_proxy": value_last,
            "weak_high_volume_sessions_5": weak_high_vol,
            "bull_trap_events_5": bull_traps, "bear_trap_events_5": bear_traps,
            "breakout_confirmations": breakouts, "trap_events": trap_events,
            "confirmed_bull_traps": len(confirmed_bull),
            "pending_bull_traps": len(pending_bull),
            "failed_breakouts": len(failed_breakouts),
            "confirmed_breakouts": len(confirmed_breakouts),
        }, "cautions": cautions,
    }
