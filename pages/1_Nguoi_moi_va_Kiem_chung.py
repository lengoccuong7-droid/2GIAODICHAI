import pandas as pd, streamlit as st
from modules.data_loader import load_market_file
from modules.price_units import normalize_price_unit
from modules.indicators import add_indicators
from modules.swing_trading_expert import analyze_swing_setup
from modules.beginner_summary import build_beginner_summary, GLOSSARY
from modules.plan_backtest import backtest_plan, trap_forward_stats
from modules.levels import holder_guide, watch_levels_text, volume_narrative
from modules.charts import price_volume_figure

st.set_page_config(page_title="Cho người mới", layout="centered")
st.title("Tóm tắt cho người mới + Kiểm chứng")
up = st.file_uploader("File giá OHLCV (CSV/XLSX)", type=["csv", "xlsx", "xls"])
c1, c2 = st.columns(2)
unit = c1.selectbox("Đơn vị giá trong file", ["Đồng/cổ phiếu", "Nghìn đồng/cổ phiếu"])
cap = c2.number_input("Vốn (VND)", 1_000_000, 100_000_000_000, 100_000_000, 1_000_000)
risk = st.slider("Mức lỗ tối đa mỗi lệnh (% vốn)", 0.25, 2.0, 1.0, 0.25)
if not up: st.info("Tải file để bắt đầu. Có thể thử với sample_data/DEMO_SYNTHETIC_OHLCV.csv (dữ liệu giả)."); st.stop()
try:
    df = normalize_price_unit(load_market_file(up), 1000.0 if unit.startswith("Nghìn") else 1.0)
except Exception as e:
    st.error(f"Không đọc được file: {e}"); st.stop()
sym = st.selectbox("Mã", sorted(df.symbol.astype(str).unique()))
f = df[df.symbol.astype(str) == sym].sort_values("date")
holding = st.toggle("Tôi đang có hàng mã này")
pos = {}
if holding:
    p1, p2, p3 = st.columns(3)
    qty = p1.number_input("Số lượng cp", 0, 10_000_000, 0, 100)
    entry = p2.number_input("Giá vốn (đ)", 0.0, 10_000_000.0, 0.0, 50.0)
    stop_in = p3.number_input("Cắt lỗ đã đặt (đ, 0=chưa)", 0.0, 10_000_000.0, 0.0, 50.0)
    max_loss = st.number_input("Số tiền tối đa chấp nhận mất trên vị thế này (đ, 0=bỏ qua)", 0, 10_000_000_000, 0, 100_000)
    if qty > 0 and entry > 0: pos = {"quantity": int(qty), "entry_price": float(entry), "stop_price": float(stop_in) if stop_in > 0 else None}
rep = analyze_swing_setup(add_indicators(f.copy()), capital_vnd=cap, risk_pct=risk, position=pos)
t1, tc, t2 = st.tabs(["Hôm nay nên làm gì?", "Biểu đồ", "Kiểm chứng chiến lược"])
with t1:
    s = build_beginner_summary(rep)
    {"ok": st.success, "wait": st.warning, "hold": st.info, "stop": st.error}[s["level"]](f"**{s['symbol']}: {s['headline']}**")
    st.write(s["action"])
    for l in s["plan_lines"]: st.markdown(f"- {l}")
    if s["why"]:
        st.markdown("**Lý do chính**"); [st.markdown(f"- {w}") for w in s["why"]]
    if s["blockers"]:
        with st.expander("Vì sao chưa mua?"): [st.markdown(f"- {b}") for b in s["blockers"]]
    if holding and pos:
        g = holder_guide(f, pos["entry_price"], pos["quantity"], pos.get("stop_price"), max_loss or None)
        st.markdown(f"**Vị thế của bạn: {g['status']}**")
        st.write(f"Lãi/lỗ: {g['pnl_pct']:+.1f}% ({g['pnl_vnd']:+,.0f}đ). Cắt lỗ tham chiếu: {g['stop_reference']:,.0f}đ "
                 f"(nếu chạm: {g['loss_if_stop_pct']:+.1f}%, {-g['loss_if_stop_vnd']:+,.0f}đ). Stop dời theo: {g['trailing_reference']:,.0f}đ. Kháng cự: {g['resistance']:,.0f}đ.")
        for n_ in g["notes"]: st.markdown(f"- {n_}")
    elif holding: st.info("Nhập số lượng và giá vốn để nhận hướng dẫn cho vị thế.")
    st.markdown("**Khối lượng & mốc theo dõi**")
    st.markdown(f"- {volume_narrative(f)['text']}")
    for m in watch_levels_text(f): st.markdown(f"- {m}")
    for w in s["warnings"]: st.caption("⚠️ " + w)
    with st.expander("Giải thích thuật ngữ"): [st.markdown(f"**{k}**: {v}") for k, v in GLOSSARY.items()]
with tc:
    st.plotly_chart(price_volume_figure(f, sym), use_container_width=True)
    st.caption("Xanh dương = MA10, vàng = MA20 (cả giá và khối lượng). Nến xanh tăng, đỏ giảm.")
with t2:
    st.caption("Chạy lại đúng kế hoạch của app trên lịch sử: T+2, biên độ ±7%, phí/thuế/trượt giá, gap, stop ưu tiên.")
    if st.button("Chạy kiểm chứng"):
        with st.spinner("Đang mô phỏng..."):
            r = backtest_plan(f, capital=cap, risk_pct=risk)
        if "error" in r: st.error(r["error"])
        else:
            a, b, c = st.columns(3)
            a.metric("Số lệnh", r["closed_trades"]); b.metric("Tỷ lệ thắng", f"{r['win_rate_pct']:.0f}%" if r["win_rate_pct"] is not None else "—")
            c.metric("R trung bình", f"{r['avg_r']:.2f}" if r["avg_r"] is not None else "—")
            a.metric("Lợi nhuận chiến lược", f"{r['total_return_pct']:.1f}%"); b.metric("Mua & giữ", f"{r['buy_hold_pct']:.1f}%")
            c.metric("Sụt vốn tối đa", f"{r['max_drawdown_pct']:.1f}%")
            (st.success if r["enough_sample"] else st.warning)(r["note"] + f" (hiện có {r['closed_trades']} lệnh)")
            if r["trades"]: st.dataframe(pd.DataFrame(r["trades"]), use_container_width=True)
            ts = trap_forward_stats(f)
            if ts: st.markdown("**Sau tín hiệu bẫy/dòng tiền, 5 phiên tiếp theo giá thế nào?**"); st.json(ts)
