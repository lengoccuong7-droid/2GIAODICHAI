import io, json, math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from config import APP_NAME, DEFAULT_CAPITAL_VND, APP_VERSION, PRICE_UNIT_OPTIONS, DEFAULT_PRICE_UNIT, DEFAULT_TAX_PCT
from modules.data_loader import load_market_file
from modules.indicators import add_indicators
from modules.support_resistance import find_support_resistance
from modules.data_quality import assess_data_quality
from modules.backtest import simple_swing_backtest
from modules.risk_manager import calculate_position_size
from modules.relative_strength import relative_strength
from modules.market_analysis import analyze_symbol, analyze_universe
from modules.stock_scanner import scan_universe
from modules.market_regime import assess_market_regime
from modules.multi_timeframe import timeframe_snapshot
from modules.fundamental_analysis import valuation_metrics, quality_flags
from modules.price_units import normalize_price_unit
from modules.assistant_reasoning import build_reasoning_report
from modules.knowledge_hub import index_document, search_documents, list_documents
from modules.master_counsel import build_master_counsel, answer_counsel_question, save_decision_record, list_decision_records
from modules.swing_trading_expert import analyze_swing_setup, scan_swing_universe
from modules.shared_analysis import build_shared_analysis_context
from modules.market_flow_trap import analyze_flow_and_traps

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.block-container{padding-top:1.1rem;padding-bottom:2rem}
[data-testid="stMetricValue"]{font-size:1.35rem}
.stTabs [data-baseweb="tab-list"]{gap:.25rem;flex-wrap:nowrap;overflow-x:auto;position:sticky;top:0;z-index:9999;background:var(--background-color);padding:.45rem .15rem;border-bottom:1px solid rgba(128,128,128,.25);scrollbar-width:thin;isolation:isolate}
.stTabs [data-baseweb="tab"]{white-space:nowrap;padding:.48rem .68rem;min-height:2.45rem}
.stTabs [data-baseweb="tab-panel"]{padding-top:1rem;scroll-margin-top:4rem}
[data-testid="stAlert"]{padding:.55rem .75rem;border-radius:10px}
[data-testid="stJson"]{max-width:100%;overflow-x:auto}
@media(max-width:700px){.stTabs [data-baseweb="tab-list"]{top:0;position:sticky}.block-container{padding-left:.65rem;padding-right:.65rem;padding-top:.55rem}.stTabs [data-baseweb="tab-list"]{top:0}.stTabs [data-baseweb="tab"]{font-size:.76rem;padding:.4rem .55rem}.stTabs [data-baseweb="tab-panel"]{padding-left:0;padding-right:0}.stMetric{padding:.35rem}.signal-card{padding:.7rem}}

.signal-card{padding:1rem;border-radius:12px;background:rgba(128,128,128,.10);border:1px solid rgba(128,128,128,.25)}
</style>
""", unsafe_allow_html=True)
st.title(APP_NAME)
st.caption(f"V{APP_VERSION} · Phân tích dữ liệu lịch sử và mô phỏng · Không đặt lệnh tự động")
st.caption("📱 Trên điện thoại: mở bảng nhập liệu bằng biểu tượng ☰ ở góc trên bên trái. Sidebar được đặt mặc định ở trạng thái mở khi khởi động.")

# -------------------- Helpers --------------------
def fmt(v, digits=2):
    try:
        if v is None or pd.isna(v) or not np.isfinite(float(v)): return "Chưa đủ dữ liệu"
        return f"{float(v):,.{digits}f}"
    except Exception: return "Chưa đủ dữ liệu"

def read_upload(up, hint=None):
    if up is None: return None
    return load_market_file(up, symbol_hint=hint)

def enrich(frame):
    x=add_indicators(frame.sort_values("date").copy())
    x["LOG_RETURN"] = np.log(x.close / x.close.shift(1))
    x["CLV"] = np.where((x.high-x.low)!=0, ((x.close-x.low)-(x.high-x.close))/(x.high-x.low), np.nan)
    vol_mean=x.volume.rolling(20,min_periods=20).mean()
    vol_std=x.volume.rolling(20,min_periods=20).std(ddof=0)
    x["VOLUME_ZSCORE"]=(x.volume-vol_mean)/vol_std.replace(0,np.nan)
    x["TURNOVER"] = x.close*x.volume
    x["ILLIQUIDITY_AMIHUD"] = (x.LOG_RETURN.abs()/x.TURNOVER.replace(0,np.nan)).rolling(20,min_periods=20).mean()
    return x

def annualized_gk(x):
    if len(x)<20:return None
    term=.5*np.log(x.high/x.low)**2-(2*np.log(2)-1)*np.log(x.close/x.open)**2
    v=term.replace([np.inf,-np.inf],np.nan).dropna()
    return float(np.sqrt(max(v.mean(),0)*252)*100) if len(v)>=10 else None

def annualized_parkinson(x):
    if len(x)<20:return None
    v=(np.log(x.high/x.low)**2).replace([np.inf,-np.inf],np.nan).dropna()
    return float(np.sqrt(v.mean()/(4*np.log(2))*252)*100) if len(v)>=10 else None

def hurst_rs(series):
    s=pd.Series(series).dropna().astype(float).values
    if len(s)<100:return None
    lags=[8,16,32,64]
    vals=[]
    for lag in lags:
        chunks=[]
        for i in range(0,len(s)-lag+1,lag):
            z=s[i:i+lag]; z=z-z.mean(); sd=z.std()
            if sd>0: chunks.append((z.cumsum().max()-z.cumsum().min())/sd)
        if chunks: vals.append((lag,float(np.mean(chunks))))
    if len(vals)<3:return None
    return float(np.polyfit(np.log([a for a,b in vals]),np.log([b for a,b in vals]),1)[0])

def return_entropy(series, bins=10):
    s=pd.Series(series).dropna().astype(float)
    if len(s)<50:return None
    counts=np.histogram(s,bins=bins)[0].astype(float)
    p=counts[counts>0]/counts.sum()
    return float(-(p*np.log(p)).sum()/np.log(bins))

def audit_flags(x):
    flags=[]
    if x.empty:return flags
    last=x.iloc[-1]
    if len(x)>1:
        prev=x.iloc[-2]
        if last.close>0 and prev.close>0 and abs(last.close/prev.close-1)>.25:
            flags.append("Biến động giá ngày gần nhất vượt 25%: kiểm tra điều chỉnh giá, chia tách hoặc dữ liệu lỗi.")
    if (x.volume==0).any(): flags.append(f"Có {(x.volume==0).sum()} phiên khối lượng bằng 0.")
    if x.date.duplicated().any(): flags.append("Có ngày giao dịch trùng lặp.")
    if x.date.is_monotonic_increasing is False: flags.append("Ngày chưa sắp xếp tăng dần.")
    return flags

# -------------------- Sidebar: all inputs --------------------
with st.sidebar:
    st.header("DỮ LIỆU & THÔNG SỐ")
    st.caption("Nhập dữ liệu và cấu hình tại đây. Trên iPhone, nếu thanh này đang ẩn, chạm biểu tượng ☰ ở góc trên bên trái.")
    price_unit=st.selectbox("Đơn vị giá trong file", list(PRICE_UNIT_OPTIONS), index=list(PRICE_UNIT_OPTIONS).index(DEFAULT_PRICE_UNIT), help="Chọn đúng đơn vị đang lưu trong file. App quy đổi về đồng/cổ phiếu trước khi tính NAV, định giá và quản trị vốn.")
    price_multiplier=PRICE_UNIT_OPTIONS[price_unit]
    main_file=st.file_uploader("1. Dữ liệu OHLCV chính",type=["csv","xlsx","xls","txt","json","parquet"],key="main_ohlcv")
    symbol_hint=st.text_input("Mã cổ phiếu (bắt buộc nếu file không có cột mã)","",key="symbol_hint",help="Ví dụ: DXG, FPT, HDB. App không tự gán UNKNOWN hoặc tự đoán mã.")
    sheet=st.text_input("Tên sheet Excel (để trống = sheet đầu)","",key="sheet_name")
    benchmark_file=st.file_uploader("2. Dữ liệu VN-Index (không bắt buộc)",type=["csv","xlsx","xls","txt"],key="benchmark_file")
    sector_file=st.file_uploader("3. Dữ liệu chỉ số ngành / nhóm ngành (không bắt buộc)",type=["csv","xlsx","xls","txt"],key="sector_file")
    st.divider()
    position_mode=st.radio("TRẠNG THÁI VỊ THẾ",["🟢 CHƯA CÓ HÀNG – Cầm tiền","🔵 ĐANG CÓ HÀNG – Cầm cổ phiếu"],key="position_mode")
    nav=st.number_input("Tổng NAV (VND)",min_value=0,value=int(DEFAULT_CAPITAL_VND),step=5_000_000,key="nav_vnd")
    risk_pct=st.number_input("Rủi ro tối đa / giao dịch (%)",min_value=.1,max_value=5.0,value=1.0,step=.1,key="risk_pct")
    max_alloc=st.number_input("Tỷ trọng tối đa / mã (%)",min_value=1.0,max_value=100.0,value=20.0,step=1.0,key="max_alloc")
    lot=st.number_input("Bước lô",min_value=1,value=100,step=1,key="lot_size")
    fee=st.number_input("Phí mỗi chiều (%)",min_value=0.0,max_value=2.0,value=.15,step=.01,key="fee_pct")
    slip=st.number_input("Trượt giá mỗi chiều (%)",min_value=0.0,max_value=2.0,value=.10,step=.01,key="slippage_pct")
    tax_pct=st.number_input("Thuế bán (%)",min_value=0.0,max_value=2.0,value=float(DEFAULT_TAX_PCT),step=.01,key="tax_pct")
    st.caption("Các mục bên dưới chỉ dùng khi đang có vị thế.")
    qty=st.number_input("Số lượng đang nắm giữ",min_value=0,value=0,step=100,key="held_quantity")
    entry=st.number_input("Giá vốn (đồng/cp)",min_value=0.0,value=0.0,step=100.0,key="average_entry")
    stop_assumption=st.number_input("Giá dừng lỗ giả định (đồng/cp)",min_value=0.0,value=0.0,step=100.0,key="manual_stop")
    st.caption("Nếu thiếu dữ liệu benchmark/ngành, tab RS sẽ ghi rõ chưa đủ dữ liệu.")

# Parse uploads with safe error reporting
main_data=bench_data=sector_data=None
load_errors=[]
try:
    if main_file:
        sheet_value = sheet.strip()
        if sheet_value.isdigit():
            sheet_value = int(sheet_value)
        main_data=load_market_file(main_file,sheet_name=(sheet_value if sheet_value != "" else 0),symbol_hint=symbol_hint.strip() or None)
except Exception as e: load_errors.append(f"OHLCV: {e}")
try:
    if benchmark_file: bench_data=load_market_file(benchmark_file,symbol_hint="VNINDEX")
except Exception as e: load_errors.append(f"VN-Index: {e}")
try:
    if sector_file: sector_data=load_market_file(sector_file,symbol_hint="SECTOR")
except Exception as e: load_errors.append(f"Ngành: {e}")
for e in load_errors: st.sidebar.error(e)

# Explicit price-unit normalization: never guess whether source prices are VND or thousand VND.
if main_data is not None and not main_data.empty:
    raw_median_close = float(pd.to_numeric(main_data["close"], errors="coerce").median())
    if price_multiplier == 1000.0 and 1000 <= raw_median_close <= 500000:
        st.sidebar.warning("Đơn vị cần kiểm tra: giá gốc trong file đang ở khoảng hàng nghìn–trăm nghìn. Nếu file đã là VND/cp, hãy chọn Đồng/cổ phiếu để tránh nhân 1.000 lần.")
    elif price_multiplier == 1.0 and 1 <= raw_median_close < 1000:
        st.sidebar.warning("Đơn vị cần kiểm tra: giá gốc dưới 1.000. Nếu file ghi theo nghìn đồng/cp, hãy chọn Nghìn đồng/cổ phiếu.")
main_data=normalize_price_unit(main_data,price_multiplier)
bench_data=normalize_price_unit(bench_data,price_multiplier)
sector_data=normalize_price_unit(sector_data,price_multiplier)
if main_data is not None:
    st.sidebar.caption(f"Chuẩn giá sau quy đổi: VND/cổ phiếu · Hệ số áp dụng: ×{price_multiplier:g}")

if main_data is not None:
    symbols=sorted(main_data.symbol.astype(str).unique().tolist())
    with st.sidebar:
        selected_symbol=st.selectbox("Mã phân tích",symbols,index=0)
else: selected_symbol=None

if main_data is not None and selected_symbol:
    raw=main_data[main_data.symbol.astype(str)==selected_symbol].sort_values("date").reset_index(drop=True)
    data=enrich(raw)
else:
    raw=data=None

# One shared analysis snapshot feeds all analysis and Swing views.
shared_context = None
shared_context_error = None
if data is not None and selected_symbol:
    try:
        shared_context = build_shared_analysis_context(
            selected_symbol, raw, data, universe=main_data, benchmark=bench_data, sector=sector_data,
            capital_vnd=float(nav), risk_pct=float(risk_pct),
            max_alloc_pct=float(max_alloc), lot_size=int(lot)
        )
    except Exception as exc:
        shared_context_error = str(exc)




# -------------------- V4.5: three-tab decision interface --------------------
def _safe_num(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None

def _score01(value):
    return int(max(0, min(100, round(value))))

def _flow_status(net_value):
    if net_value is None:
        return "CHƯA CÓ DỮ LIỆU"
    if net_value > 0:
        return "VÀO"
    if net_value < 0:
        return "RA"
    return "TRUNG TÍNH"

def _load_participant_flow(upload):
    """Optional CSV/XLSX schema: date, foreign_net_value, proprietary_net_value,
    large_buy_value, large_sell_value. Values are net/turnover in VND.
    """
    if upload is None:
        return None, []
    try:
        name = upload.name.lower()
        frame = pd.read_excel(upload) if name.endswith((".xlsx", ".xls")) else pd.read_csv(upload)
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        required = ["date", "foreign_net_value", "proprietary_net_value", "large_buy_value", "large_sell_value"]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            return None, ["Thiếu cột: " + ", ".join(missing)]
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for col in required[1:]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date")
        if frame.empty:
            return None, ["Không có dòng dữ liệu hợp lệ."]
        return frame, []
    except Exception as exc:
        return None, [f"Không đọc được file dòng tiền: {exc}"]

flow_upload = st.sidebar.file_uploader(
    "4. Dữ liệu dòng tiền bổ sung (CSV/XLSX, tùy chọn)",
    type=["csv", "xlsx", "xls"], key="participant_flow_file",
    help="Cột bắt buộc: date, foreign_net_value, proprietary_net_value, large_buy_value, large_sell_value. Giá trị tiền VND; mua/bán ròng tính theo 5 phiên gần nhất."
)
participant_flow, participant_flow_errors = _load_participant_flow(flow_upload)
for msg in participant_flow_errors:
    st.sidebar.warning(msg)

# Build the single strategy report once; the three tabs read the same snapshot.
swing_report = None
if data is not None and shared_context is not None:
    try:
        swing_is_holding = str(position_mode).startswith("🔵")
        swing_position = ({"mode": "MANAGE_EXISTING", "quantity": int(qty), "entry_price": float(entry), "stop_price": float(stop_assumption)}
                          if swing_is_holding else {"mode": "NEW_ENTRY", "quantity": 0})
        swing_docs = search_documents("doanh thu lợi nhuận dòng tiền rủi ro triển vọng định giá", ticker=selected_symbol, limit=8)
        swing_report = analyze_swing_setup(
            data, market_regime=shared_context["market_regime"],
            rs_market=shared_context["rs_market"], rs_sector=shared_context["rs_sector"],
            documents=swing_docs, capital_vnd=float(nav), risk_pct=float(risk_pct),
            max_allocation_pct=float(max_alloc), lot_size=int(lot),
            position=swing_position, risk_style="CÂN BẰNG",
            data_conflicts=audit_flags(raw), data_quality=shared_context["data_quality"],
            flow_trap=shared_context.get("flow_trap")
        )
    except Exception as exc:
        st.error(f"Không thể tạo báo cáo chiến lược: {exc}")

# Compact status helpers.
def _trend_label(frame):
    if frame is None or len(frame) < 50:
        return "CHƯA ĐỦ DỮ LIỆU"
    r = frame.iloc[-1]
    close, ma20, ma50 = (_safe_num(r.get(k)) for k in ("close", "SMA20", "SMA50"))
    if None in (close, ma20, ma50):
        return "CHƯA ĐỦ DỮ LIỆU"
    if close > ma20 > ma50:
        return "TĂNG"
    if close < ma20 < ma50:
        return "GIẢM"
    return "ĐI NGANG / CHUYỂN PHA"

def _five_scores(frame, context, report):
    if frame is None or frame.empty or report is None or report.get("score") is None:
        return None
    r = frame.iloc[-1]
    close = _safe_num(r.get("close")); ma20 = _safe_num(r.get("SMA20")); ma50 = _safe_num(r.get("SMA50"))
    rsi = _safe_num(r.get("RSI14")); macd = _safe_num(r.get("MACD_HIST")); adx = _safe_num(r.get("ADX14"))
    rvol = _safe_num(r.get("REL_VOLUME20")); mfi = _safe_num(r.get("MFI14")); cmf = _safe_num(r.get("CMF20"))
    trend = 50
    if close and ma20 and ma50:
        trend = 85 if close > ma20 > ma50 else 20 if close < ma20 < ma50 else 50
    momentum = 50
    if rsi is not None:
        momentum += 20 if 50 <= rsi <= 68 else 8 if 45 <= rsi < 50 or 68 < rsi <= 72 else -15 if rsi < 40 or rsi > 78 else 0
    if macd is not None: momentum += 15 if macd >= 0 else -10
    if adx is not None: momentum += 10 if adx >= 25 else 0
    volume = 50
    if rvol is not None: volume += 20 if rvol >= 1.3 else 8 if rvol >= 1 else -12
    if mfi is not None: volume += 10 if 50 <= mfi <= 80 else -8 if mfi < 35 or mfi > 85 else 0
    if cmf is not None: volume += 10 if cmf > 0.05 else -10 if cmf < -0.05 else 0
    flow = context.get("flow_trap", {}) if context else {}
    flow_score = 50 + (15 * int(flow.get("flow_score") or 0))
    if flow.get("trap_risk") == "HIGH": flow_score -= 20
    elif flow.get("trap_risk") == "MODERATE": flow_score -= 8
    structure = 50
    if report.get("setup") in ("BREAKOUT XÁC NHẬN", "PULLBACK TRONG XU HƯỚNG TĂNG", "HỒI PHỤC CÓ XÁC NHẬN"): structure = 82
    elif report.get("setup") == "NỀN TÍCH LŨY": structure = 65
    elif "BẬT GIÁ" in str(report.get("setup")) or "HỒI PHỤC SỚM" in str(report.get("setup")): structure = 38
    if flow.get("trap_risk") == "HIGH": structure -= 15
    return {
        "Xu hướng": _score01(trend), "Động lượng": _score01(momentum),
        "Khối lượng": _score01(volume), "Dòng tiền & bẫy giá": _score01(flow_score),
        "Cấu trúc & thị trường": _score01(structure),
    }

st.markdown("### Phân tích tập trung · 3 tab")
st.caption("Một bộ dữ liệu – một phiên phân tích – một kế hoạch giao dịch nhất quán. Chỉ dùng dữ liệu đã nạp; không tự tạo dữ liệu thị trường.")
main_tabs = st.tabs(["1 · TỔNG QUAN", "2 · DẤU CHÂN DÒNG TIỀN", "3 · CHIẾN LƯỢC"])

# TAB 1 — OVERVIEW
with main_tabs[0]:
    if data is None:
        st.info("Mở thanh DỮ LIỆU & THÔNG SỐ ở bên trái, tải OHLCV và chọn mã cổ phiếu để bắt đầu.")
    elif swing_report is None:
        st.warning("Chưa tạo được báo cáo. Kiểm tra dữ liệu đầu vào và thông báo lỗi phía trên.")
    else:
        last = data.iloc[-1]
        score = swing_report.get("score")
        a, b, c, d = st.columns(4)
        a.metric("MÃ CỔ PHIẾU", selected_symbol)
        a.caption(f"Ngày dữ liệu cuối: {pd.to_datetime(last.date).strftime('%d/%m/%Y')}")
        b.metric("ĐIỂM TỔNG QUAN", "—" if score is None else f"{int(score)}/100")
        c.metric("XU HƯỚNG", _trend_label(data))
        d.metric("GIÁ ĐÓNG CỬA", fmt(last.close, 0) + " đ")
        if score is not None:
            st.progress(int(max(0, min(100, score))), text=f"Điểm sàng lọc Swing: {int(score)}/100 · Không phải xác suất thắng")
        st.markdown("#### 5 nhóm chỉ số")
        scores = _five_scores(data, shared_context, swing_report)
        if scores:
            cols = st.columns(5)
            for col, (label, value) in zip(cols, scores.items()):
                col.metric(label, f"{value}/100")
                col.progress(value)
        else:
            st.warning("Chưa đủ dữ liệu để tính điểm tổng quan và 5 nhóm chỉ số.")
        st.markdown("#### Nhận định nhanh")
        st.write(f"**Mô hình hiện tại:** {swing_report.get('setup', 'Chưa xác định')}")
        st.write(f"**Kết luận:** {swing_report.get('conclusion', swing_report.get('decision', 'Chưa đủ dữ liệu'))}")
        positives = swing_report.get("positives", []) or []
        negatives = swing_report.get("negatives", []) or []
        left, right = st.columns(2)
        with left:
            st.markdown("**Điểm ủng hộ**")
            for item in positives[:4]: st.success(item)
            if not positives: st.caption("Chưa có luận cứ tích cực đủ điều kiện.")
        with right:
            st.markdown("**Rủi ro cần lưu ý**")
            for item in negatives[:4]: st.warning(item)
            if not negatives: st.caption("Chưa phát hiện theo bộ lọc hiện tại; không đồng nghĩa không có rủi ro.")
        with st.expander("Biểu đồ giá và MA20/MA50"):
            chart = data.tail(120)
            fig = go.Figure(data=[go.Candlestick(x=chart.date, open=chart.open, high=chart.high, low=chart.low, close=chart.close, name=selected_symbol)])
            for col_name in ("SMA20", "SMA50"):
                if col_name in chart.columns:
                    fig.add_trace(go.Scatter(x=chart.date, y=chart[col_name], mode="lines", name=col_name, line=dict(width=1)))
            fig.update_layout(height=390, margin=dict(l=5,r=5,t=20,b=5), xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        quality = shared_context.get("data_quality", {}) if shared_context else {}
        st.caption(f"Chất lượng dữ liệu: {quality.get('status', 'CHƯA RÕ')} · Số phiên: {len(raw)} · Nguồn: file người dùng tải lên.")

# TAB 2 — FLOW FOOTPRINT
with main_tabs[1]:
    if data is None:
        st.info("Tải OHLCV để xem OBV và dấu chân cung cầu. Dữ liệu khối ngoại/tự doanh/lệnh lớn là file bổ sung tùy chọn ở Sidebar.")
    else:
        flow = (shared_context or {}).get("flow_trap", {})
        fm = flow.get("metrics", {})
        st.markdown("#### Dấu chân dòng tiền")
        st.caption("Dữ liệu bên tham gia chỉ hiển thị khi có file bổ sung. OBV/CLV từ OHLCV là proxy, không phải dữ liệu lệnh chủ động.")
        ext = None
        if participant_flow is not None and not participant_flow.empty:
            ext = participant_flow.tail(5)
        if ext is not None:
            foreign_net = float(ext.foreign_net_value.fillna(0).sum())
            prop_net = float(ext.proprietary_net_value.fillna(0).sum())
            large_buy = float(ext.large_buy_value.fillna(0).sum())
            large_sell = float(ext.large_sell_value.fillna(0).sum())
            large_net = large_buy - large_sell
        else:
            foreign_net = prop_net = large_net = None
        vals = [
            ("Khối ngoại", foreign_net, "Giá trị mua/bán ròng 5 phiên"),
            ("Tự doanh", prop_net, "Giá trị mua/bán ròng 5 phiên"),
            ("Lệnh lớn", large_net, "Mua lớn trừ bán lớn · 5 phiên"),
            ("OBV / CLV proxy", _safe_num(fm.get("obv_proxy_change_20")), "Thay đổi OBV proxy 20 phiên"),
        ]
        cols = st.columns(4)
        for col, (label, value, help_text) in zip(cols, vals):
            col.metric(label, _flow_status(value), help=help_text)
            if value is not None:
                col.caption(f"Giá trị: {value:,.0f}" if label != "OBV / CLV proxy" else f"OBV proxy: {value:,.0f}")
        st.markdown("#### Kết luận dòng tiền")
        st.write(f"**Proxy cung cầu OHLCV:** {flow.get('flow_label', 'Chưa đủ dữ liệu')}")
        st.write(f"**Rủi ro bẫy giá:** {flow.get('trap_label', 'Chưa đủ dữ liệu')}")
        # Trap Detection 2.0: distinguish a suspected event from a follow-through-confirmed event.
        trap_detail = flow.get("trap_analysis", {})
        with st.expander("Trap Detection 2.0 · Xác nhận bẫy qua nhiều phiên", expanded=True):
            st.caption("Quét mẫu hình trong 8 phiên gần nhất; dùng tối đa 3 phiên tiếp theo để xác nhận. Sự kiện mới xuất hiện được gắn CHỜ XÁC NHẬN.")
            trap_cols = st.columns(3)
            trap_cols[0].metric("Bull Trap xác nhận", fm.get("confirmed_bull_traps", 0))
            trap_cols[1].metric("Bull Trap chờ xác nhận", fm.get("pending_bull_traps", 0))
            trap_cols[2].metric("Breakout giữ được vùng", fm.get("confirmed_breakouts", 0))
            events = trap_detail.get("bull_trap", []) + trap_detail.get("bear_trap", []) + trap_detail.get("breakout", [])
            events = sorted(events, key=lambda e: e.get("date", ""), reverse=True)
            if events:
                display_events = []
                for ev in events[:8]:
                    display_events.append({
                        "Ngày": ev.get("date"),
                        "Loại": {"BULL_TRAP": "Nghi vấn bẫy tăng", "BEAR_TRAP": "Nghi vấn bẫy giảm", "BREAKOUT": "Breakout"}.get(ev.get("type"), ev.get("type")),
                        "Trạng thái": ev.get("label"),
                        "Mốc giá": round(float(ev.get("level", 0)), 3),
                        "RVOL": round(float(ev.get("relative_volume", 0)), 2),
                        "Phiên theo dõi": ev.get("followup_sessions", 0),
                        "Diễn giải": ev.get("reason"),
                    })
                st.dataframe(pd.DataFrame(display_events), use_container_width=True, hide_index=True)
            else:
                st.info("Chưa phát hiện mẫu hình bẫy hoặc breakout đủ điều kiện trong cửa sổ quét.")
            st.caption("CONFIRMED = đã có diễn biến tiếp theo ủng hộ mẫu hình; PENDING = chưa đủ 3 phiên; INVALIDATED = nghi vấn bị vô hiệu; UNCONFIRMED = chưa đủ bằng chứng.")
        if ext is None:
            st.warning("Chưa có dữ liệu khối ngoại, tự doanh và lệnh lớn. Không được suy diễn các nhóm này từ OHLCV.")
        else:
            st.caption(f"Dữ liệu bổ sung đến ngày {ext.date.max().strftime('%d/%m/%Y')} · cộng gộp 5 dòng/phiên gần nhất có trong file.")
        for item in flow.get("signals", [])[:8]: st.write("• " + str(item))
        for item in flow.get("cautions", [])[:4]: st.caption("Lưu ý: " + str(item))
        if participant_flow is not None:
            with st.expander("Kiểm tra dữ liệu dòng tiền đã nạp"):
                st.dataframe(participant_flow.tail(10), use_container_width=True, hide_index=True)

# TAB 3 — STRATEGY
with main_tabs[2]:
    if data is None:
        st.info("Tải dữ liệu OHLCV và chọn trạng thái vị thế ở Sidebar để lập chiến lược.")
    elif swing_report is None:
        st.warning("Chưa đủ điều kiện tạo chiến lược.")
    else:
        st.markdown("#### Kết luận hành động")
        st.subheader(swing_report.get("conclusion", swing_report.get("decision", "CHƯA ĐỦ DỮ LIỆU")))
        st.caption("Kế hoạch tham chiếu theo quy tắc; không tự đặt lệnh. Nếu dữ liệu thiếu hoặc có điều kiện chặn, ưu tiên đứng ngoài/chờ xác nhận.")
        pos = swing_report.get("position", {})
        if pos.get("mode") == "MANAGE_EXISTING":
            st.markdown("### ĐANG CÓ HÀNG · Quản trị vị thế")
            p1,p2,p3,p4 = st.columns(4)
            p1.metric("Số lượng", f"{int(pos.get('quantity', 0)):,} cp")
            p2.metric("Giá vốn", fmt(pos.get("entry_price"), 0) + " đ")
            p3.metric("Giá hiện tại", fmt(pos.get("market_price"), 0) + " đ")
            p4.metric("Lãi/lỗ tạm tính", fmt(pos.get("pnl_pct_before_costs"), 2) + "%")
            p5,p6,p7 = st.columns(3)
            p5.metric("Mốc bảo vệ / SL", fmt(pos.get("reference_invalidation"), 0) + " đ")
            p6.metric("Mục tiêu 1", fmt(pos.get("target_1_reference"), 0) + " đ")
            p7.metric("Mục tiêu 2", fmt(pos.get("target_2_reference"), 0) + " đ")
            st.info(f"Đánh giá vị thế: {pos.get('status', 'CHƯA RÕ')} · Nguồn SL: {pos.get('stop_source', 'Tham chiếu')}")
            st.write("**Dời Stop Loss tham chiếu:** " + fmt(pos.get("trailing_stop_reference"), 0) + " đ")
            for stage in pos.get("management_stages", []):
                st.write(f"• **{stage.get('stage')}** — {stage.get('condition')}: {stage.get('action')}")
            if int(pos.get("quantity", 0)) <= 0 or not pos.get("entry_price"):
                st.warning("Nhập số lượng đang nắm giữ và giá vốn ở Sidebar để hệ thống đánh giá đúng vị thế.")
        else:
            st.markdown("### CHƯA CÓ HÀNG · Kế hoạch mở vị thế")
            plan = swing_report.get("plan", {})
            p1,p2,p3,p4 = st.columns(4)
            p1.metric("Giá kích hoạt", fmt(plan.get("reference_trigger"), 0) + " đ")
            p2.metric("Stop Loss", fmt(plan.get("invalidation"), 0) + " đ")
            p3.metric("Take Profit 1", fmt(plan.get("target_1"), 0) + " đ")
            p4.metric("Take Profit 2", fmt(plan.get("target_2"), 0) + " đ")
            st.write(f"**Tỷ lệ R:R:** TP1 {fmt(plan.get('rr_target_1'), 2)} : 1 · TP2 {fmt(plan.get('rr_target_2'), 2)} : 1")
            sizing = swing_report.get("sizing", {})
            q1,q2,q3 = st.columns(3)
            q1.metric("Khối lượng tham chiếu", f"{int(sizing.get('quantity', 0)):,} cp")
            q2.metric("Giá trị vị thế", f"{float(sizing.get('position_value', 0) or 0):,.0f} đ")
            q3.metric("Rủi ro mô phỏng", f"{float(sizing.get('risk_amount', 0) or 0):,.0f} đ")
            st.caption(f"Ngân sách rủi ro: {float(sizing.get('risk_budget', 0) or 0):,.0f} đ · Sizing: {sizing.get('status', 'CHƯA RÕ')}")
        blockers = swing_report.get("blockers", []) or []
        if blockers:
            st.markdown("#### Điều kiện chặn / chưa đạt")
            for item in blockers: st.warning(item)
        else:
            st.success("Không có điều kiện chặn theo bộ quy tắc hiện tại; vẫn cần xác minh dữ liệu và tuân thủ kế hoạch.")
        if swing_report.get("next_steps"):
            st.markdown("#### Việc cần làm tiếp theo")
            for item in swing_report.get("next_steps", []): st.write("• " + str(item))
        with st.expander("Luận cứ và dữ liệu kiểm tra"):
            st.write("**Điểm ủng hộ**")
            for item in swing_report.get("positives", []): st.write("• " + str(item))
            st.write("**Điểm bất lợi**")
            for item in swing_report.get("negatives", []): st.write("• " + str(item))
            st.json(swing_report.get("evidence", {}))
        st.download_button(
            "TẢI BÁO CÁO CHIẾN LƯỢC JSON",
            data=json.dumps(swing_report, ensure_ascii=False, indent=2, default=str).encode("utf-8-sig"),
            file_name=f"{selected_symbol}_swing_strategy_v4.5.1.json",
            mime="application/json", key="download_swing_strategy_v45"
        )

st.divider()
st.caption("V4.5.1 · 3 tab: Tổng quan / Dấu chân dòng tiền / Chiến lược. Dữ liệu dòng tiền khối ngoại, tự doanh, lệnh lớn chỉ được hiển thị khi người dùng cung cấp file tương ứng. Phân tích OHLCV là proxy, không xác định danh tính tổ chức và chưa được kiểm định lợi nhuận bằng walk-forward.")
