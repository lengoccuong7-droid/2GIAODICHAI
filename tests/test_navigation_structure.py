from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_three_primary_navigation_tabs_are_declared():
    source = APP.read_text(encoding="utf-8")
    expected = [
        "1 · TỔNG QUAN",
        "2 · DẤU CHÂN DÒNG TIỀN",
        "3 · CHIẾN LƯỢC",
    ]
    start = source.index("main_tabs = st.tabs([", source.index("V4.5: three-tab decision interface"))
    end = source.index("])", start)
    nav = source[start:end]
    assert all(label in nav for label in expected)
    assert source.count("with main_tabs[0]:") == 1
    assert source.count("with main_tabs[1]:") == 1
    assert source.count("with main_tabs[2]:") == 1
    assert "with main_tabs[3]:" not in source
    assert "with main_tabs[4]:" not in source


def test_flow_participant_data_is_optional_and_explicit():
    source = APP.read_text(encoding="utf-8")
    for field in ("foreign_net_value", "proprietary_net_value", "large_buy_value", "large_sell_value"):
        assert field in source
    assert "Không được suy diễn các nhóm này từ OHLCV" in source


def test_strategy_tab_has_both_position_modes():
    source = APP.read_text(encoding="utf-8")
    assert "CHƯA CÓ HÀNG · Kế hoạch mở vị thế" in source
    assert "ĐANG CÓ HÀNG · Quản trị vị thế" in source
    assert "TẢI BÁO CÁO CHIẾN LƯỢC JSON" in source
