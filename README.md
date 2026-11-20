# BOT TRADER SWING MULTI V5.1

Công cụ hỗ trợ lập kế hoạch swing 5–20 phiên cho cổ phiếu VN. Không tự đặt lệnh, không cam kết lợi nhuận.

## Chạy
```
pip install -r requirements.txt
streamlit run app.py
```
- `app.py`: bản phân tích đầy đủ (3 tab) cho người có kinh nghiệm.
- Trang **"Nguoi moi va Kiem chung"** (thanh bên trái): kết luận 1 màn hình bằng tiếng thường + kiểm chứng.

## Mới ở V5.1
- `modules/beginner_summary.py`: chuyển kết quả engine thành hành động cụ thể (mua ở đâu, bao nhiêu, cắt lỗ, chốt lời).
- `modules/plan_backtest.py`: chạy lại đúng kế hoạch của app trên lịch sử, có T+2, biên độ ±7%, gap, phí/thuế/trượt giá, stop ưu tiên; thống kê 5 phiên sau tín hiệu bẫy/dòng tiền.
- Tài liệu cũ chuyển vào `docs/archive/`.

## Giới hạn
- Dữ liệu nạp tay; chưa có nguồn giá realtime.
- Dòng tiền từ OHLCV chỉ là proxy. Khối ngoại/tự doanh chỉ có khi bạn nạp file.
- Kiểm chứng cần ≥30 lệnh trên dữ liệu thật mới có ý nghĩa; dữ liệu demo là dữ liệu giả.
- Chưa test giao diện Streamlit trên trình duyệt/điện thoại.
