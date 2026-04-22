# 📈 Dự báo Giá & Biến động Cổ phiếu Ngân hàng Việt Nam

> Sử dụng Machine Learning và Deep Learning để phân tích và dự báo cổ phiếu 3 ngân hàng lớn: **BID** (BIDV), **CTG** (VietinBank), **VCB** (Vietcombank).

---

## 📁 Cấu trúc dự án

```
├── 📊 Data
│   ├── banks_BID_dataset.csv              # Dữ liệu cổ phiếu BIDV
│   ├── banks_CTG_dataset.csv              # Dữ liệu cổ phiếu VietinBank
│   └── banks_VCB_dataset.csv              # Dữ liệu cổ phiếu Vietcombank
│
├── 🔍 Phase 1 – Phân tích Khám phá (3 Lăng kính)
│   ├── langkinh1_xgboost_shap.py          # Lăng kính 1: XGBoost + SHAP
│   ├── langkinh2_neuralprophet_seasonality.py  # Lăng kính 2: NeuralProphet + Seasonality
│   └── langkinh3_tft_memory.py            # Lăng kính 3: TFT + ACF Memory
│
├── 🧪 Phase 2 – Đánh giá Mô hình
│   ├── run_4fold_all_models_both_targets.py   # 4-Fold Cross Validation (5 models)
│   ├── run_perday_5models.py              # Mô phỏng giao dịch theo ngày
│   ├── run_market_event_validation.py     # Kiểm định ngày biến động cao
│   └── run_sensitivity_analysis.py        # Phân tích độ nhạy hyperparameter
│
├── 📂 Kết quả Phase 1
│   ├── langkinh1_xgboost_shap/            # Charts + CSV: Feature importance, SHAP
│   ├── langkinh2_neuralprophet_seasonality/  # Charts + CSV: Seasonality, Calendar effects
│   └── langkinh3_tft_memory/              # Charts + CSV: ACF/PACF, Attention
│
├── 📂 Kết quả Phase 2
│   ├── four_fold_all_targets/             # Kết quả 4-Fold CV tổng hợp
│   ├── perday_all_models/                 # Kết quả dự báo theo ngày
│   ├── market_event_outputs/              # Kết quả kiểm định sự kiện thị trường
│   └── sensitivity_outputs/               # Kết quả phân tích độ nhạy
│
├── split_by_bank.py                       # Script tiền xử lý dữ liệu
└── requirements.txt                       # Thư viện Python cần cài đặt
```

---

## 📊 Dữ liệu

| Thông tin | Chi tiết |
|-----------|----------|
| **Ngân hàng** | BID (BIDV), CTG (VietinBank), VCB (Vietcombank) |
| **Giai đoạn** | 2016 – 2026 (~2.500 phiên giao dịch) |
| **Đặc trưng** | OHLCV + Chỉ báo kỹ thuật + Chỉ số vĩ mô (VN-Index, USD/VND, Lãi suất) |

---

## 🤖 Mô hình sử dụng

| Mô hình | Loại | Mục đích |
|---------|------|----------|
| **Naive** | Baseline | Dự báo bằng giá trị gần nhất |
| **XGBoost** | Gradient Boosting | Dự báo biến động (Volatility) |
| **NeuralProphet** | Facebook Time Series | Phát hiện mùa vụ (Seasonality) |
| **TFT** | Temporal Fusion Transformer | Biến động + Attention mechanism |
| **Hybrid** | GARCH + Ridge Regression | Dự báo giá (Price) |

---

## 🏆 Kết quả chính

| Mục tiêu | Mô hình tốt nhất | Cải thiện so với Naive |
|-----------|-------------------|------------------------|
| **Biến động (Volatility)** | TFT | +30–34% |
| **Giá (Price)** | Hybrid (GARCH + Ridge) | +35–41% |

---

## 🚀 Cài đặt & Chạy

### Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Chạy phân tích

```bash
# Phase 1 – Phân tích Khám phá
python langkinh1_xgboost_shap.py
python langkinh2_neuralprophet_seasonality.py
python langkinh3_tft_memory.py

# Phase 2 – Đánh giá Mô hình
python run_4fold_all_models_both_targets.py
python run_perday_5models.py
python run_market_event_validation.py
python run_sensitivity_analysis.py
```

---

## 📦 Dependencies

- Python 3.8+
- pandas, numpy, scikit-learn
- xgboost, shap
- neuralprophet
- pytorch (TFT)
- arch (GARCH)
- matplotlib, seaborn