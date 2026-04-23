# LĂNG KÍNH 1: XGBoost + SHAP — Feature Importance
## Charts Documentation

---

## Tổng Quan

**Mục tiêu:** Phân tích feature importance để trả lời câu hỏi "Cái gì chi phối mức độ biến động ngân hàng VN?"

**Target:** `|log_return|` (absolute daily return, proxy cho volatility)

**Model:** XGBoost + SHAP

**Features:** 12 features (8 Technical + 4 Macro) — đã loại bỏ 3 leaky features

**Validation:** 4-fold walk-forward

---

## 1. `naive_comparison.png`
### Model vs Naive Baseline Comparison

**Mục đích:** So sánh XGBoost với Naive baseline (predict last known volatility)

**Nội dung:** Bar chart cho 3 banks (BID, CTG, VCB), mỗi bank có 2 bars:
- XGBoost MAE
- Naive MAE

**Replaces:** (Chart mới - không có tương đương cũ)

**Ý nghĩa:** Cho thấy model có thực sự tốt hơn baseline không

---

## 2. `partA_group_comparison.png`
### Technical vs Macro Feature Group Importance

**Mục đích:** So sánh tổng SHAP importance giữa Technical và Macro groups

**Nội dung:** 3 pie charts cho BID, CTG, VCB
- Technical (màu xanh lá)
- Macro (màu xanh dương)

**Replaces:** `langkinh1_xgboost_shap/partA_group_comparison.png` (cũ)

**Ý nghĩa:**
- Technical features chiếm ~55-60% importance
- Macro features chiếm ~40-45% importance
- Technical features có ảnh hưởng nhất định đến volatility

---

## 3. `partA_feature_bars.png`
### Individual Feature Importance (All 12 Features)

**Mục đích:** Xem chi tiết importance của từng feature

**Nội dung:** 3 horizontal bar charts (1 cho mỗi bank)
- Tất cả 12 features được xếp theo SHAP importance
- Màu xanh lá = Technical, Màu xanh dương = Macro

**Replaces:** `langkinh1_xgboost_shap/partA_feature_bars.png` (cũ)

**Ý nghĩa:** Cho biết feature nào quan trọng nhất cho từng bank

---

## 4. `partB_crossbank_heatmap.png`
### Cross-Bank Feature Ranking Heatmap

**Mục đích:** So sánh feature rankings giữa 3 banks

**Nội dung:** Heatmap với:
- Rows = 12 features
- Columns = BID, CTG, VCB
- Màu đậm = rank cao (quan trọng), Màu nhạt = rank thấp

**Replaces:** `langkinh1_xgboost_shap/partB_crossbank_heatmap.png` (cũ)

**Ý nghĩa:** Cho thấy:
- VNIndex, VN30, USD/VND rank cao ở tất cả banks
- Volatility lags quan trọng ở một số banks

---

## 5. `partB_radar_dna.png`
### Cross-Bank DNA Radar Chart

**Mục đích:** So sánh normalized SHAP profiles giữa 3 banks

**Nội dung:** Radar chart với 8 top features

**Replaces:** `langkinh1_xgboost_shap/partB_radar_dna.png` (cũ)

**Ý nghĩa:** Mỗi bank có "DNA" riêng về feature importance

---

## 6. `BID_shap_summary.png` / `CTG_shap_summary.png` / `VCB_shap_summary.png`
### SHAP Summary Beeswarm Plot

**Mục đích:** Xem phân bố SHAP values cho từng feature

**Nội dung:** Beeswarm plot cho mỗi bank
- Mỗi dot = một prediction
- Vị trí trục x = SHAP value (ảnh hưởng đến prediction)
- Màu sắc = giá trị feature

**Replaces:** `langkinh1_xgboost_shap/BID_shap_summary.png` (v.v.) (cũ)

**Ý nghĩa:** Cho thấy:
- Feature nào có ảnh hưởng lớn nhất
- Chiều tác động (tăng/giảm volatility)

---

## 7. `BID_partC_top5_dependence.png` / `CTG_partC_top5_dependence.png` / `VCB_partC_top5_dependence.png`
### Top 5 Feature Dependence Plots

**Mục đích:** Xem mối quan hệ giữa top 5 features và SHAP values

**Nội dung:** 5 scatter plots cho mỗi bank
- X-axis = giá trị feature
- Y-axis = SHAP value

**Replaces:** `langkinh1_xgboost_shap/BID_partC_top5_dependence.png` (v.v.) (cũ)

**Ý nghĩa:** Cho thấy feature effect có tuyến tính hay phi tuyến

---

## OLD CHARTS TO REPLACE

| Old Chart | New Chart |
|-----------|-----------|
| `langkinh1_xgboost_shap/partA_group_comparison.png` | `partA_group_comparison.png` ✅ |
| `langkinh1_xgboost_shap/partA_feature_bars.png` | `partA_feature_bars.png` ✅ |
| `langkinh1_xgboost_shap/partB_crossbank_heatmap.png` | `partB_crossbank_heatmap.png` ✅ |
| `langkinh1_xgboost_shap/partB_radar_dna.png` | `partB_radar_dna.png` ✅ |
| `langkinh1_xgboost_shap/BID_shap_summary.png` | `BID_shap_summary.png` ✅ |
| `langkinh1_xgboost_shap/CTG_shap_summary.png` | `CTG_shap_summary.png` ✅ |
| `langkinh1_xgboost_shap/VCB_shap_summary.png` | `VCB_shap_summary.png` ✅ |
| (Không có) | `naive_comparison.png` ✅ (MỚI) |

---

## IMPORTANT NOTES

### Về Leakage:
- **ĐÃ LOẠI:** ma_ratio, ma50_ratio, volume_ratio (do leakage)
- **SỬ DỤNG:** 12 features sạch

### Về Naive Comparison:
- Model không always beat Naive
- Một số folds/banks, Naive tốt hơn
- Điều này **TRUNG THỰC** với thực tế

### Về 4-Fold Validation:
- Kết quả có thể khác nhau giữa các folds
- Điều này cho thấy **uncertainty** trong feature importance
- Không nên claim feature X là #1 một cách chắc chắn

---

## SUMMARY

Tất cả charts trong folder này sử dụng:
- **Target:** |log_return| (volatility proxy)
- **Features:** 12 (không leakage)
- **Validation:** 4-fold walk-forward
- **Baseline:** Naive comparison
- **Method:** XGBoost + SHAP

Kết quả **trung thực** và **defensible** trước hội đồng.
