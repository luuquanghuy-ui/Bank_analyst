# PHASE 2: VOLATILITY & RETURN PREDICTION
## Charts Documentation

---

## 1. `01_vol_ret_comparison.png`
### Volatility & Return MAE Comparison

**Mục đích:** So sánh MAE của 5 mô hình (Naive, XGBoost, NeuralProphet, TFT, Hybrid) trên cả 2 targets: Volatility và Return.

**Nội dung:**
- **Left panel (Volatility):** TFT thắng tất cả các bank (BID, CTG, VCB)
- **Right panel (Return):** Naive thắng tất cả các bank

**Replaces:** `four_fold_all_targets/4fold_vol_ret_comparison.png`

**Ý nghĩa:**
- Volatility: Models beat Naive → Volatility clustering is real
- Return: Naive beats Models → Martingale hypothesis holds

---

## 2. `02_tft_improvement.png`
### TFT Improvement over Naive Baseline

**Mục đích:** Định lượng mức độ cải thiện của TFT và XGBoost so với Naive baseline trên volatility prediction.

**Nội dung:** Bar chart showing percentage improvement over Naive
- TFT improvement: ~20-30%
- XGBoost improvement: ~10-15%

**Replaces:** (New chart - không có tương đương cũ)

**Ý nghĩa:** TFT provides meaningful improvement in volatility prediction

---

## 3. `03_market_event_analysis.png`
### Market Event Analysis: High Volatility Days

**Mục đích:** Kiểm tra model performance đặc biệt trên những ngày có biến động cao (top 20%).

**Nội dung:** So sánh MAE trên High Vol Days vs Normal Days cho cả Volatility và Return
- Row 1: Volatility - Models beat Naive cả 2 loại ngày
- Row 2: Return - Naive still wins cả 2 loại ngày

**Replaces:** `market_event_outputs_vol_ret/market_event_validation.png`

**Ý nghĩa:** Models có khả năng bắt volatility spikes tốt hơn Naive

---

## 4. `04_summary_table.png`
### Phase 2 Results Summary Table

**Mục đích:** Bảng tổng hợp ngắn gọn kết quả chính.

**Nội dung:** Bảng gồm:
- Bank
- Best model cho Volatility (TFT)
- Vol MAE của TFT
- % improvement so với Naive
- Best model cho Return (Naive)
- Return MAE của Naive

**Replaces:** (New chart - summary table)

**Ý nghĩa:** Nhìn nhanh kết quả chính

---

## 5. `05_honest_results.png`
### Why Results Are Valid

**Mục đích:** Giải thích tại sao kết quả là honest và có thể defend được.

**Nội dung:**
- Left: Volatility - TFT beats Naive vì volatility clustering tồn tại (ACF(|r|) ≈ 0.24)
- Right: Return - Naive wins vì martingale hypothesis đúng (ACF(r) ≈ 0)

**Replaces:** (New chart - không có tương đương cũ)

**Ý nghĩa:** Kết quả nhất quán với Phase 1 (ACF analysis)

---

## 6. `06_phase_consistency.png`
### Phase 1 & Phase 2 Consistency Check

**Mục đích:** Chứng minh Phase 1 và Phase 2 cho cùng một kết luận.

**Nội dung:** Bar chart xác nhận:
- ACF(Return) ≈ 0 → Martingale confirmed
- ACF(|Return|) ≈ 0.24 → Volatility clustering confirmed
- Models beat Naive on Volatility → Consistent
- Naive beats Models on Return → Consistent

**Replaces:** (New chart - không có tương đương cũ)

**Ý nghĩa:** Scientific consistency giữa 2 phases

---

## 7. `07_per_bank_volatility.png`
### Per-Bank Volatility Model Comparison

**Mục đích:** So sánh chi tiết từng ngân hàng trên volatility prediction.

**Nội dung:** 3 panels cho BID, CTG, VCB
- Mỗi bank: 5 bars cho 5 models
- Winner được highlight màu xanh

**Replaces:** (New chart - chi tiết hơn)

**Ý nghĩa:** TFT consistently wins across all banks

---

## 8. `08_tft_sensitivity.png`
### TFT Sensitivity: Hidden Size

**Mục đích:** Kiểm tra TFT performance với different hidden sizes (8, 16, 32).

**Nội dung:** Line charts
- Left: Hidden size vs Volatility MAE
- Right: Hidden size vs Return MAE

**Replaces:** `sensitivity_outputs_vol_ret/sensitivity_analysis_charts.png` (TFT part)

**Ý nghĩa:** Hidden size = 16 là optimal cho cả 2 targets

---

## 9. `09_xgb_sensitivity.png`
### XGBoost Sensitivity

**Mục đích:** Kiểm tra XGBoost performance với different hyperparameters.

**Nội dung:** Line charts
- Left: Max depth vs Volatility MAE
- Right: n_estimators vs Volatility MAE

**Replaces:** `sensitivity_outputs_vol_ret/sensitivity_analysis_charts.png` (XGB part)

**Ý nghĩa:** XGBoost robust across parameter ranges, depth=6, n_est=100 optimal

---

## 10. `10_hybrid_sensitivity.png`
### Hybrid Model Sensitivity: GARCH Weight

**Mục đích:** Kiểm tra Hybrid model (GARCH + Ridge) với different GARCH weights.

**Nội dung:** Line chart GARCH weight (0.0 to 1.0) vs Volatility MAE

**Replaces:** `sensitivity_outputs_vol_ret/sensitivity_analysis_charts.png` (Hybrid part)

**Ý nghĩa:** GARCH weight = 0.5 balanced cho hầu hết banks

---

## 11. `11_np_sensitivity.png`
### NeuralProphet Sensitivity: Learning Rate

**Mục đích:** Kiểm tra NeuralProphet với different learning rates.

**Nội dung:** Line charts
- Left: Learning rate vs Volatility MAE
- Right: Learning rate vs Return MAE

**Replaces:** `sensitivity_outputs_vol_ret/sensitivity_analysis_charts.png` (NP part)

**Ý nghĩa:** Learning rate = 0.01 stable và tốt

---

## 12. `12_key_takeaways.png`
### Key Conclusions

**Mục đích:** Tổng hợp tất cả key findings.

**Nội dung:** Text box với 5 key takeaways:
1. TFT wins volatility prediction (15-30% improvement)
2. Naive wins return prediction (martingale)
3. Models better on high vol days
4. Models are robust (sensitivity analysis)
5. Consistent with Phase 1 (ACF analysis)

**Replaces:** (New chart - summary)

**Ý nghĩa:** Nhìn nhanh toàn bộ kết luận

---

## OLD CHARTS TO REPLACE

| Old Chart (Price - FRAUD) | New Chart (Return/Vol - HONEST) |
|---------------------------|--------------------------------|
| `four_fold_all_targets/4fold_5models_comparison.png` | `01_vol_ret_comparison.png` |
| `market_event_outputs/market_event_validation.png` | `03_market_event_analysis.png` |
| `sensitivity_outputs/sensitivity_analysis_charts.png` | `08_tft_sensitivity.png`, `09_xgb_sensitivity.png`, `10_hybrid_sensitivity.png`, `11_np_sensitivity.png` |

**Lưu ý:** Các chart cũ dùng price prediction và có thể có leakage. Charts mới dùng return/volatility và đã được kiểm tra không có leakage.

---

## SUMMARY

Tất cả charts trong folder này sử dụng:
- **Target:** Return (martingale) và Volatility (clustering)
- **Validation:** 4-fold walk-forward
- **Baseline:** Naive (predict 0 for return, last vol for volatility)
- **No data leakage:** Features properly lagged with .shift(1)

Kết quả honest và có thể defend trước hội đồng.
