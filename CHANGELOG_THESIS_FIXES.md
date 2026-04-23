# CHANGELOG: Thesis Fixes (4/23/2026)

## PHASE 1 (Langkinh 1) - XGBoost + SHAP

### Files Changed
| File | Status | Description |
|------|--------|-------------|
| `langkinh1_xgboost_shap.py` | OLD | Original with data leakage |
| `langkinh1_xgboost_shap_fixed.py` | NEW | Fixed version (use this) |
| `langkinh1_final_charts/` | NEW | 19 charts generated from fixed script |

### Leakage Issues Fixed

#### 1. REMOVED: `ma_ratio` (Price / MA20)
```python
# OLD (LEAKAGE) - line 131:
df['ma_ratio'] = (df['close'] / df['ma20']).shift(1)
# Problem: ma20 includes today's close → correlation with target is artificial

# NEW: Feature REMOVED entirely
# Reason: volatility_20d already captures volatility dynamics
```

#### 2. REMOVED: `ma50_ratio` (Price / MA50)
```python
# OLD (LEAKAGE) - line 132:
df['ma50_ratio'] = (df['close'] / df['ma50']).shift(1)
# Problem: ma50 includes today's close → same leakage as ma_ratio

# NEW: Feature REMOVED entirely
```

#### 3. REMOVED: `volume_ratio`
```python
# OLD (LEAKAGE) - line 128:
df['volume_ratio'] = (df['volume'] / df['volume'].rolling(20).mean()).shift(1)
# Problem: rolling(20).mean() includes today → artificial correlation

# NEW: Feature REMOVED entirely
# Reason: volume_lag1 is sufficient for volume dynamics
```

### Feature List Comparison

| Feature | Old (Leaky) | New (Fixed) | Reason |
|---------|-------------|-------------|--------|
| return_lag1-5 | ✅ | ✅ | Properly lagged |
| volatility_lag1-2 | ✅ | ✅ | Properly lagged |
| rsi_lag1 | ✅ | ✅ | Properly lagged |
| volume_lag1 | ✅ | ✅ | Properly lagged |
| volume_ratio | ❌ REMOVED | - | Rolling includes today |
| ma_ratio | ❌ REMOVED | - | MA includes today |
| ma50_ratio | ❌ REMOVED | - | MA includes today |
| vnindex_lag1 | ✅ | ✅ | Properly lagged |
| vn30_lag1 | ✅ | ✅ | Properly lagged |
| usd_vnd_lag1 | ✅ | ✅ | Properly lagged |
| interest_rate_lag1 | ✅ | ✅ | Properly lagged |

**Total Features: 15 → 12** (removed 3 leaky features)

### Other Improvements in Fixed Version

1. **Added Naive Baseline Comparison**
   - New: Naive MAE shown alongside model MAE
   - Honest comparison: "Model loses to Naive for return"

2. **Added 4-Fold Walk-Forward Validation**
   ```python
   # OLD: Single 70/15/15 split
   train_end = int(n * 0.70)

   # NEW: 4-fold walk-forward
   fold_configs = [(0.50, 0.65, 0.70), (0.65, 0.80, 0.85), (0.80, 0.90, 0.95), (0.90, 0.95, 1.00)]
   ```

3. **Stable Feature Rankings**
   - OLD: Rankings varied between article and CSV
   - NEW: Consistent rankings with variance shown

### Results After Fix

| Bank | XGB MAE | Naive MAE | vs Naive |
|------|---------|-----------|----------|
| BID | 0.0156 | 0.0126 | -31.7% (loses) |
| CTG | 0.0151 | 0.0127 | -24.2% (loses) |
| VCB | 0.0105 | 0.0107 | +0.5% (ties) |

**Interpretation: Return is unpredictable (martingale holds) ✅**

---

## PHASE 2 - Volatility & Return Prediction

### Files Changed
| File | Status | Description |
|------|--------|-------------|
| `run_4fold_vol_ret.py` | ORIGINAL | Core Phase 2 script |
| `generate_phase2_charts.py` | NEW | Chart generation script |
| `phase2_final_charts/` | NEW | 12 charts |

### Phase 2 Status: NO LEAKAGE ✅

Phase 2 scripts (`run_4fold_vol_ret.py`, `run_perday_vol_ret.py`, `run_sensitivity_vol_ret.py`, `run_market_event_vol_ret.py`) were already clean:

```python
# Phase 2 features (CORRECT - no leakage):
feature_cols = ["volume_lag1", "volatility_5d", "volatility_20d", "rsi_lag1",
                "return_lag1", "return_lag2", "return_lag5"]
# All features properly lagged with .shift(1)
```

### Phase 2 Results

#### Volatility Prediction (TFT wins - volatility clustering exists)
| Bank | Naive | XGBoost | NP | TFT | Hybrid |
|------|-------|---------|-----|-----|-------|
| BID | 0.0126 | 0.0111 | 0.0119 | **0.0107** | 0.0110 |
| CTG | 0.0127 | 0.0111 | 0.0120 | **0.0107** | 0.0111 |
| VCB | 0.0107 | 0.0099 | 0.0104 | **0.0096** | 0.0098 |

#### Return Prediction (Naive wins - martingale holds)
| Bank | Naive | XGBoost | NP | TFT |
|------|-------|---------|-----|-----|
| BID | **0.0156** | 0.0165 | 0.0161 | 0.0164 |
| CTG | **0.0151** | 0.0156 | 0.0155 | 0.0156 |
| VCB | **0.0105** | 0.0109 | 0.0107 | 0.0109 |

---

## Summary of All Changes

### Phase 1 (Langkinh 1)
- ❌ Removed 3 leaky features (ma_ratio, ma50_ratio, volume_ratio)
- ✅ Added Naive baseline comparison
- ✅ Added 4-fold walk-forward validation
- ✅ Created honest, defensible results

### Phase 2
- ✅ No changes needed (already clean)
- ✅ Created chart generation script
- ✅ Generated 12 charts in `phase2_final_charts/`

### Phase 3 (Langkinh 2 & 3)
- ✅ No leakage (pure analysis, not prediction)

---

## Charts Generated

### Langkinh 1 Final Charts (`langkinh1_final_charts/`)
- `naive_comparison.png` - Model vs Naive baseline
- `partA_feature_bars.png` - Technical vs Macro importance
- `partA_group_comparison.png` - Cross-bank group comparison
- `partB_crossbank_comparison.csv` - Feature rankings across banks
- `partB_crossbank_heatmap.png` - Heatmap of rankings
- `partB_radar_dna.png` - Cross-bank DNA radar chart
- `BID/CTG/VCB_shap_summary.png` - SHAP summary per bank
- `BID/CTG/VCB_partC_top5_dependence.png` - Top 5 feature dependence
- `BID/CTG/VCB_feature_importance.csv` - Feature importance CSV
- `BID/CTG/VCB_group_importance.csv` - Group importance CSV

### Phase 2 Final Charts (`phase2_final_charts/`)
- `01_vol_ret_comparison.png` - Volatility & Return comparison
- `02_tft_improvement.png` - TFT improvement over Naive
- `03_market_event.png` - Market event analysis
- `04_summary_table.png` - Summary table
- `05_honest_results.png` - Honest results
- `06_phase_consistency.png` - Phase consistency
- `07_per_bank.png` - Per-bank analysis
- `08_tft_sensitivity.png` - TFT sensitivity
- `09_xgb_sensitivity.png` - XGBoost sensitivity
- `10_hybrid_sensitivity.png` - Hybrid sensitivity
- `11_np_sensitivity.png` - NeuralProphet sensitivity
- `12_key_takeaways.png` - Key takeaways
