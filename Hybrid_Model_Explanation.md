# Hybrid Model: GARCH + Ridge

## Mô hình

**Hybrid = GARCH(1,1) + Ridge Regression**

- **GARCH**: Dự báo volatility (phương sai) từ quá khứ
- **Ridge Regression**: Feature-based prediction
- **Hybrid**: Kết hợp GARCH signal với ML features

## Tại sao chỉ dùng cho Volatility?

### Lý thuyết tài chính

| Target | GARCH dự báo | Phù hợp? |
|--------|--------------|-----------|
| **Volatility** | Phương sai (variance) | ✅ Đúng |
| **Price** | Không dự báo được | ❌ Sai |

### Giải thích

**Volatility có tính bầy đàn (Clustering):**
- Hôm nay volatility cao → Ngày mai volatility có xu hướng cao
- GARCH bắt được pattern này qua conditional variance
- → **Hybrid cho volatility CÓ economic intuition**

**Price là Martingale:**
- Giá không có trend dự báo được
- GARCH dự báo volatility, KHÔNG phải giá
- Volatility cao = độ lớn thay đổi lớn, nhưng GIÁ tăng hay giảm?
- → **Không có câu trả lời** → Hybrid cho price THIẾU economic intuition

### Ví dụ

```
Ngày mai:
- Volatility cao → Đúng: GARCH predict tốt
- Price tăng → Sai: GARCH không cho biết hướng
```

## Vai trò trong đồ án

1. **Benchmark cho Volatility**: GARCH là "tượng đài" kinh điển của econometrics
2. **So sánh với TFT**: Nếu TFT thắng Hybrid → AI vượt phương pháp truyền thống
3. **Economic intuition**: Thể hiện sự hiểu biết về cả Kinh tế lượng lẫn Machine Learning

## Code

```python
def hybrid_vol_walkforward(train_df, test_df, train_ret, test_ret, garch_weight=0.5):
    """Hybrid cho volatility: GARCH + Ridge"""
    garch_pred = garch_walkforward(train_ret, test_ret)

    # Ridge với GARCH volatility làm feature
    X_train = train_df[feature_cols].values
    X_train_with_vol = np.column_stack([X_train, np.abs(train_ret)])
    X_test_with_vol = np.column_stack([test_df[feature_cols].values, garch_pred])

    model = Ridge(alpha=1.0)
    model.fit(X_train_with_vol, y_train)
    ridge_pred = model.predict(X_test_with_vol)

    return garch_weight * garch_pred + (1 - garch_weight) * ridge_pred
```

## Kết luận

| | Volatility | Price |
|--|-----------|-------|
| **Hybrid (GARCH+Ridge)** | ✅ Hợp lệ | ❌ Không hợp lệ |
| **Lý do** | Volatility clustering là thực | Price = Martingale, GARCH không dự báo được |

> **Note**: Hybrid cho price bị LOẠI BOỎ khỏi Phase 2 vì thiếu economic intuition theo đúng ý kiến chuyên gia.
