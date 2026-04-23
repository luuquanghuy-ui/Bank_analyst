# KIỂM TRA GIAN LẬN VÀ PHƯƠNG ÁN KHẮC PHỤC

## 1. TÓM TẮT CÁC VẤN ĐỀ

### Langkinh 1 (XGBoost + SHAP) - CÓ LEAKAGE ⚠️

#### Vấn đề Data Leakage:
```python
# Code cũ (LEAKAGE):
df['ma_ratio'] = (df['close'] / df['ma20']).shift(1)  # ma20 chứa close hôm nay!
df['volume_ratio'] = (df['volume'] / df['volume'].rolling(20).mean()).shift(1)  # rolling mean cũng chứa today
```

**Cơ chế leakage:**
- `target[t+1] = |log(close[t+1]/close[t])|` - phụ thuộc close[t]
- `ma_ratio[t+1] = close[t] / ma20[t]` - cũng phụ thuộc close[t]
- Model học được correlation giả tạo này → đánh bại Naive một cách giả tạo

#### Các vấn đề khác:
| Vấn đề | Mức độ | Chi tiết |
|---------|--------|----------|
| Không có walk-forward | ❌ Cao | Chỉ train 1 lần với 70% data |
| Không so sánh Naive | ❌ Cao | Không có baseline để đánh giá |
| Rankings không stable | ⚠️ Trung bình | Bài viết vs CSV không khớp |

---

## 2. LANGKINH 2 & 3 - KHÔNG CÓ LEAKAGE ✅

### Langkinh 2 (NeuralProphet Seasonality):
- Part A: NP decomposition - chỉ visualize trend/seasonality
- Part B: Statistical tests (Kruskal-Wallis, Mann-Whitney) - pure statistics
- **Không có leakage** vì không dùng cho prediction comparison

### Langkinh 3 (TFT + ACF Memory):
- Part A: ACF/PACF - pure time series analysis
- Part B: TFT attention - chỉ extract và visualize
- **Không có leakage** vì không dùng cho prediction comparison

---

## 3. PHƯƠNG ÁN SỬA LANGKINH 1

### Bước 1: Sửa Leakage (5 phút)
```python
# Cũ (LEAKAGE):
df['ma_ratio'] = (df['close'] / df['ma20']).shift(1)

# Mới (SẠCH):
# Cách 1: Shift đúng
df['ma20_shifted'] = df['ma20'].shift(1)
df['ma_ratio'] = (df['close'].shift(1) / df['ma20_shifted'])

# Cách 2: Bỏ hẳn vì volatility_20d đã capture rồi
```

### Bước 2: Thêm Naive Comparison (10 phút)
```python
# Thêm vào hàm train_and_shap():
naive_pred = np.full(len(y_test), y_train[-1])
naive_mae = mean_absolute_error(y_test, naive_pred)
print(f"  Model MAE: {mae:.6f}, Naive MAE: {naive_mae:.6f}")
```

### Bước 3: Walk-Forward 4 Folds (30 phút)
```python
fold_configs = [(0.50, 0.65, 0.70), (0.65, 0.80, 0.85), (0.80, 0.90, 0.95), (0.90, 0.95, 1.00)]
for fold_idx, (train_pct, val_pct, test_pct) in enumerate(fold_configs):
    # split, train, evaluate
```

### Bước 4: Diễn giải khiêm nhường hơn
- Không claim "Volume Ratio #1"
- Thay bằng "Các yếu tố kỹ thuật có ảnh hưởng nhất định"
- Thừa nhận uncertainty trong rankings

---

## 4. SAU FIX - KẾT QUẢ SẼ THAY ĐỔI

| Trước (gian lận) | Sau (honest) | Đánh giá |
|-------------------|--------------|----------|
| ma_ratio rank 1-2 | ma_ratio rank thấp hơn | ✅ Tốt hơn - phản ánh đúng |
| MAE thấp giả tạo | MAE gần Naive hơn | ✅ Tốt hơn - honest result |
| Rankings chắc chắn | Rankings có variance | ✅ Tốt hơn - thành thật |

---

## 5. KẾT LUẬN

### Về mặt "gian lận":
- ✅ Sau khi fix 3 bước trên → **KHÔNG còn gian lận methodology**

### Về mặt "kết quả":
- ❌ Kết quả sẽ kém "wow" hơn
- ✅ Nhưng **TRUNG THỰC** và **DEFENSIBLE** trước hội đồng

### Điều quan trọng nhất:
> **Trung thực > Ấn tượng**

Một kết quả khiêm nhường nhưng trung thực tốt hơn một kết quả ấn tượng nhưng có thể bị phát hiện và đánh trượt.
