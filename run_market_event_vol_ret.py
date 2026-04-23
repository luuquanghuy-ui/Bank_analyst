"""
Market Event Validation: 5 Models x 2 Targets

Target 1: VOLATILITY (legitimate - volatility clustering)
Target 2: RETURN (martingale-consistent)

Tests model performance on highest volatility days.
Uses 4-fold walk-forward validation on full dataset.

Models:
- Naive (baseline)
- XGBoost (core model 1)
- NeuralProphet (NP) (core model 2)
- TFT (core model 3)
- Hybrid = GARCH volatility signal + Ridge (volatility only)

Note: Hybrid only for volatility - GARCH predicts variance, not return.
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from arch import arch_model

import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from lightning.pytorch import Trainer

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
OUTPUT_DIR = BASE_DIR / "market_event_outputs_vol_ret"
OUTPUT_DIR.mkdir(exist_ok=True)

BANK_FILES = {
    "BID": BASE_DIR / "banks_BID_dataset.csv",
    "CTG": BASE_DIR / "banks_CTG_dataset.csv",
    "VCB": BASE_DIR / "banks_VCB_dataset.csv",
}

FOLD_CONFIGS = [
    (0.50, 0.65, 0.70),
    (0.65, 0.80, 0.85),
    (0.80, 0.90, 0.95),
    (0.90, 0.95, 1.00),
]

TOP_N_PCT = 0.20


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df.rename(columns={"date": "ds"})
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = np.abs(df["log_return"])
    return df.dropna().reset_index(drop=True)


def create_features(df):
    data = df.copy()
    data["volume_lag1"] = data["volume"].shift(1)
    data["volatility_5d"] = data["log_return"].rolling(5).std().shift(1)
    data["volatility_20d"] = data["log_return"].rolling(20).std().shift(1)
    data["rsi_lag1"] = data["rsi"].shift(1)
    data["return_lag1"] = data["log_return"].shift(1)
    data["return_lag2"] = data["log_return"].shift(2)
    data["return_lag5"] = data["log_return"].shift(5)
    return data.dropna().reset_index(drop=True)


# ===== NAIVE BASELINES =====

def naive_vol_walkforward(train_df, test_df):
    last_train_vol = train_df["volatility"].values[-1]
    return np.full(len(test_df), last_train_vol)


def naive_ret_walkforward(test_df):
    return np.zeros(len(test_df))


# ===== XGBOOST =====

def xgboost_vol_predict(train_df, test_df):
    feature_cols = ["volume_lag1", "volatility_5d", "volatility_20d", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5"]
    X_train = train_df[feature_cols].values
    y_train = train_df["volatility"].values
    X_test = test_df[feature_cols].values
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def xgboost_ret_predict(train_df, test_df):
    feature_cols = ["volume_lag1", "volatility_5d", "volatility_20d", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5"]
    X_train = train_df[feature_cols].values
    y_train = train_df["log_return"].values
    X_test = test_df[feature_cols].values
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    return model.predict(X_test)


# ===== NEURALPROPHET =====

def np_vol_predict(train_df, test_df):
    train_prophet = train_df[["ds", "volatility"]].copy()
    train_prophet.columns = ["ds", "y"]
    val_prophet = test_df[["ds", "volatility"]].copy()
    val_prophet.columns = ["ds", "y"]

    model = NeuralProphet(learning_rate=0.01, epochs=15, n_lags=10, n_forecasts=1, loss_func="MAE", weekly_seasonality=False)
    model.fit(train_prophet, freq="D", validation_df=val_prophet)
    predictions = model.predict(val_prophet)

    if "yhat1" in predictions.columns:
        pred_values = predictions["yhat1"].values
    elif "yhat" in predictions.columns:
        pred_values = predictions["yhat"].values
    else:
        pred_values = np.array([])

    result = np.full(len(test_df), np.nan)
    n = min(len(pred_values), len(test_df))
    if n > 0:
        result[:n] = pred_values[:n]

    result = np.nan_to_num(result, nan=0.0)
    if np.all(result == 0) and n == 0:
        result = np.full(len(test_df), train_df["volatility"].values[-1])

    return result


def np_ret_predict(train_df, test_df):
    train_prophet = train_df[["ds", "log_return"]].copy()
    train_prophet.columns = ["ds", "y"]
    val_prophet = test_df[["ds", "log_return"]].copy()
    val_prophet.columns = ["ds", "y"]

    model = NeuralProphet(learning_rate=0.01, epochs=15, n_lags=10, n_forecasts=1, loss_func="MAE", weekly_seasonality=False)
    model.fit(train_prophet, freq="D", validation_df=val_prophet)
    predictions = model.predict(val_prophet)

    if "yhat1" in predictions.columns:
        pred_values = predictions["yhat1"].values
    elif "yhat" in predictions.columns:
        pred_values = predictions["yhat"].values
    else:
        pred_values = np.array([])

    result = np.full(len(test_df), np.nan)
    n = min(len(pred_values), len(test_df))
    if n > 0:
        result[:n] = pred_values[:n]

    result = np.nan_to_num(result, nan=0.0)

    return result


# ===== TFT =====

def tft_vol_predict(train_df, test_df, bank="UNKNOWN"):
    lookback = 24
    train_d = train_df.copy()
    test_d = test_df.copy()

    train_d["time_idx"] = range(len(train_d))
    test_d["time_idx"] = range(len(train_d), len(train_d) + len(test_d))
    train_d["bank"] = bank
    test_d["bank"] = bank

    training = TimeSeriesDataSet(
        train_d,
        time_idx="time_idx",
        target="volatility",
        group_ids=["bank"],
        max_encoder_length=lookback,
        max_prediction_length=1,
        static_categoricals=["bank"],
        time_varying_unknown_reals=["volatility"],
        scalers={},
    )

    train_dataloader = training.to_dataloader(batch_size=64, num_workers=0, shuffle=False)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        learning_rate=0.001,
        optimizer="adam",
    )

    trainer = Trainer(max_epochs=15, accelerator="cpu", enable_progress_bar=False, logger=False)
    trainer.fit(tft, train_dataloader)

    train_tail = train_d.iloc[-lookback:].copy()
    pred_df = pd.concat([train_tail, test_d], ignore_index=True).copy().reset_index(drop=True)

    pred_dataset = TimeSeriesDataSet.from_dataset(training, pred_df)
    pred_dataloader = pred_dataset.to_dataloader(batch_size=64, num_workers=0, shuffle=False)

    raw_preds = tft.predict(pred_dataloader, mode="raw", return_x=True)
    all_preds = raw_preds[0].prediction.cpu().numpy()[:, 0, 3].tolist()

    return np.array(all_preds)


def tft_ret_predict(train_df, test_df, bank="UNKNOWN"):
    lookback = 24
    train_d = train_df.copy()
    test_d = test_df.copy()

    train_d["time_idx"] = range(len(train_d))
    test_d["time_idx"] = range(len(train_d), len(train_d) + len(test_d))
    train_d["bank"] = bank
    test_d["bank"] = bank

    training = TimeSeriesDataSet(
        train_d,
        time_idx="time_idx",
        target="log_return",
        group_ids=["bank"],
        max_encoder_length=lookback,
        max_prediction_length=1,
        static_categoricals=["bank"],
        time_varying_unknown_reals=["log_return"],
        scalers={},
    )

    train_dataloader = training.to_dataloader(batch_size=64, num_workers=0, shuffle=False)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        learning_rate=0.001,
        optimizer="adam",
    )

    trainer = Trainer(max_epochs=15, accelerator="cpu", enable_progress_bar=False, logger=False)
    trainer.fit(tft, train_dataloader)

    train_tail = train_d.iloc[-lookback:].copy()
    pred_df = pd.concat([train_tail, test_d], ignore_index=True).copy().reset_index(drop=True)

    pred_dataset = TimeSeriesDataSet.from_dataset(training, pred_df)
    pred_dataloader = pred_dataset.to_dataloader(batch_size=64, num_workers=0, shuffle=False)

    raw_preds = tft.predict(pred_dataloader, mode="raw", return_x=True)
    all_preds = raw_preds[0].prediction.cpu().numpy()[:, 0, 3].tolist()

    return np.array(np.nan_to_num(all_preds, nan=0.0))


# ===== HYBRID (VOLATILITY ONLY) =====

def garch_predict(train_ret, test_ret):
    ret_scaled = train_ret * 100.0
    model = arch_model(ret_scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    res = model.fit(disp="off", show_warning=False)
    mu = float(res.params.get("mu", res.params.get("Const", 0.0))) / 100.0
    omega = float(res.params["omega"])
    alpha = float(res.params["alpha[1]"])
    beta = float(res.params["beta[1]"])
    sigma2 = np.empty(len(test_ret))
    sigma2_last = max(float(np.var(ret_scaled)), 1e-6)
    eps_last = (train_ret[-1] - mu) * 100.0
    for i in range(len(test_ret)):
        sigma2[i] = omega + alpha * (eps_last ** 2) + beta * sigma2_last
        eps_last = (test_ret[i] - mu) * 100.0
        sigma2_last = sigma2[i]
    return np.sqrt(sigma2) / 100.0


def hybrid_vol_predict(train_df, test_df, train_ret, test_ret, garch_weight=0.5, ridge_alpha=1.0):
    garch_pred = garch_predict(train_ret, test_ret)
    feature_cols = ["volume_lag1", "volatility_5d", "volatility_20d", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5"]
    X_train = train_df[feature_cols].values
    y_train = train_df["volatility"].values
    X_test = test_df[feature_cols].values
    train_vol = np.abs(train_ret)
    X_train_with_vol = np.column_stack([X_train, train_vol])
    X_test_with_vol = np.column_stack([X_test, garch_pred])

    model = Ridge(alpha=ridge_alpha)
    model.fit(X_train_with_vol, y_train)
    ridge_pred = model.predict(X_test_with_vol)

    return garch_weight * garch_pred + (1 - garch_weight) * ridge_pred


# ===== MARKET EVENT VALIDATION =====

def run_market_event_validation():
    print("=" * 70)
    print("MARKET EVENT VALIDATION: VOLATILITY & RETURN")
    print("=" * 70)
    print("Using 4-fold walk-forward on full dataset (2016-2026)")
    print("=" * 70)

    all_results = []

    for bank, path in BANK_FILES.items():
        print(f"\n{'='*50}\n{bank}\n{'='*50}")

        df = load_data(path)
        df = create_features(df)
        n = len(df)

        fold_results = []

        for fold_idx, (train_pct, val_pct, test_pct) in enumerate(FOLD_CONFIGS):
            train_end = int(n * train_pct)
            val_end = int(n * val_pct)
            test_end = int(n * test_pct)

            if test_end - val_end < 30 or train_end < 500:
                continue

            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[val_end:test_end].copy()

            train_ret = train_df["log_return"].values
            test_ret = test_df["log_return"].values
            actual_vol = np.abs(test_ret)
            actual_ret = test_ret

            if len(test_df) < 30:
                continue

            print(f"\n  Fold {fold_idx+1}: test days={len(test_df)}")

            # VOLATILITY
            naive_vol_pred = naive_vol_walkforward(train_df, test_df)
            xgb_vol_pred = xgboost_vol_predict(train_df, test_df)
            
            try:
                np_vol_pred = np_vol_predict(train_df, test_df)
                np_vol_pred = np_vol_pred[:len(test_ret)]
            except:
                np_vol_pred = np.full(len(test_ret), np.nan)

            try:
                tft_vol_pred = tft_vol_predict(train_df, test_df, bank)
                tft_vol_pred = tft_vol_pred[:len(test_ret)]
            except:
                tft_vol_pred = np.full(len(test_ret), np.nan)

            try:
                hybrid_vol_pred = hybrid_vol_predict(train_df, test_df, train_ret, test_ret, 0.5)
            except:
                hybrid_vol_pred = np.full(len(test_ret), np.nan)

            # RETURN
            naive_ret_pred = naive_ret_walkforward(test_df)
            xgb_ret_pred = xgboost_ret_predict(train_df, test_df)
            
            try:
                np_ret_pred = np_ret_predict(train_df, test_df)
                np_ret_pred = np_ret_pred[:len(test_ret)]
            except:
                np_ret_pred = np.full(len(test_ret), np.nan)

            try:
                tft_ret_pred = tft_ret_predict(train_df, test_df, bank)
                tft_ret_pred = tft_ret_pred[:len(test_ret)]
            except:
                tft_ret_pred = np.full(len(test_ret), np.nan)

            # IDENTIFY HIGH VOLATILITY DAYS
            n_high = max(5, int(len(test_ret) * TOP_N_PCT))
            sorted_indices = np.argsort(actual_vol)[::-1]
            high_vol_indices = sorted_indices[:n_high]
            normal_indices = sorted_indices[n_high:]

            vol_threshold = actual_vol[high_vol_indices[-1]]

            print(f"    High vol threshold: {vol_threshold:.4f} ({n_high} days)")

            # CALCULATE MAE ON HIGH VOL VS NORMAL DAYS
            # Volatility
            naive_high_vol = mean_absolute_error(actual_vol[high_vol_indices], naive_vol_pred[high_vol_indices])
            naive_normal_vol = mean_absolute_error(actual_vol[normal_indices], naive_vol_pred[normal_indices])

            xgb_high_vol = mean_absolute_error(actual_vol[high_vol_indices], xgb_vol_pred[high_vol_indices])
            xgb_normal_vol = mean_absolute_error(actual_vol[normal_indices], xgb_vol_pred[normal_indices])

            np_high_vol = mean_absolute_error(actual_vol[high_vol_indices], np_vol_pred[high_vol_indices])
            np_normal_vol = mean_absolute_error(actual_vol[normal_indices], np_vol_pred[normal_indices])

            tft_high_vol = mean_absolute_error(actual_vol[high_vol_indices], tft_vol_pred[high_vol_indices])
            tft_normal_vol = mean_absolute_error(actual_vol[normal_indices], tft_vol_pred[normal_indices])

            hybrid_high_vol = mean_absolute_error(actual_vol[high_vol_indices], hybrid_vol_pred[high_vol_indices])
            hybrid_normal_vol = mean_absolute_error(actual_vol[normal_indices], hybrid_vol_pred[normal_indices])

            # Return
            naive_high_ret = mean_absolute_error(actual_ret[high_vol_indices], naive_ret_pred[high_vol_indices])
            naive_normal_ret = mean_absolute_error(actual_ret[normal_indices], naive_ret_pred[normal_indices])

            xgb_high_ret = mean_absolute_error(actual_ret[high_vol_indices], xgb_ret_pred[high_vol_indices])
            xgb_normal_ret = mean_absolute_error(actual_ret[normal_indices], xgb_ret_pred[normal_indices])

            np_high_ret = mean_absolute_error(actual_ret[high_vol_indices], np_ret_pred[high_vol_indices])
            np_normal_ret = mean_absolute_error(actual_ret[normal_indices], np_ret_pred[normal_indices])

            tft_high_ret = mean_absolute_error(actual_ret[high_vol_indices], tft_ret_pred[high_vol_indices])
            tft_normal_ret = mean_absolute_error(actual_ret[normal_indices], tft_ret_pred[normal_indices])

            print(f"    VOL: Naive={naive_high_vol:.4f}/{naive_normal_vol:.4f}, XGB={xgb_high_vol:.4f}/{xgb_normal_vol:.4f}, NP={np_high_vol:.4f}/{np_normal_vol:.4f}, TFT={tft_high_vol:.4f}/{tft_normal_vol:.4f}, Hybrid={hybrid_high_vol:.4f}/{hybrid_normal_vol:.4f}")
            print(f"    RET: Naive={naive_high_ret:.4f}/{naive_normal_ret:.4f}, XGB={xgb_high_ret:.4f}/{xgb_normal_ret:.4f}, NP={np_high_ret:.4f}/{np_normal_ret:.4f}, TFT={tft_high_ret:.4f}/{tft_normal_ret:.4f}, Hybrid=N/A")

            fold_results.append({
                "fold": fold_idx + 1,
                "n_test_days": len(test_df),
                "n_high_vol_days": n_high,
                "vol_threshold": vol_threshold,
                # Volatility - high vol days
                "naive_high_vol": naive_high_vol,
                "xgb_high_vol": xgb_high_vol,
                "np_high_vol": np_high_vol,
                "tft_high_vol": tft_high_vol,
                "hybrid_high_vol": hybrid_high_vol,
                # Volatility - normal days
                "naive_normal_vol": naive_normal_vol,
                "xgb_normal_vol": xgb_normal_vol,
                "np_normal_vol": np_normal_vol,
                "tft_normal_vol": tft_normal_vol,
                "hybrid_normal_vol": hybrid_normal_vol,
                # Return - high vol days
                "naive_high_ret": naive_high_ret,
                "xgb_high_ret": xgb_high_ret,
                "np_high_ret": np_high_ret,
                "tft_high_ret": tft_high_ret,
                # Return - normal days
                "naive_normal_ret": naive_normal_ret,
                "xgb_normal_ret": xgb_normal_ret,
                "np_normal_ret": np_normal_ret,
                "tft_normal_ret": tft_normal_ret,
            })

        if fold_results:
            df_folds = pd.DataFrame(fold_results)
            avg = df_folds.mean(numeric_only=True)

            print(f"\n  AVERAGE ({len(fold_results)} folds):")
            print(f"    VOLATILITY (High Vol Days):")
            print(f"      Naive:   {avg['naive_high_vol']:.6f}")
            print(f"      XGBoost: {avg['xgb_high_vol']:.6f}")
            print(f"      NP:      {avg['np_high_vol']:.6f}")
            print(f"      TFT:     {avg['tft_high_vol']:.6f}")
            print(f"      Hybrid:  {avg['hybrid_high_vol']:.6f}")
            print(f"    VOLATILITY (Normal Days):")
            print(f"      Naive:   {avg['naive_normal_vol']:.6f}")
            print(f"      XGBoost: {avg['xgb_normal_vol']:.6f}")
            print(f"      NP:      {avg['np_normal_vol']:.6f}")
            print(f"      TFT:     {avg['tft_normal_vol']:.6f}")
            print(f"      Hybrid:  {avg['hybrid_normal_vol']:.6f}")
            print(f"    RETURN (High Vol Days):")
            print(f"      Naive:   {avg['naive_high_ret']:.6f}")
            print(f"      XGBoost: {avg['xgb_high_ret']:.6f}")
            print(f"      NP:      {avg['np_high_ret']:.6f}")
            print(f"      TFT:     {avg['tft_high_ret']:.6f}")
            print(f"    RETURN (Normal Days):")
            print(f"      Naive:   {avg['naive_normal_ret']:.6f}")
            print(f"      XGBoost: {avg['xgb_normal_ret']:.6f}")
            print(f"      NP:      {avg['np_normal_ret']:.6f}")
            print(f"      TFT:     {avg['tft_normal_ret']:.6f}")

            df_folds.to_csv(OUTPUT_DIR / f"high_vol_days_{bank}.csv", index=False)

            all_results.append({
                "bank": bank,
                "n_folds": len(fold_results),
                # High vol days
                "avg_naive_high_vol": avg["naive_high_vol"],
                "avg_xgb_high_vol": avg["xgb_high_vol"],
                "avg_np_high_vol": avg["np_high_vol"],
                "avg_tft_high_vol": avg["tft_high_vol"],
                "avg_hybrid_high_vol": avg["hybrid_high_vol"],
                # Normal days
                "avg_naive_normal_vol": avg["naive_normal_vol"],
                "avg_xgb_normal_vol": avg["xgb_normal_vol"],
                "avg_np_normal_vol": avg["np_normal_vol"],
                "avg_tft_normal_vol": avg["tft_normal_vol"],
                "avg_hybrid_normal_vol": avg["hybrid_normal_vol"],
                # Return high vol
                "avg_naive_high_ret": avg["naive_high_ret"],
                "avg_xgb_high_ret": avg["xgb_high_ret"],
                "avg_np_high_ret": avg["np_high_ret"],
                "avg_tft_high_ret": avg["tft_high_ret"],
                # Return normal
                "avg_naive_normal_ret": avg["naive_normal_ret"],
                "avg_xgb_normal_ret": avg["xgb_normal_ret"],
                "avg_np_normal_ret": avg["np_normal_ret"],
                "avg_tft_normal_ret": avg["tft_normal_ret"],
            })

    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(OUTPUT_DIR / "market_event_summary.csv", index=False)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, result in enumerate(all_results):
        bank = result["bank"]

        # Volatility: High Vol vs Normal
        ax = axes[0, idx]
        models = ["Naive", "XGBoost", "NP", "TFT", "Hybrid"]
        high_vol_mves = [result["avg_naive_high_vol"], result["avg_xgb_high_vol"],
                        result["avg_np_high_vol"], result["avg_tft_high_vol"], result["avg_hybrid_high_vol"]]
        normal_mves = [result["avg_naive_normal_vol"], result["avg_xgb_normal_vol"],
                      result["avg_np_normal_vol"], result["avg_tft_normal_vol"], result["avg_hybrid_normal_vol"]]

        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width/2, high_vol_mves, width, label="High Vol Days", color="red", alpha=0.7)
        ax.bar(x + width/2, normal_mves, width, label="Normal Days", color="blue", alpha=0.7)
        ax.set_ylabel("MAE")
        ax.set_title(f"{bank}: Volatility MAE")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend(fontsize=8)

        # Return: High Vol vs Normal
        ax = axes[1, idx]
        high_ret_mves = [result["avg_naive_high_ret"], result["avg_xgb_high_ret"],
                        result["avg_np_high_ret"], result["avg_tft_high_ret"], np.nan]
        normal_ret_mves = [result["avg_naive_normal_ret"], result["avg_xgb_normal_ret"],
                          result["avg_np_normal_ret"], result["avg_tft_normal_ret"], np.nan]

        ax.bar(x - width/2, high_ret_mves, width, label="High Vol Days", color="red", alpha=0.7)
        ax.bar(x + width/2, normal_ret_mves, width, label="Normal Days", color="blue", alpha=0.7)
        ax.set_ylabel("MAE")
        ax.set_title(f"{bank}: Return MAE")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "market_event_validation.png", dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print("MARKET EVENT VALIDATION COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print("=" * 70)

    return df_summary


if __name__ == "__main__":
    run_market_event_validation()