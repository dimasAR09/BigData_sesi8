"""
Module 3: Sales Forecasting
============================
Meramalkan jumlah penjualan untuk 3 bulan ke depan (90 hari).

Pendekatan  : Polynomial Regression + Rolling Features (pendekatan regresi)
Alasan regresi vs klasifikasi:
  - Forecasting menghasilkan nilai KONTINU (jumlah penjualan besok = Rp X)
  - Klasifikasi hanya menghasilkan kategori ("tinggi/rendah") → kehilangan info
  - Regresi memetakan fitur waktu → nilai numerik penjualan secara langsung
  - Lebih cocok dievaluasi dengan MAE/RMSE bukan akurasi

Tantangan utama data deret waktu:
  - Autokorelasi: nilai hari ini dipengaruhi hari sebelumnya
  - Musiman/seasonality: pola berulang (akhir pekan, harbolnas, dll.)
  - Non-stasioneritas: mean/variance berubah seiring waktu
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
HORIZON = 90   # 3 bulan ke depan


def rekayasa_fitur_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah fitur lag, rolling, dan fitur kalender."""
    df = df.copy().sort_values("tgl_transaksi").reset_index(drop=True)
    df["tgl_transaksi"] = pd.to_datetime(df["tgl_transaksi"])

    # Target: total penjualan harian
    y_col = "total_penjualan"

    # Fitur lag (hari sebelumnya)
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = df[y_col].shift(lag)

    # Rolling mean & std
    for window in [7, 14, 30]:
        df[f"roll_mean_{window}"] = df[y_col].shift(1).rolling(window).mean()
        df[f"roll_std_{window}"]  = df[y_col].shift(1).rolling(window).std()

    # Fitur tren (hari ke berapa)
    df["tren"] = df["hari_ke"].astype(float)

    # Fitur kalender
    df["sin_hari_tahun"] = np.sin(2 * np.pi * df["tgl_transaksi"].dt.dayofyear / 365)
    df["cos_hari_tahun"] = np.cos(2 * np.pi * df["tgl_transaksi"].dt.dayofyear / 365)
    df["sin_hari_minggu"] = np.sin(2 * np.pi * df["hari_minggu"] / 7)
    df["cos_hari_minggu"] = np.cos(2 * np.pi * df["hari_minggu"] / 7)

    df = df.dropna().reset_index(drop=True)
    return df


FITUR_TS = [
    "tren", "bulan", "is_weekend", "is_harbolnas",
    "lag_1", "lag_7", "lag_14", "lag_30",
    "roll_mean_7", "roll_mean_14", "roll_mean_30",
    "roll_std_7", "roll_std_14",
    "sin_hari_tahun", "cos_hari_tahun",
    "sin_hari_minggu", "cos_hari_minggu",
]


def latih_model(df: pd.DataFrame):
    df = rekayasa_fitur_ts(df)
    X = df[FITUR_TS].values
    y = df["total_penjualan"].values

    # TimeSeriesSplit (tidak boleh acak untuk data time-series)
    tscv  = TimeSeriesSplit(n_splits=5)
    mae_list, rmse_list, r2_list = [], [], []

    # Pipeline: Polynomial degree 2 + Ridge Regression
    pipe = Pipeline([
        ("poly",    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
        ("scaler",  StandardScaler()),
        ("ridge",   Ridge(alpha=10.0)),
    ])

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
        pipe.fit(X[tr_idx], y[tr_idx])
        y_pred = pipe.predict(X[va_idx])
        y_pred = np.clip(y_pred, 0, None)   # penjualan tidak bisa negatif
        mae_list.append(mean_absolute_error(y[va_idx], y_pred))
        rmse_list.append(np.sqrt(mean_squared_error(y[va_idx], y_pred)))
        r2_list.append(r2_score(y[va_idx], y_pred))

    # Latih final pada semua data
    pipe.fit(X, y)
    y_fitted = np.clip(pipe.predict(X), 0, None)

    return pipe, df, y_fitted, mae_list, rmse_list, r2_list


def buat_forecast(pipe, df_hist: pd.DataFrame) -> pd.DataFrame:
    """Forecast 90 hari ke depan secara iteratif."""
    df_ext = df_hist.copy()
    tgl_terakhir = df_ext["tgl_transaksi"].max()
    hari_ke_terakhir = df_ext["hari_ke"].max()

    prediksi = []
    for i in range(1, HORIZON + 1):
        tgl_baru = tgl_terakhir + pd.Timedelta(days=i)

        # Buat baris baru
        baris = {
            "tgl_transaksi" : tgl_baru,
            "total_penjualan": np.nan,
            "hari_ke"        : hari_ke_terakhir + i,
            "bulan"          : tgl_baru.month,
            "hari_minggu"    : tgl_baru.dayofweek,
            "is_weekend"     : int(tgl_baru.dayofweek >= 5),
            "is_harbolnas"   : int(tgl_baru.month == 12 and tgl_baru.day == 12),
        }

        # Fitur lag dari historis + prediksi sebelumnya
        semua = pd.concat([
            df_ext[["tgl_transaksi", "total_penjualan", "hari_ke",
                     "bulan", "hari_minggu", "is_weekend", "is_harbolnas"]],
            pd.DataFrame([baris])
        ], ignore_index=True)
        semua = rekayasa_fitur_ts(semua)

        if len(semua) == 0:
            prediksi.append(0)
            continue

        baris_feat = semua.iloc[[-1]][FITUR_TS].values
        pred = float(np.clip(pipe.predict(baris_feat), 0, None)[0])
        prediksi.append(pred)

        # Masukkan ke df_ext untuk lag berikutnya
        baris["total_penjualan"] = pred
        df_ext = pd.concat([df_ext, pd.DataFrame([baris])], ignore_index=True)

    df_forecast = pd.DataFrame({
        "tgl_forecast"  : [tgl_terakhir + pd.Timedelta(days=i) for i in range(1, HORIZON + 1)],
        "penjualan_pred": prediksi,
    })
    return df_forecast


def simpan_visualisasi(df_hist: pd.DataFrame, y_fitted: np.ndarray,
                       df_forecast: pd.DataFrame, mae_list, rmse_list, r2_list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle("Modul 3: Forecasting Penjualan 3 Bulan ke Depan (Polynomial Ridge Regression)",
                 fontsize=13, fontweight="bold")

    # ── Plot 1: Historis vs Fitted ──
    n_show = min(365, len(df_hist))
    axes[0, 0].plot(df_hist["tgl_transaksi"].tail(n_show),
                    df_hist["total_penjualan"].tail(n_show) / 1e6,
                    color="#95a5a6", linewidth=0.8, label="Aktual", alpha=0.8)
    axes[0, 0].plot(df_hist["tgl_transaksi"].tail(n_show),
                    y_fitted[-n_show:] / 1e6,
                    color="#e74c3c", linewidth=1.2, label="Fitted", alpha=0.9)
    axes[0, 0].set_title("Historis vs Fitted (365 hari terakhir)")
    axes[0, 0].set_ylabel("Penjualan (juta Rp)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.2)

    # ── Plot 2: Forecast 90 hari ──
    axes[0, 1].fill_between(df_forecast["tgl_forecast"],
                             df_forecast["penjualan_pred"] / 1e6 * 0.85,
                             df_forecast["penjualan_pred"] / 1e6 * 1.15,
                             alpha=0.2, color="#3498db", label="Rentang ±15%")
    axes[0, 1].plot(df_forecast["tgl_forecast"],
                    df_forecast["penjualan_pred"] / 1e6,
                    color="#2980b9", linewidth=2, marker="o", markersize=3, label="Prediksi")
    axes[0, 1].set_title(f"Forecast {HORIZON} Hari ke Depan")
    axes[0, 1].set_ylabel("Penjualan Prediksi (juta Rp)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.2)
    axes[0, 1].tick_params(axis="x", rotation=30)

    # ── Plot 3: Cross-Val Scores ──
    folds = [f"Fold {i+1}" for i in range(len(mae_list))]
    x = np.arange(len(folds))
    axes[1, 0].bar(x - 0.2, [m / 1e6 for m in mae_list], 0.4, label="MAE (juta)", color="#27ae60")
    axes[1, 0].bar(x + 0.2, [r / 1e6 for r in rmse_list], 0.4, label="RMSE (juta)", color="#8e44ad")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(folds)
    axes[1, 0].set_title("Metrik Error per Fold (TimeSeriesSplit)")
    axes[1, 0].set_ylabel("Error (juta Rp)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.2, axis="y")

    # ── Plot 4: R² per fold ──
    axes[1, 1].plot(folds, r2_list, marker="o", color="#e67e22", linewidth=2, markersize=8)
    axes[1, 1].axhline(y=np.mean(r2_list), color="red", linestyle="--",
                       label=f"Rata-rata R²={np.mean(r2_list):.3f}")
    axes[1, 1].set_title("R² Score per Fold")
    axes[1, 1].set_ylabel("R² Score")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.2)

    plt.tight_layout()
    path_out = os.path.join(OUTPUT_DIR, "3_sales_forecasting.png")
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()
    return path_out


def jalankan(path_harian: str) -> dict:
    print("\n" + "="*60)
    print("  MODUL 3: FORECASTING PENJUALAN")
    print("="*60)

    print("  Memuat data penjualan harian...")
    df = pd.read_csv(path_harian, parse_dates=["tgl_transaksi"])

    print(f"  Data historis     : {len(df):,} hari ({df['tgl_transaksi'].min().date()} s/d {df['tgl_transaksi'].max().date()})")

    print("  Melatih model Polynomial Ridge Regression...")
    pipe, df_feat, y_fitted, mae_list, rmse_list, r2_list = latih_model(df)

    print(f"\n  Evaluasi TimeSeriesSplit (5 fold):")
    print(f"    MAE  rata-rata : Rp {np.mean(mae_list):>12,.0f}")
    print(f"    RMSE rata-rata : Rp {np.mean(rmse_list):>12,.0f}")
    print(f"    R²   rata-rata : {np.mean(r2_list):.4f}")

    print(f"\n  Membuat forecast {HORIZON} hari ke depan...")
    df_forecast = buat_forecast(pipe, df_feat)

    total_pred = df_forecast["penjualan_pred"].sum()
    monthly = total_pred / 3
    print(f"\n  Total prediksi 3 bulan : Rp {total_pred:>15,.0f}")
    print(f"  Rata-rata per bulan    : Rp {monthly:>15,.0f}")

    path_img = simpan_visualisasi(df_feat, y_fitted, df_forecast, mae_list, rmse_list, r2_list)
    print(f"\n  Visualisasi disimpan: {path_img}")

    hasil_path = os.path.join(OUTPUT_DIR, "hasil_forecast.csv")
    df_forecast.to_csv(hasil_path, index=False)
    print(f"  Hasil forecast tersimpan: {hasil_path}")

    print("\n  >> Tantangan utama forecasting deret waktu:")
    print("     Non-stasioneritas: pola tren/musiman bisa berubah mendadak")
    print("     (misal: pandemi, kebijakan baru, kompetitor baru)")

    print("\n  >> Rekomendasi Stok:")
    bulan_pred = df_forecast.groupby(df_forecast["tgl_forecast"].dt.month)["penjualan_pred"].sum()
    for bln, val in bulan_pred.items():
        print(f"     Bulan {bln:02d}: Rp {val:>12,.0f} → Siapkan stok proporsional")

    return {
        "model"          : pipe,
        "df_forecast"    : df_forecast,
        "mae_mean"       : np.mean(mae_list),
        "rmse_mean"      : np.mean(rmse_list),
        "r2_mean"        : np.mean(r2_list),
        "total_3bln"     : total_pred,
        "monthly_avg"    : monthly,
    }
