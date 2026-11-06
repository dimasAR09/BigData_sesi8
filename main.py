"""
main.py — E-Commerce Analytics Pipeline
=========================================
Menjalankan seluruh pipeline analitik secara berurutan:
  1. Generate dataset sintetis
  2. Klasifikasi Churn (Random Forest + SMOTE)
  3. Segmentasi Pelanggan (DBSCAN)
  4. Forecasting Penjualan (Polynomial Ridge Regression)
  5. Dashboard Integrasi Keputusan Bisnis

Cara menjalankan:
  python main.py
"""

import os
import sys
import time

# ── Path setup ──
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
MODULES_DIR = os.path.join(BASE_DIR, "modules")

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, MODULES_DIR)

# Path file CSV
PATH_PELANGGAN = os.path.join(DATA_DIR, "data_pelanggan.csv")
PATH_TRANSAKSI = os.path.join(DATA_DIR, "data_transaksi.csv")
PATH_ULASAN    = os.path.join(DATA_DIR, "data_ulasan.csv")
PATH_HARIAN    = os.path.join(DATA_DIR, "data_penjualan_harian.csv")


def banner():
    print("\n" + "█"*62)
    print("█" + " "*60 + "█")
    print("█   E-COMMERCE ANALYTICS: RETENSI PELANGGAN & STOK OPTIM  █")
    print("█" + " "*60 + "█")
    print("█   Modul:                                                 █")
    print("█   1. Klasifikasi Churn    (Random Forest + SMOTE)        █")
    print("█   2. Segmentasi Pelanggan (DBSCAN)                       █")
    print("█   3. Forecasting Penjualan(Polynomial Ridge Regression)  █")
    print("█   4. Dashboard Integrasi  (Keputusan Bisnis Terpadu)     █")
    print("█" + " "*60 + "█")
    print("█"*62 + "\n")


def cek_dependencies():
    """Cek library yang dibutuhkan."""
    required = {
        "numpy"       : "numpy",
        "pandas"      : "pandas",
        "matplotlib"  : "matplotlib",
        "sklearn"     : "scikit-learn",
    }
    optional = {
        "imblearn"    : "imbalanced-learn (opsional, untuk SMOTE)",
    }
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("  [ERROR] Library berikut belum terinstall:")
        for m in missing:
            print(f"          pip install {m}")
        sys.exit(1)

    for mod, pkg in optional.items():
        try:
            __import__(mod)
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [WARN] {pkg} tidak ditemukan → SMOTE dinonaktifkan")


def main():
    banner()
    t_total = time.time()

    # ── Cek dependencies ──
    print("  Memeriksa dependencies...")
    cek_dependencies()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ══════════════════════════════════════════
    # STEP 1: Generate Dataset
    # ══════════════════════════════════════════
    t0 = time.time()
    print("\n" + "─"*62)
    print("  STEP 1/5 — Membuat Dataset Sintetis")
    print("─"*62)

    from data.generate_data import generate_all
    generate_all()
    print(f"  Selesai dalam {time.time() - t0:.1f} detik")

    # ══════════════════════════════════════════
    # STEP 2: Churn Classification
    # ══════════════════════════════════════════
    t0 = time.time()
    print("\n" + "─"*62)
    print("  STEP 2/5 — Klasifikasi Churn")
    print("─"*62)

    from modules.churn_classification import jalankan as jalankan_churn
    hasil_churn = jalankan_churn(PATH_PELANGGAN)
    print(f"  Selesai dalam {time.time() - t0:.1f} detik")

    # ══════════════════════════════════════════
    # STEP 3: Customer Clustering
    # ══════════════════════════════════════════
    t0 = time.time()
    print("\n" + "─"*62)
    print("  STEP 3/5 — Segmentasi Pelanggan")
    print("─"*62)

    from modules.customer_clustering import jalankan as jalankan_cluster
    hasil_cluster = jalankan_cluster(PATH_PELANGGAN, PATH_TRANSAKSI, PATH_ULASAN)
    print(f"  Selesai dalam {time.time() - t0:.1f} detik")

    # ══════════════════════════════════════════
    # STEP 4: Sales Forecasting
    # ══════════════════════════════════════════
    t0 = time.time()
    print("\n" + "─"*62)
    print("  STEP 4/5 — Forecasting Penjualan")
    print("─"*62)

    from modules.sales_forecasting import jalankan as jalankan_forecast
    hasil_forecast = jalankan_forecast(PATH_HARIAN)
    print(f"  Selesai dalam {time.time() - t0:.1f} detik")

    # ══════════════════════════════════════════
    # STEP 5: Integration Dashboard
    # ══════════════════════════════════════════
    t0 = time.time()
    print("\n" + "─"*62)
    print("  STEP 5/5 — Dashboard Integrasi")
    print("─"*62)

    from modules.integration_dashboard import jalankan as jalankan_integrasi
    hasil_integrasi = jalankan_integrasi(
        hasil_churn, hasil_cluster, hasil_forecast, PATH_TRANSAKSI
    )
    print(f"  Selesai dalam {time.time() - t0:.1f} detik")

    # ══════════════════════════════════════════
    # RINGKASAN AKHIR
    # ══════════════════════════════════════════
    print("\n" + "█"*62)
    print("  PIPELINE SELESAI")
    print(f"  Total waktu eksekusi : {time.time() - t_total:.1f} detik")
    print("─"*62)
    print("  Output yang dihasilkan di folder  output/ :")
    output_files = [
        ("1_churn_classification.png",  "Visualisasi model churn (CM, ROC, Feature Importance)"),
        ("2_customer_clustering.png",   "Visualisasi segmentasi pelanggan (DBSCAN)"),
        ("3_sales_forecasting.png",     "Visualisasi forecast penjualan 3 bulan"),
        ("4_integration_dashboard.png", "Dashboard integrasi keputusan bisnis"),
        ("hasil_prediksi_churn.csv",    "Probabilitas churn per pelanggan"),
        ("hasil_clustering.csv",        "Segmen per pelanggan"),
        ("hasil_forecast.csv",          "Prediksi penjualan 90 hari"),
        ("master_analitik.csv",         "Tabel master gabungan semua analitik"),
    ]
    for fname, desc in output_files:
        status = "✓" if os.path.exists(os.path.join(OUTPUT_DIR, fname)) else "✗"
        print(f"  [{status}] {fname:<40} {desc}")
    print("█"*62 + "\n")


if __name__ == "__main__":
    main()
