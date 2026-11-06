"""
generate_data.py
================
Membuat dataset sintetis untuk simulasi e-commerce:
  - data_pelanggan.csv  : demografi + fitur perilaku pelanggan
  - data_transaksi.csv  : riwayat transaksi per pelanggan
  - data_ulasan.csv     : ulasan produk per transaksi
  - data_penjualan_harian.csv : penjualan agregat harian (untuk forecasting)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N_PELANGGAN = 2000
N_TRANSAKSI = 12000
N_HARI      = 730          # ~2 tahun data harian
TANGGAL_MULAI = datetime(2024, 11, 1)

KATEGORI_PRODUK = [
    "Elektronik", "Fashion", "Rumah Tangga",
    "Makanan & Minuman", "Kesehatan", "Olahraga"
]

# ──────────────────────────────────────────────
# 1. DATA PELANGGAN
# ──────────────────────────────────────────────
def buat_data_pelanggan():
    ids = [f"C{str(i).zfill(5)}" for i in range(1, N_PELANGGAN + 1)]

    usia = np.random.randint(18, 65, N_PELANGGAN)
    gender = np.random.choice(["Pria", "Wanita"], N_PELANGGAN)
    kota = np.random.choice(
        ["Jakarta", "Surabaya", "Bandung", "Medan", "Makassar"],
        N_PELANGGAN, p=[0.35, 0.25, 0.20, 0.12, 0.08]
    )
    tgl_daftar = [
        (TANGGAL_MULAI - timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in np.random.randint(30, 730, N_PELANGGAN)
    ]

    # Fitur perilaku (akan di-update setelah transaksi dibuat)
    frekuensi_beli   = np.zeros(N_PELANGGAN, dtype=int)
    total_belanja    = np.zeros(N_PELANGGAN)
    hari_terakhir    = np.full(N_PELANGGAN, 999, dtype=int)

    # Label churn: ~20% pelanggan churn (imbalanced)
    prob_churn = np.where(usia < 30, 0.25, np.where(usia > 55, 0.28, 0.18))
    churn = np.array([
        1 if random.random() < p else 0
        for p in prob_churn
    ])

    df = pd.DataFrame({
        "id_pelanggan"   : ids,
        "usia"           : usia,
        "gender"         : gender,
        "kota"           : kota,
        "tgl_daftar"     : tgl_daftar,
        "frekuensi_beli" : frekuensi_beli,
        "total_belanja"  : total_belanja,
        "hari_sejak_beli": hari_terakhir,
        "churn"          : churn,
    })
    return df


# ──────────────────────────────────────────────
# 2. DATA TRANSAKSI
# ──────────────────────────────────────────────
def buat_data_transaksi(df_pelanggan):
    ids_pelanggan = df_pelanggan["id_pelanggan"].tolist()
    churn_map     = dict(zip(df_pelanggan["id_pelanggan"], df_pelanggan["churn"]))

    transaksi = []
    prod_id = 1

    for _ in range(N_TRANSAKSI):
        cid = random.choice(ids_pelanggan)
        is_churn = churn_map[cid]

        # Pelanggan churn cenderung transaksi lebih lama yang lalu
        if is_churn:
            hari_lalu = np.random.randint(60, 700)
        else:
            hari_lalu = np.random.randint(1, 300)

        tgl = TANGGAL_MULAI + timedelta(days=N_HARI) - timedelta(days=hari_lalu)

        kategori = np.random.choice(KATEGORI_PRODUK)

        # Nilai transaksi berdasar kategori
        harga_base = {
            "Elektronik": (500_000, 5_000_000),
            "Fashion": (100_000, 800_000),
            "Rumah Tangga": (150_000, 2_000_000),
            "Makanan & Minuman": (30_000, 300_000),
            "Kesehatan": (50_000, 600_000),
            "Olahraga": (100_000, 1_500_000),
        }
        lo, hi = harga_base[kategori]
        nilai = round(np.random.uniform(lo, hi), -3)
        qty   = np.random.randint(1, 6)

        transaksi.append({
            "id_transaksi"  : f"T{str(prod_id).zfill(7)}",
            "id_pelanggan"  : cid,
            "tgl_transaksi" : tgl.strftime("%Y-%m-%d"),
            "kategori"      : kategori,
            "id_produk"     : f"P{np.random.randint(1, 500):04d}",
            "qty"           : qty,
            "nilai_transaksi": nilai * qty,
            "metode_bayar"  : np.random.choice(
                ["Transfer Bank", "Kartu Kredit", "E-Wallet", "COD"],
                p=[0.30, 0.20, 0.35, 0.15]
            ),
            "diskon_pct"    : np.random.choice([0, 5, 10, 15, 20, 30], p=[0.40, 0.20, 0.18, 0.10, 0.08, 0.04]),
        })
        prod_id += 1

    df_trx = pd.DataFrame(transaksi)
    df_trx["tgl_transaksi"] = pd.to_datetime(df_trx["tgl_transaksi"])

    # Update fitur perilaku di df_pelanggan
    tgl_akhir = TANGGAL_MULAI + timedelta(days=N_HARI)
    grp = df_trx.groupby("id_pelanggan").agg(
        frekuensi_beli   = ("id_transaksi", "count"),
        total_belanja    = ("nilai_transaksi", "sum"),
        tgl_terakhir     = ("tgl_transaksi", "max"),
    ).reset_index()
    grp["hari_sejak_beli"] = (tgl_akhir - grp["tgl_terakhir"]).dt.days

    df_pelanggan = df_pelanggan.merge(
        grp[["id_pelanggan", "frekuensi_beli", "total_belanja", "hari_sejak_beli"]],
        on="id_pelanggan", how="left", suffixes=("_old", "")
    )
    df_pelanggan = df_pelanggan.drop(columns=[c for c in df_pelanggan.columns if c.endswith("_old")])
    df_pelanggan["frekuensi_beli"] = df_pelanggan["frekuensi_beli"].fillna(0)
    df_pelanggan["total_belanja"]  = df_pelanggan["total_belanja"].fillna(0)
    df_pelanggan["hari_sejak_beli"]= df_pelanggan["hari_sejak_beli"].fillna(999)

    return df_pelanggan, df_trx


# ──────────────────────────────────────────────
# 3. DATA ULASAN
# ──────────────────────────────────────────────
def buat_data_ulasan(df_trx):
    # Hanya ~60% transaksi punya ulasan
    df_sample = df_trx.sample(frac=0.60, random_state=SEED).copy()

    sentimen_pool = {
        5: "Produk sangat bagus, pengiriman cepat!",
        4: "Kualitas sesuai harga, cukup puas.",
        3: "Biasa saja, tidak ada yang istimewa.",
        2: "Kecewa dengan kualitas produk.",
        1: "Sangat buruk, tidak sesuai deskripsi.",
    }
    rating = np.random.choice([1, 2, 3, 4, 5], len(df_sample), p=[0.05, 0.08, 0.17, 0.35, 0.35])

    df_ulasan = pd.DataFrame({
        "id_ulasan"     : [f"R{str(i).zfill(7)}" for i in range(1, len(df_sample) + 1)],
        "id_transaksi"  : df_sample["id_transaksi"].values,
        "id_pelanggan"  : df_sample["id_pelanggan"].values,
        "rating"        : rating,
        "komentar"      : [sentimen_pool[r] for r in rating],
        "tgl_ulasan"    : df_sample["tgl_transaksi"].values,
    })
    return df_ulasan


# ──────────────────────────────────────────────
# 4. DATA PENJUALAN HARIAN
# ──────────────────────────────────────────────
def buat_penjualan_harian(df_trx):
    grp = df_trx.groupby("tgl_transaksi").agg(
        total_penjualan = ("nilai_transaksi", "sum"),
        jumlah_order    = ("id_transaksi", "count"),
        qty_terjual     = ("qty", "sum"),
    ).reset_index()

    # Pastikan semua hari ada (isi 0 untuk hari tanpa transaksi)
    semua_hari = pd.date_range(
        start=TANGGAL_MULAI,
        end=TANGGAL_MULAI + timedelta(days=N_HARI - 1)
    )
    df_all = pd.DataFrame({"tgl_transaksi": semua_hari})
    df_harian = df_all.merge(grp, on="tgl_transaksi", how="left").fillna(0)

    # Tambah fitur waktu
    df_harian["hari_ke"]   = range(len(df_harian))
    df_harian["bulan"]     = df_harian["tgl_transaksi"].dt.month
    df_harian["hari_minggu"] = df_harian["tgl_transaksi"].dt.dayofweek
    df_harian["is_weekend"] = (df_harian["hari_minggu"] >= 5).astype(int)

    # Tambah musiman buatan (harbolnas, lebaran, dll.)
    df_harian["is_harbolnas"] = df_harian["tgl_transaksi"].apply(
        lambda x: 1 if x.month == 12 and x.day == 12 else 0
    )

    return df_harian


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def generate_all():
    out_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)

    print("  [1/4] Membuat data pelanggan...")
    df_pel = buat_data_pelanggan()

    print("  [2/4] Membuat data transaksi...")
    df_pel, df_trx = buat_data_transaksi(df_pel)

    print("  [3/4] Membuat data ulasan...")
    df_ulasan = buat_data_ulasan(df_trx)

    print("  [4/4] Membuat data penjualan harian...")
    df_harian = buat_penjualan_harian(df_trx)

    # Simpan
    df_pel.to_csv(os.path.join(out_dir, "data_pelanggan.csv"), index=False)
    df_trx.to_csv(os.path.join(out_dir, "data_transaksi.csv"), index=False)
    df_ulasan.to_csv(os.path.join(out_dir, "data_ulasan.csv"), index=False)
    df_harian.to_csv(os.path.join(out_dir, "data_penjualan_harian.csv"), index=False)

    print(f"\n  Dataset tersimpan di: {out_dir}")
    print(f"    - data_pelanggan.csv     : {len(df_pel):,} baris")
    print(f"    - data_transaksi.csv     : {len(df_trx):,} baris")
    print(f"    - data_ulasan.csv        : {len(df_ulasan):,} baris")
    print(f"    - data_penjualan_harian.csv: {len(df_harian):,} baris")

    return df_pel, df_trx, df_ulasan, df_harian


if __name__ == "__main__":
    generate_all()
