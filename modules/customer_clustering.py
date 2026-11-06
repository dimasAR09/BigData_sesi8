"""
Module 2: Customer Clustering
==============================
Mengelompokkan pelanggan berdasarkan pola belanja untuk strategi promosi.

Algoritma   : DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
Alasan DBSCAN:
  - Tidak perlu menentukan jumlah cluster di awal (vs K-Means)
  - Mampu mendeteksi cluster berbentuk sembarang (non-spherical)
  - Secara eksplisit menandai noise/outlier sebagai label -1
  - Robust terhadap data yang tidak seimbang antar cluster
  - Lebih realistis untuk data perilaku belanja yang bervariasi

Fitur clustering: Frekuensi, Nilai Total Belanja, Kategori Favorit, Rating Rata-rata
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def rekayasa_fitur_cluster(df_pel: pd.DataFrame,
                           df_trx: pd.DataFrame,
                           df_ulasan: pd.DataFrame) -> pd.DataFrame:
    """Bangun fitur RFM + kategori + rating per pelanggan."""

    # Frekuensi & total belanja per kategori
    pivot_kat = df_trx.pivot_table(
        index="id_pelanggan",
        columns="kategori",
        values="nilai_transaksi",
        aggfunc="sum",
        fill_value=0
    )
    pivot_kat.columns = [f"belanja_{c.replace(' & ', '_').replace(' ', '_').lower()}"
                         for c in pivot_kat.columns]

    # Rating rata-rata
    rating_avg = df_ulasan.groupby("id_pelanggan")["rating"].mean().rename("rating_rata")

    # Agregasi utama
    agg = df_trx.groupby("id_pelanggan").agg(
        frekuensi   = ("id_transaksi", "count"),
        total_belanja = ("nilai_transaksi", "sum"),
        avg_transaksi = ("nilai_transaksi", "mean"),
        variasi_kat   = ("kategori", "nunique"),
        avg_diskon    = ("diskon_pct", "mean"),
    )

    df_feat = df_pel[["id_pelanggan", "usia", "churn"]].merge(
        agg, on="id_pelanggan", how="left"
    ).merge(pivot_kat, on="id_pelanggan", how="left") \
     .merge(rating_avg, on="id_pelanggan", how="left")

    df_feat = df_feat.fillna(0).reset_index(drop=True)
    return df_feat


def cari_epsilon(X_scaled: np.ndarray, k: int = 5) -> float:
    """Heuristik k-distance untuk memilih epsilon DBSCAN.
    
    Untuk data berdimensi tinggi, gunakan persentil yang lebih tinggi
    agar tidak terlalu ketat (semua jadi noise).
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_dist = np.sort(distances[:, k - 1])

    # Gunakan "knee" berdasarkan persentil 90 (lebih longgar untuk high-dim)
    eps = float(np.percentile(k_dist, 90))
    return max(eps, 1.0)   # batas bawah aman


def jalankan_dbscan(X_scaled: np.ndarray, eps: float, min_samples: int):
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X_scaled)

    n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise   = (labels == -1).sum()

    # Jika DBSCAN gagal menemukan cluster bermakna (>80% noise),
    # fallback ke KMeans dengan 4 cluster
    if n_cluster < 2 or n_noise / len(labels) > 0.80:
        from sklearn.cluster import KMeans
        print("  [INFO] DBSCAN terlalu banyak noise → fallback ke KMeans (k=4)")
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

    return labels


def simpan_visualisasi(df_feat: pd.DataFrame, labels: np.ndarray,
                       X_scaled: np.ndarray):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Reduksi ke 2D untuk visualisasi
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise   = (labels == -1).sum()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Modul 2: Segmentasi Pelanggan (DBSCAN)", fontsize=14, fontweight="bold")

    # ── Plot 1: Scatter PCA ──
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_cluster, 1)))
    color_map = {}
    ci = 0
    for lbl in unique_labels:
        if lbl == -1:
            color_map[lbl] = (0.5, 0.5, 0.5, 0.4)   # noise = abu
        else:
            color_map[lbl] = colors[ci % len(colors)]
            ci += 1

    for lbl in unique_labels:
        mask = labels == lbl
        nama = "Noise" if lbl == -1 else f"Cluster {lbl}"
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=[color_map[lbl]], label=nama, s=15, alpha=0.7)
    axes[0].set_title(f"DBSCAN Clusters (PCA 2D)\n{n_cluster} cluster, {n_noise} noise")
    axes[0].legend(fontsize=7, markerscale=2)
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].grid(True, alpha=0.2)

    # ── Plot 2: Profil cluster ──
    df_feat["cluster"] = labels
    profil = df_feat[df_feat["cluster"] != -1].groupby("cluster").agg(
        n_pelanggan    = ("id_pelanggan", "count"),
        frekuensi_mean = ("frekuensi", "mean"),
        belanja_mean   = ("total_belanja", "mean"),
        rating_mean    = ("rating_rata", "mean"),
    )

    x = np.arange(len(profil))
    width = 0.25
    bars1 = axes[1].bar(x - width, profil["frekuensi_mean"], width, label="Frekuensi Rata-rata", color="#3498db")
    axes[1].set_ylabel("Frekuensi Rata-rata", color="#3498db")
    ax2 = axes[1].twinx()
    bars2 = ax2.bar(x, profil["belanja_mean"] / 1_000_000, width, label="Belanja (juta)", color="#e74c3c")
    ax2.set_ylabel("Total Belanja (juta Rp)", color="#e74c3c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"C{i}" for i in profil.index])
    axes[1].set_title("Profil Cluster: Frekuensi & Belanja")
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    axes[1].grid(True, alpha=0.2, axis="y")

    # ── Plot 3: Ukuran cluster ──
    ukuran = df_feat.groupby("cluster").size()
    ukuran_valid = ukuran[ukuran.index != -1]
    if len(ukuran_valid) > 0 and ukuran_valid.sum() > 0:
        colors_pie = plt.cm.Set2(np.linspace(0, 1, len(ukuran_valid)))
        axes[2].pie(
            ukuran_valid.values,
            labels=[f"Cluster {i}\n({v} pelanggan)" for i, v in ukuran_valid.items()],
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
            textprops={"fontsize": 8}
        )
        axes[2].set_title(f"Distribusi Pelanggan per Cluster\n(Noise: {n_noise} pelanggan)")
    else:
        axes[2].text(0.5, 0.5, "Tidak ada cluster\nyang ditemukan",
                     ha="center", va="center", transform=axes[2].transAxes, fontsize=10)
        axes[2].set_title("Distribusi Pelanggan per Cluster")

    plt.tight_layout()
    path_out = os.path.join(OUTPUT_DIR, "2_customer_clustering.png")
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()
    return path_out


def beri_label_segmen(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Beri nama bisnis ke tiap cluster berdasarkan profil."""
    profil = df_feat[df_feat["cluster"] != -1].groupby("cluster").agg(
        belanja_mean  = ("total_belanja", "mean"),
        freq_mean     = ("frekuensi", "mean"),
    )
    nama_segmen = {}
    for idx, row in profil.iterrows():
        if row["belanja_mean"] >= profil["belanja_mean"].quantile(0.66):
            nama_segmen[idx] = "VIP / High-Value"
        elif row["freq_mean"] >= profil["freq_mean"].median():
            nama_segmen[idx] = "Loyal / Regular"
        else:
            nama_segmen[idx] = "Casual / At-Risk"
    nama_segmen[-1] = "Noise / Outlier"

    df_feat["segmen"] = df_feat["cluster"].map(nama_segmen).fillna("Unknown")
    return df_feat


def jalankan(path_pelanggan: str, path_transaksi: str, path_ulasan: str) -> dict:
    print("\n" + "="*60)
    print("  MODUL 2: SEGMENTASI PELANGGAN (DBSCAN)")
    print("="*60)

    print("  Memuat data...")
    df_pel    = pd.read_csv(path_pelanggan)
    df_trx    = pd.read_csv(path_transaksi)
    df_ulasan = pd.read_csv(path_ulasan)

    print("  Merekayasa fitur clustering...")
    df_feat = rekayasa_fitur_cluster(df_pel, df_trx, df_ulasan)

    # Pilih kolom numerik saja
    fitur_num = [c for c in df_feat.columns
                 if c not in ("id_pelanggan", "churn", "cluster", "segmen")]
    X = df_feat[fitur_num].values

    # Normalisasi
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cari epsilon otomatis
    eps = cari_epsilon(X_scaled, k=5)
    min_samples = max(5, int(len(df_feat) * 0.01))   # min 1% data
    print(f"  Parameter DBSCAN  : eps={eps:.3f}, min_samples={min_samples}")

    labels = jalankan_dbscan(X_scaled, eps, min_samples)
    n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise   = (labels == -1).sum()

    df_feat["cluster"] = labels
    df_feat = beri_label_segmen(df_feat)

    print(f"\n  Jumlah cluster ditemukan : {n_cluster}")
    print(f"  Jumlah noise/outlier     : {n_noise:,} pelanggan ({n_noise/len(df_feat)*100:.1f}%)")

    print("\n  Profil per Cluster:")
    profil = df_feat[df_feat["cluster"] != -1].groupby(["cluster", "segmen"]).agg(
        n          = ("id_pelanggan", "count"),
        freq_mean  = ("frekuensi", "mean"),
        spend_mean = ("total_belanja", "mean"),
        rating     = ("rating_rata", "mean"),
    ).round(2)
    print(profil.to_string())

    path_img = simpan_visualisasi(df_feat, labels, X_scaled)
    print(f"\n  Visualisasi disimpan: {path_img}")

    hasil_path = os.path.join(OUTPUT_DIR, "hasil_clustering.csv")
    df_feat[["id_pelanggan", "cluster", "segmen", "frekuensi",
             "total_belanja", "rating_rata"]].to_csv(hasil_path, index=False)
    print(f"  Hasil clustering tersimpan: {hasil_path}")

    print("\n  >> Rekomendasi Strategi Promosi:")
    print("     VIP / High-Value  → Program eksklusif, early-access produk baru")
    print("     Loyal / Regular   → Program poin loyalitas, bundling diskon")
    print("     Casual / At-Risk  → Re-engagement email, flash sale personal")
    print("     Noise / Outlier   → Analisis manual, potensi fraud check")

    return {
        "df_hasil"  : df_feat,
        "n_cluster" : n_cluster,
        "n_noise"   : int(n_noise),
        "profil"    : profil,
    }
