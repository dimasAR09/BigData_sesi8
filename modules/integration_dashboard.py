"""
Module 4: Integration Dashboard
=================================
Menggabungkan hasil dari 3 modul (Churn, Clustering, Forecasting)
menjadi dashboard keputusan bisnis terpadu.

Contoh keputusan bisnis yang didukung integrasi:
  1. Kirim promo tepat sasaran: Cluster VIP + Prob Churn Tinggi → retensi premium
  2. Kelola stok per segmen  : Forecast penjualan × proporsi kategori per cluster
  3. Prioritas re-engagement : Cluster Casual + churn tinggi + rating rendah
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def gabung_hasil(df_churn: pd.DataFrame,
                 df_cluster: pd.DataFrame,
                 df_forecast: pd.DataFrame,
                 df_trx: pd.DataFrame) -> pd.DataFrame:
    """Merge semua hasil analitik ke dalam satu tabel master."""

    # Merge churn + cluster
    df_master = df_churn[["id_pelanggan", "prob_churn", "prediksi_churn",
                           "RFM_total", "usia", "total_belanja", "frekuensi_beli",
                           "hari_sejak_beli"]].merge(
        df_cluster[["id_pelanggan", "cluster", "segmen", "rating_rata"]],
        on="id_pelanggan", how="left"
    )

    # Tambah kategori favorit per pelanggan
    fav_kat = df_trx.groupby("id_pelanggan")["kategori"].agg(
        lambda x: x.value_counts().idxmax()
    ).rename("kategori_favorit").reset_index()
    df_master = df_master.merge(fav_kat, on="id_pelanggan", how="left")

    # Klasifikasi prioritas keputusan
    def prioritas(row):
        if row["prob_churn"] >= 0.7 and row["segmen"] == "VIP / High-Value":
            return "KRITIS - Retensi VIP"
        elif row["prob_churn"] >= 0.7:
            return "TINGGI - Intervensi Segera"
        elif row["prob_churn"] >= 0.5:
            return "SEDANG - Pantau & Promosi"
        else:
            return "RENDAH - Pertahankan"

    df_master["prioritas_aksi"] = df_master.apply(prioritas, axis=1)
    return df_master


def hitung_kebutuhan_stok(df_forecast: pd.DataFrame,
                           df_trx: pd.DataFrame,
                           df_master: pd.DataFrame) -> pd.DataFrame:
    """Hitung distribusi kebutuhan stok per kategori berdasarkan forecast."""
    total_forecast = df_forecast["penjualan_pred"].sum()

    # Proporsi historis per kategori
    proporsi_kat = df_trx.groupby("kategori")["nilai_transaksi"].sum()
    proporsi_kat = proporsi_kat / proporsi_kat.sum()

    # Alokasi forecast ke kategori
    df_stok = pd.DataFrame({
        "kategori"        : proporsi_kat.index,
        "proporsi_hist"   : proporsi_kat.values,
        "alokasi_forecast": proporsi_kat.values * total_forecast,
    })

    # Bonus: segmen VIP cenderung membeli Elektronik lebih banyak
    vip_kat = df_trx[df_trx["id_pelanggan"].isin(
        df_master[df_master["segmen"] == "VIP / High-Value"]["id_pelanggan"]
    )].groupby("kategori")["nilai_transaksi"].sum()
    if len(vip_kat) > 0:
        vip_kat = vip_kat / vip_kat.sum()
        df_stok = df_stok.merge(
            vip_kat.rename("proporsi_vip").reset_index(),
            on="kategori", how="left"
        )
        df_stok["proporsi_vip"] = df_stok["proporsi_vip"].fillna(0)
        df_stok["alokasi_vip"] = df_stok["proporsi_vip"] * total_forecast * 0.3
    else:
        df_stok["alokasi_vip"] = 0

    df_stok["total_alokasi"] = df_stok["alokasi_forecast"] + df_stok["alokasi_vip"]
    return df_stok.sort_values("total_alokasi", ascending=False)


def simpan_dashboard(df_master: pd.DataFrame,
                     df_stok: pd.DataFrame,
                     df_forecast: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(22, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(
        "DASHBOARD INTEGRASI: Retensi Pelanggan & Optimisasi Stok\n"
        "[ Churn Prediction + Customer Segmentation + Sales Forecasting ]",
        fontsize=13, fontweight="bold", y=0.98
    )

    # ── Panel 1: Distribusi Prioritas Aksi ──
    ax1 = fig.add_subplot(gs[0, 0])
    urutan  = ["KRITIS - Retensi VIP", "TINGGI - Intervensi Segera",
               "SEDANG - Pantau & Promosi", "RENDAH - Pertahankan"]
    warna   = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60"]
    jumlah  = [df_master["prioritas_aksi"].value_counts().get(u, 0) for u in urutan]
    bars = ax1.barh(urutan, jumlah, color=warna)
    ax1.bar_label(bars, padding=3, fontsize=8)
    ax1.set_title("Prioritas Aksi Pelanggan", fontweight="bold")
    ax1.set_xlabel("Jumlah Pelanggan")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.2, axis="x")

    # ── Panel 2: Churn Risk × Segmen (Heatmap) ──
    ax2 = fig.add_subplot(gs[0, 1])
    pivot_heat = df_master.pivot_table(
        index="segmen", columns="prioritas_aksi",
        values="id_pelanggan", aggfunc="count", fill_value=0
    )
    valid_cols = [c for c in urutan if c in pivot_heat.columns]
    pivot_heat = pivot_heat[valid_cols]
    im = ax2.imshow(pivot_heat.values, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(len(valid_cols)))
    ax2.set_xticklabels([c.split(" - ")[0] for c in valid_cols], rotation=30, ha="right", fontsize=7)
    ax2.set_yticks(range(len(pivot_heat.index)))
    ax2.set_yticklabels(pivot_heat.index, fontsize=8)
    for i in range(len(pivot_heat.index)):
        for j in range(len(valid_cols)):
            ax2.text(j, i, pivot_heat.values[i, j], ha="center", va="center",
                     fontsize=8, color="black")
    ax2.set_title("Matriks Churn Risk × Segmen", fontweight="bold")
    plt.colorbar(im, ax=ax2, shrink=0.7)

    # ── Panel 3: Forecast 90 Hari ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(df_forecast["tgl_forecast"],
             df_forecast["penjualan_pred"] / 1e6,
             color="#2980b9", linewidth=2)
    ax3.fill_between(df_forecast["tgl_forecast"],
                     df_forecast["penjualan_pred"] / 1e6 * 0.85,
                     df_forecast["penjualan_pred"] / 1e6 * 1.15,
                     alpha=0.15, color="#2980b9")
    ax3.set_title("Forecast Penjualan 90 Hari", fontweight="bold")
    ax3.set_ylabel("Juta Rp")
    ax3.tick_params(axis="x", rotation=30, labelsize=7)
    ax3.grid(True, alpha=0.2)

    # ── Panel 4: Alokasi Stok per Kategori ──
    ax4 = fig.add_subplot(gs[1, 0:2])
    colors_stok = plt.cm.Paired(np.linspace(0, 1, len(df_stok)))
    bars2 = ax4.bar(df_stok["kategori"], df_stok["total_alokasi"] / 1e9,
                    color=colors_stok)
    ax4.bar_label(bars2, labels=[f"Rp {v:.1f}M" for v in df_stok["total_alokasi"] / 1e6],
                  padding=3, fontsize=8)
    ax4.set_title("Rekomendasi Alokasi Stok per Kategori (3 Bulan)", fontweight="bold")
    ax4.set_ylabel("Nilai (miliar Rp)")
    ax4.tick_params(axis="x", rotation=20, labelsize=8)
    ax4.grid(True, alpha=0.2, axis="y")

    # ── Panel 5: Distribusi Prob Churn per Segmen ──
    ax5 = fig.add_subplot(gs[1, 2])
    segmen_list = [s for s in df_master["segmen"].unique() if s != "Noise / Outlier"]
    bp_data = [df_master[df_master["segmen"] == s]["prob_churn"].values
               for s in segmen_list]
    bp = ax5.boxplot(bp_data, tick_labels=[s.split("/")[0].strip() for s in segmen_list],
                     patch_artist=True, notch=False)
    colors_box = ["#e74c3c", "#3498db", "#2ecc71"]
    for patch, color in zip(bp["boxes"], colors_box[:len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax5.axhline(y=0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax5.set_title("Distribusi Prob Churn per Segmen", fontweight="bold")
    ax5.set_ylabel("Probabilitas Churn")
    ax5.tick_params(axis="x", rotation=20, labelsize=7)
    ax5.grid(True, alpha=0.2, axis="y")

    # ── Panel 6: Top 10 Pelanggan KRITIS ──
    ax6 = fig.add_subplot(gs[2, :])
    df_kritis = df_master[df_master["prioritas_aksi"] == "KRITIS - Retensi VIP"].sort_values(
        "prob_churn", ascending=False
    ).head(10)
    if len(df_kritis) == 0:
        df_kritis = df_master[df_master["prioritas_aksi"] == "TINGGI - Intervensi Segera"].sort_values(
            "prob_churn", ascending=False
        ).head(10)

    cols_show = ["id_pelanggan", "segmen", "prob_churn", "RFM_total",
                 "total_belanja", "frekuensi_beli", "kategori_favorit", "prioritas_aksi"]
    df_show = df_kritis[cols_show].copy()
    df_show["prob_churn"]    = df_show["prob_churn"].map("{:.2%}".format)
    df_show["total_belanja"] = df_show["total_belanja"].map("Rp {:,.0f}".format)

    ax6.axis("off")
    tbl = ax6.table(
        cellText=df_show.values,
        colLabels=[c.replace("_", " ").title() for c in cols_show],
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#ecf0f1")
    ax6.set_title("Top Pelanggan Prioritas Intervensi", fontweight="bold", pad=12)

    path_out = os.path.join(OUTPUT_DIR, "4_integration_dashboard.png")
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()
    return path_out


def cetak_ringkasan_bisnis(df_master: pd.DataFrame,
                            df_stok: pd.DataFrame,
                            df_forecast: pd.DataFrame,
                            hasil_churn: dict,
                            hasil_cluster: dict,
                            hasil_forecast: dict):
    total = len(df_master)
    print("\n" + "="*60)
    print("  INTEGRASI: RINGKASAN KEPUTUSAN BISNIS")
    print("="*60)

    print(f"\n  Total pelanggan dianalisis : {total:,}")
    print(f"  Akurasi model churn (AUC)  : {hasil_churn['auc']:.4f}")
    print(f"  Jumlah segmen pelanggan    : {hasil_cluster['n_cluster']}")
    print(f"  Forecast total 3 bulan     : Rp {hasil_forecast['total_3bln']:,.0f}")

    print("\n  KEPUTUSAN BISNIS TERINTEGRASI:")
    print("  " + "-"*56)

    # Keputusan 1
    n_kritis = df_master["prioritas_aksi"].value_counts().get("KRITIS - Retensi VIP", 0)
    n_tinggi = df_master["prioritas_aksi"].value_counts().get("TINGGI - Intervensi Segera", 0)
    print(f"\n  [1] RETENSI PELANGGAN VIP")
    print(f"      {n_kritis:,} pelanggan VIP berisiko churn tinggi.")
    print(f"      Aksi: Program loyalitas eksklusif, diskon 20-30%, personal outreach.")
    print(f"      Estimasi potensi pendapatan yang diselamatkan:")
    avg_spend_vip = df_master[df_master["segmen"] == "VIP / High-Value"]["total_belanja"].mean()
    print(f"      Rp {avg_spend_vip * n_kritis:,.0f} (asumsi retensi 100%)")

    # Keputusan 2
    print(f"\n  [2] INTERVENSI PELANGGAN AT-RISK")
    print(f"      {n_tinggi:,} pelanggan non-VIP dengan prob churn ≥ 0.70.")
    print(f"      Aksi: Flash sale personal, push notifikasi, email re-engagement.")

    # Keputusan 3
    print(f"\n  [3] OPTIMISASI STOK 3 BULAN KE DEPAN")
    print(f"      Total alokasi stok yang direkomendasikan:")
    for _, row in df_stok.iterrows():
        print(f"        {row['kategori']:<25}: Rp {row['total_alokasi']:>14,.0f}")

    # Keputusan 4
    bulan_tertinggi = df_forecast.groupby(df_forecast["tgl_forecast"].dt.month)["penjualan_pred"].sum().idxmax()
    print(f"\n  [4] PERENCANAAN KAPASITAS")
    print(f"      Bulan dengan prediksi penjualan tertinggi: Bulan {bulan_tertinggi}")
    print(f"      Tingkatkan stok & SDM 20% sebelum bulan tersebut.")

    print("\n" + "="*60)


def jalankan(hasil_churn: dict, hasil_cluster: dict,
             hasil_forecast: dict, path_transaksi: str) -> dict:
    print("\n" + "="*60)
    print("  MODUL 4: INTEGRASI & DASHBOARD KEPUTUSAN BISNIS")
    print("="*60)

    df_trx = pd.read_csv(path_transaksi)

    print("  Menggabungkan hasil analitik...")
    df_master = gabung_hasil(
        hasil_churn["df_hasil"],
        hasil_cluster["df_hasil"],
        hasil_forecast["df_forecast"],
        df_trx
    )

    print("  Menghitung kebutuhan stok...")
    df_stok = hitung_kebutuhan_stok(
        hasil_forecast["df_forecast"], df_trx, df_master
    )

    path_img = simpan_dashboard(df_master, df_stok, hasil_forecast["df_forecast"])
    print(f"\n  Dashboard disimpan: {path_img}")

    cetak_ringkasan_bisnis(
        df_master, df_stok, hasil_forecast["df_forecast"],
        hasil_churn, hasil_cluster, hasil_forecast
    )

    master_path = os.path.join(OUTPUT_DIR, "master_analitik.csv")
    df_master.to_csv(master_path, index=False)
    print(f"\n  Tabel master tersimpan: {master_path}")

    return {
        "df_master": df_master,
        "df_stok"  : df_stok,
    }
