"""
Module 1: Churn Classification
==============================
Memprediksi apakah pelanggan akan churn dalam 30 hari ke depan.

Algoritma   : Random Forest Classifier
Imbalance   : SMOTE (Synthetic Minority Over-sampling Technique)
Alasan RF   : - Tahan overfitting, tidak perlu feature scaling
               - Mampu menangani fitur campuran (numerik + kategorik)
               - Memberikan feature importance yang interpretatif
               - Robust terhadap outlier
Alasan SMOTE: - Data churn biasanya <25%, SMOTE membuat sampel sintetis
               pada kelas minoritas sehingga model tidak bias ke mayoritas
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def muat_data(path_pelanggan: str) -> pd.DataFrame:
    df = pd.read_csv(path_pelanggan)
    return df


def rekayasa_fitur(df: pd.DataFrame) -> pd.DataFrame:
    """Encoding kategorik + fitur tambahan RFM."""
    df = df.copy()

    le_gender = LabelEncoder()
    le_kota   = LabelEncoder()
    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    df["kota_enc"]   = le_kota.fit_transform(df["kota"])

    # RFM score sederhana (1–5)
    df["R_score"] = pd.qcut(df["hari_sejak_beli"].clip(upper=365), 5,
                             labels=[5, 4, 3, 2, 1], duplicates="drop").astype(float)
    df["F_score"] = pd.qcut(df["frekuensi_beli"].clip(lower=1), 5,
                             labels=[1, 2, 3, 4, 5], duplicates="drop").astype(float)
    df["M_score"] = pd.qcut(df["total_belanja"].clip(lower=1), 5,
                             labels=[1, 2, 3, 4, 5], duplicates="drop").astype(float)
    df["R_score"] = df["R_score"].fillna(1)
    df["F_score"] = df["F_score"].fillna(1)
    df["M_score"] = df["M_score"].fillna(1)
    df["RFM_total"] = df["R_score"] + df["F_score"] + df["M_score"]

    return df


FITUR = [
    "usia", "gender_enc", "kota_enc",
    "frekuensi_beli", "total_belanja", "hari_sejak_beli",
    "R_score", "F_score", "M_score", "RFM_total"
]
TARGET = "churn"


def latih_model(df: pd.DataFrame):
    df = rekayasa_fitur(df)
    X = df[FITUR].fillna(0).values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # ── Tangani imbalance dengan SMOTE ──
    if SMOTE_AVAILABLE:
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        smote_note = "SMOTE diterapkan"
    else:
        X_train_res, y_train_res = X_train, y_train
        smote_note = "SMOTE tidak tersedia (imbalanced-learn tidak terinstall)"

    # ── Model ──
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",   # cadangan jika SMOTE tidak ada
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_res, y_train_res)

    # ── Evaluasi ──
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    return clf, X_test, y_test, y_pred, y_proba, auc, cv_scores, smote_note, df


def simpan_visualisasi(clf, X_test, y_test, y_pred, y_proba, auc, cv_scores):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Modul 1: Prediksi Churn Pelanggan (Random Forest + SMOTE)",
                 fontsize=14, fontweight="bold")

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Tidak Churn", "Churn"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    # 3. Feature Importance
    importances = pd.Series(clf.feature_importances_, index=FITUR).sort_values()
    colors = ["#e74c3c" if i >= len(importances) - 3 else "#3498db"
              for i in range(len(importances))]
    axes[2].barh(importances.index, importances.values, color=colors)
    axes[2].set_title("Feature Importance")
    axes[2].set_xlabel("Importance Score")
    top_patch  = mpatches.Patch(color="#e74c3c", label="Top 3 Fitur")
    rest_patch = mpatches.Patch(color="#3498db", label="Fitur Lainnya")
    axes[2].legend(handles=[top_patch, rest_patch], fontsize=8)

    plt.tight_layout()
    path_out = os.path.join(OUTPUT_DIR, "1_churn_classification.png")
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()
    return path_out


def jalankan(path_pelanggan: str) -> dict:
    print("\n" + "="*60)
    print("  MODUL 1: KLASIFIKASI CHURN PELANGGAN")
    print("="*60)

    print("  Memuat & memproses data...")
    df = muat_data(path_pelanggan)

    distrib = df[TARGET].value_counts()
    print(f"  Distribusi label  : Tidak Churn={distrib.get(0,0):,}  |  Churn={distrib.get(1,0):,}")
    print(f"  Rasio churn       : {distrib.get(1,0)/len(df)*100:.1f}%")

    print("  Melatih model Random Forest...")
    clf, X_test, y_test, y_pred, y_proba, auc, cv_scores, smote_note, df_feat = \
        latih_model(df)

    print(f"\n  {smote_note}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Tidak Churn", "Churn"]))
    print(f"  ROC-AUC Score     : {auc:.4f}")
    print(f"  Cross-Val AUC (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    path_img = simpan_visualisasi(clf, X_test, y_test, y_pred, y_proba, auc, cv_scores)
    print(f"\n  Visualisasi disimpan: {path_img}")

    # Prediksi seluruh pelanggan
    X_all = df_feat[FITUR].fillna(0).values
    df_feat["prob_churn"] = clf.predict_proba(X_all)[:, 1]
    df_feat["prediksi_churn"] = (df_feat["prob_churn"] >= 0.5).astype(int)

    # Simpan hasil prediksi
    hasil_path = os.path.join(OUTPUT_DIR, "hasil_prediksi_churn.csv")
    df_feat[["id_pelanggan", "prob_churn", "prediksi_churn", "RFM_total"]].to_csv(
        hasil_path, index=False
    )
    print(f"  Hasil prediksi tersimpan: {hasil_path}")

    # Ringkasan bisnis
    n_at_risk = (df_feat["prob_churn"] >= 0.7).sum()
    print(f"\n  >> Pelanggan berisiko tinggi churn (prob ≥ 0.7): {n_at_risk:,} orang")
    print("  >> Rekomendasi: Kirim voucher / program loyalitas ke segmen ini.")

    return {
        "model"       : clf,
        "auc"         : auc,
        "cv_auc_mean" : cv_scores.mean(),
        "df_hasil"    : df_feat,
        "n_at_risk"   : int(n_at_risk),
    }
