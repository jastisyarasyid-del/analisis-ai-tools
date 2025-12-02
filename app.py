import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, shapiro
import matplotlib.pyplot as plt

# -----------------------------
# Konfigurasi halaman
# -----------------------------
st.set_page_config(page_title="Analisis Survey AI Tools", layout="wide")

# -----------------------------
# Bahasa / Language Strings
# -----------------------------
STRINGS = {
    "id": {
        "lang_label": "Bahasa",
        "lang_id": "Indonesia",
        "lang_en": "Inggris",
        "app_title": "📊 Analisis Pengaruh Penggunaan AI Tools terhadap Efektivitas Belajar Mahasiswa",
        "intro": """
Aplikasi ini akan:
- Membaca file CSV hasil Google Forms mengenai penggunaan AI tools
- Menghitung skor komposit:
  - **X_total** = skor penggunaan AI tools (10 item)
  - **Y_total** = skor efektivitas belajar (10 item)
- Menampilkan statistik deskriptif
- Menguji normalitas (Shapiro-Wilk)
- Menghitung korelasi (Pearson / Spearman) antara X_total dan Y_total
- Menampilkan visualisasi (histogram & scatter plot)
        """,
        "sidebar_title": "Navigasi Analisis",
        "page_overview": "Ringkasan & Data",
        "page_desc": "Statistik Deskriptif",
        "page_reliability": "Reliabilitas & Normalitas",
        "page_corr": "Korelasi & Visualisasi",
        "page_download": "Download Data",
        "upload_label": "📤 Upload file CSV (hasil download dari Google Forms)",
        "upload_info": "Silakan upload file CSV terlebih dahulu (misalnya: Survey_ai.csv).",
        "preview_title": "👀 Preview Data (5 baris pertama)",
        "missing_cols_error": "Masih ada kolom yang tidak ditemukan di CSV.",
        "missing_cols_hint": "Pastikan teks pertanyaan di Google Forms tidak diubah ketika export CSV.",
        "respondent_after_clean": "Jumlah responden setelah cleaning (min 18/20 item terisi):",
        "no_valid_resp": "Tidak ada responden yang memenuhi kriteria valid (>=18 item terisi).",
        "desc_item_title": "📌 Statistik Deskriptif — Item-level",
        "desc_comp_title": "📌 Statistik Deskriptif — Skor Komposit (X_total & Y_total)",
        "reliability_title": "🧪 Reliabilitas (Cronbach's Alpha)",
        "alpha_x_label": "α untuk Penggunaan AI Tools (X):",
        "alpha_y_label": "α untuk Efektivitas Belajar (Y):",
        "normality_title": "🧪 Uji Normalitas (Shapiro-Wilk) pada X_total dan Y_total",
        "normality_x_label": "X_total",
        "normality_y_label": "Y_total",
        "corr_title": "📈 Korelasi antara Penggunaan AI (X_total) dan Efektivitas Belajar (Y_total)",
        "corr_method_pearson": "Metode korelasi: **Pearson Correlation** (karena data normal).",
        "corr_method_spearman": "Metode korelasi: **Spearman Rank Correlation** (karena data tidak normal).",
        "corr_coef_label": "Koefisien korelasi (r):",
        "p_value_label": "p-value:",
        "corr_sig": "Kesimpulan: Terdapat hubungan yang **signifikan** antara penggunaan AI tools dan efektivitas belajar mahasiswa (p < 0.05).",
        "corr_nonsig": "Kesimpulan: Tidak terdapat hubungan yang signifikan antara penggunaan AI tools dan efektivitas belajar mahasiswa (p ≥ 0.05).",
        "viz_title": "📊 Visualisasi Distribusi dan Hubungan X_total & Y_total",
        "hist_x_title": "Histogram X_total (Penggunaan AI Tools)",
        "hist_y_title": "Histogram Y_total (Efektivitas Belajar)",
        "hist_xlabel_x": "X_total",
        "hist_xlabel_y": "Y_total",
        "hist_ylabel": "Frekuensi",
        "scatter_title": "Scatter Plot: X_total vs Y_total",
        "scatter_xlabel": "X_total (Penggunaan AI Tools)",
        "scatter_ylabel": "Y_total (Efektivitas Belajar)",
        "download_title": "📥 Download Data dengan Skor Komposit",
        "download_label": "Download CSV (data + X_total & Y_total)",
    },
    "en": {
        "lang_label": "Language",
        "lang_id": "Indonesian",
        "lang_en": "English",
        "app_title": "📊 Analysis of the Impact of AI Tools Usage on Students' Learning Effectiveness",
        "intro": """
This app will:
- Read a CSV file exported from Google Forms about AI tools usage
- Compute composite scores:
  - **X_total** = AI tools usage score (10 items)
  - **Y_total** = learning effectiveness score (10 items)
- Show descriptive statistics
- Test normality (Shapiro-Wilk)
- Compute correlation (Pearson / Spearman) between X_total and Y_total
- Show visualizations (histogram & scatter plot)
        """,
        "sidebar_title": "Analysis Navigation",
        "page_overview": "Overview & Data",
        "page_desc": "Descriptive Statistics",
        "page_reliability": "Reliability & Normality",
        "page_corr": "Correlation & Visualization",
        "page_download": "Download Data",
        "upload_label": "📤 Upload CSV file (exported from Google Forms)",
        "upload_info": "Please upload a CSV file first (e.g., Survey_ai.csv).",
        "preview_title": "👀 Data Preview (first 5 rows)",
        "missing_cols_error": "Some required columns are missing in the CSV.",
        "missing_cols_hint": "Make sure the question texts in Google Forms are not changed when exporting the CSV.",
        "respondent_after_clean": "Number of respondents after cleaning (min 18/20 items filled):",
        "no_valid_resp": "No respondents met the valid criteria (>=18 items answered).",
        "desc_item_title": "📌 Descriptive Statistics — Item-level",
        "desc_comp_title": "📌 Descriptive Statistics — Composite Scores (X_total & Y_total)",
        "reliability_title": "🧪 Reliability (Cronbach's Alpha)",
        "alpha_x_label": "α for AI Tools Usage (X):",
        "alpha_y_label": "α for Learning Effectiveness (Y):",
        "normality_title": "🧪 Normality Test (Shapiro-Wilk) on X_total and Y_total",
        "normality_x_label": "X_total",
        "normality_y_label": "Y_total",
        "corr_title": "📈 Correlation between AI Usage (X_total) and Learning Effectiveness (Y_total)",
        "corr_method_pearson": "Correlation method: **Pearson Correlation** (data are normal).",
        "corr_method_spearman": "Correlation method: **Spearman Rank Correlation** (data are not normal).",
        "corr_coef_label": "Correlation coefficient (r):",
        "p_value_label": "p-value:",
        "corr_sig": "Conclusion: There is a **significant** relationship between AI tools usage and students' learning effectiveness (p < 0.05).",
        "corr_nonsig": "Conclusion: There is **no significant** relationship between AI tools usage and students' learning effectiveness (p ≥ 0.05).",
        "viz_title": "📊 Distribution and Relationship of X_total & Y_total",
        "hist_x_title": "Histogram of X_total (AI Tools Usage)",
        "hist_y_title": "Histogram of Y_total (Learning Effectiveness)",
        "hist_xlabel_x": "X_total",
        "hist_xlabel_y": "Y_total",
        "hist_ylabel": "Frequency",
        "scatter_title": "Scatter Plot: X_total vs Y_total",
        "scatter_xlabel": "X_total (AI Tools Usage)",
        "scatter_ylabel": "Y_total (Learning Effectiveness)",
        "download_title": "📥 Download Data with Composite Scores",
        "download_label": "Download CSV (data + X_total & Y_total)",
    },
}

# -----------------------------
# Pilihan bahasa di sidebar
# -----------------------------
lang_code = st.sidebar.radio(
    "Language / Bahasa",
    ("id", "en"),
    format_func=lambda x: "Bahasa Indonesia" if x == "id" else "English"
)

T = STRINGS[lang_code]  # shortcut

# -----------------------------
# Judul & Intro
# -----------------------------
st.title(T["app_title"])
st.write(T["intro"])

# -----------------------------
# Navigasi halaman (terpisah)
# -----------------------------
page_key = st.sidebar.radio(
    T["sidebar_title"],
    ("overview", "desc", "reliability", "corr", "download"),
    format_func=lambda x: {
        "overview": T["page_overview"],
        "desc": T["page_desc"],
        "reliability": T["page_reliability"],
        "corr": T["page_corr"],
        "download": T["page_download"],
    }[x]
)

# -----------------------------
# Upload CSV (tetap di atas, dipakai semua halaman)
# -----------------------------
uploaded_file = st.file_uploader(
    T["upload_label"], type=["csv"]
)

if uploaded_file is None:
    st.info(T["upload_info"])
    st.stop()

# Baca CSV
df = pd.read_csv(uploaded_file)

# Preview hanya di halaman "overview"
if page_key == "overview":
    st.subheader(T["preview_title"])
    st.dataframe(df.head())

# Normalisasi nama kolom (hapus spasi di awal/akhir)
df.columns = df.columns.str.strip()

# -----------------------------
# Definisi kolom item X dan Y
# -----------------------------
X_COLS = [
    "Saya menggunakan AI tools untuk membantu memahami materi kuliah.",
    "AI tools membantu saya menyelesaikan tugas lebih cepat.",
    "AI tools membuat saya lebih mudah menemukan penjelasan konsep.",
    "Saya menggunakan AI tools secara rutin saat belajar mandiri.",
    "AI tools membantu saya merangkum materi kuliah.",
    "Saya menggunakan AI tools untuk mendapatkan ide saat mengerjakan tugas.",
    "AI tools membuat proses belajar saya terasa lebih efisien.",
    "Saya merasa lebih percaya diri belajar dengan bantuan AI tools.",
    "AI tools membantu saya memperbaiki kesalahan dalam tugas atau laporan.",
    "Saya merasa kualitas hasil belajar saya meningkat dengan bantuan AI Tools."
]

Y_COLS = [
    "Saya mampu memahami materi kuliah dengan baik.",
    "Saya dapat menyelesaikan tugas tepat waktu.",
    "Saya mampu fokus saat belajar.",
    "Metode belajar saya terasa semakin efektif.",
    "Produktivitas belajar saya meningkat.",
    "Saya dapat meninjau materi dengan lebih terstruktur.",
    "Saya mampu mengatur waktu belajar dengan baik.",
    "Saya mampu mengingat materi pembelajaran dengan lebih baik.",
    "Saya dapat menyelesaikan lebih banyak materi dalam waktu yang sama.",
    "Saya merasa hasil belajar saya meningkat secara keseluruhan."
]

# Cek kolom
missing_x = [c for c in X_COLS if c not in df.columns]
missing_y = [c for c in Y_COLS if c not in df.columns]

if missing_x or missing_y:
    st.error(T["missing_cols_error"])
    st.write(f"Missing kolom X (AI Tools): {missing_x}")
    st.write(f"Missing kolom Y (Efektivitas Belajar): {missing_y}")
    st.info(T["missing_cols_hint"])
    st.stop()

# -----------------------------
# Konversi ke numerik & cleaning
# -----------------------------
for c in X_COLS + Y_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

items_df = df[X_COLS + Y_COLS].copy()
items_df["valid_count"] = items_df.notna().sum(axis=1)
df_clean = items_df[items_df["valid_count"] >= 18].copy()
df_clean = df_clean.drop(columns=["valid_count"])

st.success(f"{T['respondent_after_clean']} {len(df_clean)}")

if len(df_clean) == 0:
    st.error(T["no_valid_resp"])
    st.stop()

# Skor komposit
df_clean["X_total"] = df_clean[X_COLS].sum(axis=1)
df_clean["Y_total"] = df_clean[Y_COLS].sum(axis=1)
df_clean["X_mean"] = df_clean[X_COLS].mean(axis=1)
df_clean["Y_mean"] = df_clean[Y_COLS].mean(axis=1)

# -----------------------------
# Fungsi Cronbach's Alpha
# -----------------------------
def cronbach_alpha(df_items: pd.DataFrame) -> float:
    df_items = df_items.dropna(axis=0, how="any")
    k = df_items.shape[1]
    if k <= 1 or df_items.shape[0] == 0:
        return np.nan
    item_var = df_items.var(axis=0, ddof=1)
    total_var = df_items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    alpha = (k / (k - 1)) * (1 - item_var.sum() / total_var)
    return alpha

# -----------------------------
# Halaman: Statistik Deskriptif
# -----------------------------
if page_key == "desc":
    st.subheader(T["desc_item_title"])
    st.write(df_clean[X_COLS + Y_COLS].describe())

    st.subheader(T["desc_comp_title"])
    st.write(df_clean[["X_total", "Y_total", "X_mean", "Y_mean"]].describe())

# -----------------------------
# Halaman: Reliabilitas & Normalitas
# -----------------------------
if page_key == "reliability":
    st.subheader(T["reliability_title"])
    alpha_x = cronbach_alpha(df_clean[X_COLS])
    alpha_y = cronbach_alpha(df_clean[Y_COLS])

    st.write(f"{T['alpha_x_label']} **{alpha_x:.4f}**")
    st.write(f"{T['alpha_y_label']} **{alpha_y:.4f}**")

    st.subheader(T["normality_title"])
    shapiro_x = shapiro(df_clean["X_total"])
    shapiro_y = shapiro(df_clean["Y_total"])

    st.write(f"{T['normality_x_label']} — W = {shapiro_x.statistic:.4f}, p = {shapiro_x.pvalue:.6f}")
    st.write(f"{T['normality_y_label']} — W = {shapiro_y.statistic:.4f}, p = {shapiro_y.pvalue:.6f}")

# -----------------------------
# Halaman: Korelasi & Visualisasi
# -----------------------------
if page_key == "corr":
    # tentukan metode berdasarkan normalitas
    shapiro_x = shapiro(df_clean["X_total"])
    shapiro_y = shapiro(df_clean["Y_total"])
    if shapiro_x.pvalue > 0.05 and shapiro_y.pvalue > 0.05:
        method = "pearson"
    else:
        method = "spearman"

    st.subheader(T["corr_title"])

    if method == "pearson":
        r, p = pearsonr(df_clean["X_total"], df_clean["Y_total"])
        st.write(T["corr_method_pearson"])
    else:
        r, p = spearmanr(df_clean["X_total"], df_clean["Y_total"])
        st.write(T["corr_method_spearman"])

    st.write(f"{T['corr_coef_label']} **{r:.4f}**")
    st.write(f"{T['p_value_label']} **{p:.6f}**")

    if p < 0.05:
        st.success(T["corr_sig"])
    else:
        st.warning(T["corr_nonsig"])

    # Visualisasi
    st.subheader(T["viz_title"])

    # Histogram X_total
    fig1, ax1 = plt.subplots()
    ax1.hist(df_clean["X_total"], bins=10)
    ax1.set_title(T["hist_x_title"])
    ax1.set_xlabel(T["hist_xlabel_x"])
    ax1.set_ylabel(T["hist_ylabel"])
    st.pyplot(fig1)

    # Histogram Y_total
    fig2, ax2 = plt.subplots()
    ax2.hist(df_clean["Y_total"], bins=10)
    ax2.set_title(T["hist_y_title"])
    ax2.set_xlabel(T["hist_xlabel_y"])
    ax2.set_ylabel(T["hist_ylabel"])
    st.pyplot(fig2)

    # Scatter plot X_total vs Y_total
    fig3, ax3 = plt.subplots()
    ax3.scatter(df_clean["X_total"], df_clean["Y_total"])
    ax3.set_title(T["scatter_title"])
    ax3.set_xlabel(T["scatter_xlabel"])
    ax3.set_ylabel(T["scatter_ylabel"])
    st.pyplot(fig3)

# -----------------------------
# Halaman: Download Data
# -----------------------------
if page_key == "download":
    st.subheader(T["download_title"])
    csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=T["download_label"],
        data=csv_bytes,
        file_name="cleaned_survey_with_composites.csv",
        mime="text/csv"
    )
