# 1_homebase_prodi.py
# Fuzzy Matching Prodi ke 2 Referensi (Token-aware & Matrix-mapped)
# - Referensi 1: Daftar-Rumpun-Pohon-dan-Cabang-Ilmu.xlsx (kolom: "Cabang Ilmu")
# - Referensi 2: DDC_Kode_Deskripsi.xlsx (kolom: "DDC")
# - Baca sheet pertama (tidak ada input nama sheet)
# - Tampilkan proses & hasil di 2 tab terpisah
# - Unduh 1 Excel sheet gabungan dengan kolom:
#   Nama Dosen | NUPTK | NAMA PRODI | Best Match Cabang Ilmu | Score Cabang Ilmu | Best Match DDC | Score DDC
# - Tambahan: Progress bar + estimasi waktu (mm:ss) selama perhitungan fuzzy
# - Perbaikan: deteksi kolom "Nama Dosen", "NUPTK", "NAMA PRODI" secara robust (alias, case-insensitive)

import io
import re
import time
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Fuzzy Matching Prodi → (Cabang Ilmu & DDC)",
    page_icon="🧠",
    layout="wide"
)
st.title("🧠 Fuzzy Matching Prodi → (Cabang Ilmu & DDC)")
st.caption("Dua referensi, dua tab analisis, satu berkas hasil gabungan. Sheet selalu membaca **sheet pertama**.")

# =========================
# UTIL & KONSTANTA
# =========================
_ID_STOPWORDS_ID = {
    "dan","atau","dengan","untuk","pada","di","ke","dari","yang","ilmu","program",
    "studi","prodi","teknik","jurusan","fakultas","universitas"
}
_norm_rx = re.compile(r"[^a-z0-9\s]+")

def _normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = _norm_rx.sub(" ", s)
    toks = [t for t in s.split() if t and t not in _ID_STOPWORDS_ID]
    return " ".join(toks)

def _build_tfidf(ref_list_norm):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(ref_list_norm)
    return vec, X

def _composite_score(a: str, b: str) -> dict:
    s_base  = fuzz.ratio(a, b)
    s_tsort = fuzz.token_sort_ratio(a, b)
    s_tset  = fuzz.token_set_ratio(a, b)
    s_part  = fuzz.partial_ratio(a, b)
    return {
        "rf_ratio": s_base,
        "rf_token_sort": s_tsort,
        "rf_token_set": s_tset,
        "rf_partial": s_part,
    }

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _fmt_mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"

# =========================
# FUNGSI INTI: RUN FUZZY
# =========================
def run_fuzzy_matching(df_src: pd.DataFrame,
                       ref_series: pd.Series,
                       topk_limit: int = 3,
                       progress_title: str = "Memproses"):
    """
    Menghasilkan tuple (df_hasil, topk_list) di mana:
      - df_hasil = DataFrame dengan kolom: BestMatch, Score, Top3, Top3Score
      - topk_list = list isi kandidat (untuk debugging/opsional)
    Sumber wajib punya kolom 'NAMA PRODI'.
    ref_series = Series teks referensi (sudah dipilih kolomnya).
    """

    # Guard sumber
    if "NAMA PRODI" not in df_src.columns:
        out = pd.DataFrame({
            "BestMatch": ["" for _ in range(len(df_src))],
            "Score": [0.0 for _ in range(len(df_src))],
            "Top3": ["" for _ in range(len(df_src))],
            "Top3Score": ["" for _ in range(len(df_src))]
        })
        return out, []

    # Guard referensi
    ref_series = ref_series.dropna().astype(str)
    if ref_series.empty:
        out = pd.DataFrame({
            "BestMatch": ["" for _ in range(len(df_src))],
            "Score": [0.0 for _ in range(len(df_src))],
            "Top3": ["" for _ in range(len(df_src))],
            "Top3Score": ["" for _ in range(len(df_src))]
        })
        return out, []

    ref_raw = ref_series.tolist()
    ref_norm = [_normalize(x) for x in ref_raw]
    vec, X_ref = _build_tfidf(ref_norm)

    # Bobot komposit linear
    w_ratio, w_tsort, w_tset, w_part, w_cos = 0.10, 0.25, 0.35, 0.10, 0.20

    def _score_against_all(q_raw: str):
        qn = _normalize(q_raw)
        if not qn:
            return []
            
        # 1. Isolasi 10 kandidat awal dengan irisan token tertinggi
        prelim = process.extract(qn, ref_norm, scorer=fuzz.token_set_ratio, limit=10)
        cand_idx = [idx for _, _, idx in prelim]
        cand = []
        
        # 2. Transformasi kueri dan ekstraksi array skor Cosine untuk seluruh indeks referensi
        qv = vec.transform([qn])
        cos_array = (qv @ X_ref.T).toarray().ravel()
        
        # 3. Komputasi nilai komposit hibrida khusus untuk kandidat terpilih
        for idx in cand_idx:
            r_raw = ref_raw[idx]
            rn = ref_norm[idx]
            rf = _composite_score(qn, rn)
            
            # Look-up nilai kedekatan ruang vektor berdasarkan indeks spesifik kandidat
            spesifik_cos = cos_array[idx] if idx < len(cos_array) else 0.0
            
            comp = (w_ratio * rf["rf_ratio"] +
                    w_tsort * rf["rf_token_sort"] +
                    w_tset * rf["rf_token_set"] +
                    w_part * rf["rf_partial"] +
                    w_cos * (spesifik_cos * 100.0))
            cand.append((r_raw, comp))
            
        cand.sort(key=lambda x: x[1], reverse=True)
        return cand[:topk_limit] if cand else []

    best, score, top3, top3s = [], [], [], []
    topk_debug = []

    # ---- Progress bar & ETA ----
    total = len(df_src)
    prog = st.progress(0)
    eta_txt = st.empty()
    start = time.time()
    eta_txt.markdown(f"**{progress_title}** · ETA: menghitung…")

    for i, name in enumerate(df_src["NAMA PRODI"].astype(str).tolist(), start=1):
        cand = _score_against_all(name)
        topk_debug.append(cand)
        if cand:
            best.append(cand[0][0])
            score.append(round(_safe_float(cand[0][1]), 2))
            top3.append("; ".join([c[0] for c in cand]))
            top3s.append("; ".join([str(round(_safe_float(c[1]), 2)) for c in cand]))
        else:
            best.append("")
            score.append(0.0)
            top3.append("")
            top3s.append("")

        # Perhitungan estimasi waktu (ETA)
        elapsed = time.time() - start
        avg_per_item = elapsed / i
        remaining_items = total - i
        eta_sec = avg_per_item * remaining_items
        prog.progress(i / total)
        eta_txt.markdown(f"**{progress_title}** · {i}/{total} · ETA: `{_fmt_mmss(eta_sec)}`")

    prog.progress(1.0)
    eta_txt.markdown(f"**{progress_title}** · Selesai dalam `{_fmt_mmss(time.time() - start)}`")

    return (
        pd.DataFrame({
            "BestMatch": best,
            "Score": score,
            "Top3": top3,
            "Top3Score": top3s
        }),
        topk_debug
    )

# =========================
# EXPORT UTIL
# =========================
def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Hasil") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buffer.getvalue()

# =========================
# UI: UPLOAD & PARAMETER
# =========================
st.subheader("📂 Unggah Berkas")
c1, c2 = st.columns(2)
with c1:
    src_file = st.file_uploader("Data Sumber (berisi kolom **NAMA PRODI** + (opsional) Nama Dosen & NUPTK)", type=["xlsx"], key="src")
with c2:
    topk_limit = st.number_input("Simpan Top-K kandidat", min_value=1, max_value=10, value=3, step=1)

st.markdown("#### 📚 Unggah 2 Data Referensi (sheet pertama akan dibaca otomatis)")
r1, r2 = st.columns(2)
with r1:
    ref_rumpun_file = st.file_uploader("Daftar-Rumpun-Pohon-dan-Cabang-Ilmu.xlsx", type=["xlsx"], key="ref_rumpun")
with r2:
    ref_ddc_file = st.file_uploader("DDC_Kode_Deskripsi.xlsx", type=["xlsx"], key="ref_ddc")

st.markdown("---")

# =========================
# PROSES
# =========================
btn = st.button("🚀 Jalankan Fuzzy Matching", use_container_width=True)

if btn:
    # Validasi Berkas Input
    if not src_file or not ref_rumpun_file or not ref_ddc_file:
        st.error("Mohon unggah **3 berkas**: 1) Sumber, 2) Daftar-Rumpun-Pohon-dan-Cabang-Ilmu, 3) DDC_Kode_Deskripsi.")
        st.stop()

    try:
        with st.spinner("Membaca file sumber (sheet pertama)..."):
            df_src = pd.read_excel(src_file)
        with st.spinner("Membaca Daftar-Rumpun-Pohon-dan-Cabang-Ilmu (sheet pertama)..."):
            df_rumpun = pd.read_excel(ref_rumpun_file)
        with st.spinner("Membaca DDC_Kode_Deskripsi (sheet pertama)..."):
            df_ddc = pd.read_excel(ref_ddc_file)
    except Exception as e:
        st.error(f"Gagal membaca Excel: {e}")
        st.stop()

    if df_src.empty or df_rumpun.empty or df_ddc.empty:
        st.error("Salah satu file yang diunggah kosong.")
        st.stop()

    # Validasi Keberadaan Struktur Kolom Utama
    if "Cabang Ilmu" not in df_rumpun.columns:
        st.error("Kolom 'Cabang Ilmu' tidak ditemukan pada file Daftar-Rumpun-Pohon-dan-Cabang-Ilmu.xlsx")
        st.stop()
    if "DDC" not in df_ddc.columns:
        st.error("Kolom 'DDC' tidak ditemukan pada file DDC_Kode_Deskripsi.xlsx")
        st.stop()
    if "NAMA PRODI" not in df_src.columns:
        st.error("Kolom 'NAMA PRODI' tidak ditemukan pada file sumber.")
        st.stop()

    # Konstruksi Konten Multitab
    tab1, tab2 = st.tabs(["🌿 Cabang Ilmu", "📚 DDC"])

    with tab1:
        st.markdown("**Referensi:** `Cabang Ilmu` dari Daftar-Rumpun-Pohon-dan-Cabang-Ilmu.xlsx")
        t0 = time.time()
        df_res_cab, _ = run_fuzzy_matching(
            df_src, df_rumpun["Cabang Ilmu"],
            topk_limit=int(topk_limit),
            progress_title="Cabang Ilmu"
        )
        dt1 = time.time() - t0
        st.success(f"Selesai (Cabang Ilmu) dalam {dt1:.2f} detik.")
        st.dataframe(
            pd.concat([df_src.reset_index(drop=True),
                       df_res_cab.rename(columns={
                           "BestMatch": "Best Match Cabang Ilmu",
                           "Score": "Score Cabang Ilmu",
                           "Top3": "Top3 Cabang Ilmu",
                           "Top3Score": "Top3 Score Cabang Ilmu"
                       })], axis=1),
            use_container_width=True
        )

    with tab2:
        st.markdown("**Referensi:** `DDC` dari DDC_Kode_Deskripsi.xlsx")
        t0 = time.time()
        df_res_ddc, _ = run_fuzzy_matching(
            df_src, df_ddc["DDC"],
            topk_limit=int(topk_limit),
            progress_title="DDC"
        )
        dt2 = time.time() - t0
        st.success(f"Selesai (DDC) dalam {dt2:.2f} detik.")
        st.dataframe(
            pd.concat([df_src.reset_index(drop=True),
                       df_res_ddc.rename(columns={
                           "BestMatch": "Best Match DDC",
                           "Score": "Score DDC",
                           "Top3": "Top3 DDC",
                           "Top3Score": "Top3 Score DDC"
                       })], axis=1),
            use_container_width=True
        )

    # =========================
    # PARSING HEADER SECARA ROBUST
    # =========================
    def _norm_header(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())

    def _pick_col_smart(df: pd.DataFrame, candidates: list[str], out_name: str) -> pd.Series:
        norm_map = {_norm_header(c): c for c in df.columns}
        for cand in candidates:
            norm = _norm_header(cand)
            if norm in norm_map:
                s = df[norm_map[norm]].astype(str).fillna("")
                s.name = out_name
                return s
        return pd.Series([""] * len(df), name=out_name, dtype="object")

    # =========================
    # KONSOLIDASI OUTPUT GABUNGAN (1 SHEET)
    # =========================
    st.markdown("---")
    st.subheader("📄 Hasil Gabungan (1 Sheet)")

    NAMA_DOSEN_ALIASES = [
        "Nama Dosen","NAMA DOSEN","Nama","Nama Lengkap","Nama_Lengkap","NamaDosen",
        "NAMA_DOSEN","nm_dosen","nama_dosen","nama dossen","DOSEN"
    ]
    NUPTK_ALIASES = [
        "NUPTK","No NUPTK","NUPTK DOSEN","nuptk","no_nuptk","NOMOR NUPTK","Nomor NUPTK"
    ]
    PRODI_ALIASES = [
        "NAMA PRODI","Nama Prodi","Prodi","Program Studi","PROGRAM STUDI","nama_prodi","nama prodi"
    ]

    nama_dosen = _pick_col_smart(df_src, NAMA_DOSEN_ALIASES, "Nama Dosen")
    nuptk      = _pick_col_smart(df_src, NUPTK_ALIASES, "NUPTK")
    nama_prodi = _pick_col_smart(df_src, PRODI_ALIASES, "NAMA PRODI")

    base = pd.DataFrame({
        "Nama Dosen": nama_dosen,
        "NUPTK": nuptk,
        "NAMA PRODI": nama_prodi,
    })

    combined = pd.concat([
        base.reset_index(drop=True),
        df_res_cab[["BestMatch", "Score"]].rename(columns={
            "BestMatch": "Best Match Cabang Ilmu",
            "Score": "Score Cabang Ilmu"
        }).reset_index(drop=True),
        df_res_ddc[["BestMatch", "Score"]].rename(columns={
            "BestMatch": "Best Match DDC",
            "Score": "Score DDC"
        }).reset_index(drop=True)
    ], axis=1)

    final_cols = ["Nama Dosen", "NUPTK", "NAMA PRODI", "Best Match Cabang Ilmu", "Score Cabang Ilmu", "Best Match DDC", "Score DDC"]
    st.dataframe(combined[final_cols], use_container_width=True)

    # Penyimpanan ke dalam stream memory bytes
    xlsx_bytes = to_excel_bytes(combined[final_cols], sheet_name="Gabungan_CabangIlmu_DDC")
    
    st.download_button(
        "📥 Download Hasil Gabungan (Excel, 1 Sheet)",
        data=xlsx_bytes,
        file_name="hasil_fuzzy_gabungan_cabangilmu_ddc.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    st.caption(
        "Catatan: Tab **Cabang Ilmu** dan **DDC** menampilkan hasil dengan kolom Top-3 untuk analitik. "
        "Berkas gabungan berisi 7 kolom inti yang diekstraksi menggunakan pencocokan alias kueri robust."
    )
