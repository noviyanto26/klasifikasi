import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import io, time, re

# ========================= 
# KONFIGURASI & UTILITAS 
# =========================
st.set_page_config(
    page_title="Analisis Mata Kuliah Dosen (Fuzzy Taxo-Folk)",
    page_icon="📊",
    layout="wide"
)

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
def run_fuzzy_token_aware(df_src: pd.DataFrame,
                          ref_series: pd.Series,
                          src_col: str,
                          topk_limit: int = 3,
                          progress_title: str = "Memproses"):
    """
    Fuzzy token-aware + Matrix-mapped TF-IDF Cosine (Taxo-Folk Hybrid Engine).
    """
    if src_col not in df_src.columns:
        n = len(df_src)
        return pd.DataFrame({
            "BestMatch": [""]*n, "Score": [0.0]*n, "Top3": [""]*n, "Top3Score": [""]*n
        })

    ref_series = ref_series.dropna().astype(str)
    if ref_series.empty:
        n = len(df_src)
        return pd.DataFrame({
            "BestMatch": [""]*n, "Score": [0.0]*n, "Top3": [""]*n, "Top3Score": [""]*n
        })

    ref_raw = ref_series.tolist()
    ref_norm = [_normalize(x) for x in ref_raw]
    vec, X_ref = _build_tfidf(ref_norm)

    w_ratio, w_tsort, w_tset, w_part, w_cos = 0.10, 0.25, 0.35, 0.10, 0.20

    def _score_against_all(q_raw: str):
        qn = _normalize(q_raw)
        if not qn:
            return []
            
        prelim = process.extract(qn, ref_norm, scorer=fuzz.token_set_ratio, limit=10)
        cand_idx = [idx for _, _, idx in prelim]
        cand = []
        
        # Ekstraksi array matriks spesifik untuk akurasi token-aware
        qv = vec.transform([qn])
        cos_array = (qv @ X_ref.T).toarray().ravel()
        
        for idx in cand_idx:
            r_raw = ref_raw[idx]
            rn = ref_norm[idx]
            rf = _composite_score(qn, rn)
            
            # Look-up kedekatan spasial berdasar indeks kandidat
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

    total = len(df_src)
    prog = st.progress(0)
    eta_txt = st.empty()
    start = time.time()
    eta_txt.markdown(f"**{progress_title}** · ETA: menghitung…")

    for i, val in enumerate(df_src[src_col].astype(str).tolist(), start=1):
        cand = _score_against_all(val)
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
            
        elapsed = time.time() - start
        avg = elapsed / i
        remain = avg * (total - i)
        prog.progress(i/total)
        eta_txt.markdown(f"**{progress_title}** · {i}/{total} · ETA: `{_fmt_mmss(remain)}`")

    prog.progress(1.0)
    eta_txt.markdown(f"**{progress_title}** · Selesai dalam `{_fmt_mmss(time.time() - start)}`")

    return pd.DataFrame({
        "BestMatch": best, "Score": score, "Top3": top3, "Top3Score": top3s
    })

# ========================= 
# HELPER I/O & ROBUST PARSING 
# =========================
def _read_any(file):
    if file.name.lower().endswith("csv"):
        return pd.read_csv(file)
    return pd.read_excel(file) 

def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Hasil") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buffer.getvalue()

def _sig(file):
    if file is None: return None
    size = getattr(file, "size", None)
    if size is None:
        try: size = len(file.getvalue())
        except Exception: size = None
    return (file.name, size)

def _auto_reset_if_changed(key, new_sig):
    prev_sig = st.session_state.get(key)
    if prev_sig != new_sig:
        if prev_sig is not None and st.session_state.get("run_analysis", False):
            st.session_state.run_analysis = False
            st.info("File berubah → analisis di-reset. Klik **Mulai Analisis** lagi.")
        st.session_state[key] = new_sig

def _norm_header(s: str) -> str: 
    return re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    norm_map = {_norm_header(c): c for c in df.columns}
    for cand in candidates:
        norm = _norm_header(cand)
        if norm in norm_map: return norm_map[norm]
    return None

# ========================= 
# SESSION STATE 
# =========================
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False
for k in ("sig_matkul", "sig_rumpun", "sig_ddc"):
    st.session_state.setdefault(k, None)

# ========================= 
# UI (Upload di Halaman Utama) 
# =========================
st.title("📊 Analisis Mata Kuliah Dosen (Taxo-Folk)")

st.subheader("📂 Unggah Data")
col_u1, col_u2, col_u3 = st.columns(3)
with col_u1:
    matkul_file = st.file_uploader("Unggah Data Matkul Dosen", type=["xlsx", "xls", "csv"], key="up_matkul")
with col_u2:
    rumpun_file = st.file_uploader("Unggah Daftar-Rumpun-Pohon-dan-Cabang-Ilmu.xlsx", type=["xlsx", "xls", "csv"], key="up_rumpun")
with col_u3:
    ddc_file    = st.file_uploader("Unggah DDC_Kode_Deskripsi.xlsx", type=["xlsx", "xls", "csv"], key="up_ddc")

_auto_reset_if_changed("sig_matkul", _sig(matkul_file))
_auto_reset_if_changed("sig_rumpun", _sig(rumpun_file))
_auto_reset_if_changed("sig_ddc", _sig(ddc_file))

st.markdown("---")

st.subheader("⚙️ Pengaturan & Eksekusi")
col_p1, col_p2, col_p3 = st.columns([1,1,2])
with col_p1:
    topk_limit = st.number_input("Top-K kandidat (fuzzy)", min_value=1, max_value=10, value=3, step=1)
with col_p2:
    reset_btn = st.button("🔄 Reset", use_container_width=True)
with col_p3:
    run_btn = st.button("🚀 Mulai Analisis (Taxo-Folk Engine)", type="primary", use_container_width=True, disabled=(matkul_file is None))

if run_btn: st.session_state.run_analysis = True
if reset_btn: st.session_state.run_analysis = False

if matkul_file is None:
    st.info("Unggah **Data Matkul Dosen** terlebih dahulu untuk mengaktifkan tombol Mulai Analisis.")

# ========================= 
# MULTI-TABS 
# =========================
tab1, tab2, tab3 = st.tabs(["📘 Data Matkul", "🌿 Cabang Ilmu", "📚 DDC"])
hasil_matkul = None
hasil_rumpun = None
hasil_ddc = None

# ---- Tab 1: Verifikasi Data ----
with tab1:
    st.header("📘 Verifikasi Data Matkul")
    if not st.session_state.run_analysis:
        st.info("Klik **Mulai Analisis** di atas untuk memulai.")
    else:
        if matkul_file is not None:
            matkul_df = _read_any(matkul_file)
            st.success("Data berhasil dimuat. (Proses Clustering K-Means dinonaktifkan sesuai arsitektur Taxo-Folk)")
            st.dataframe(matkul_df, use_container_width=True)
            hasil_matkul = matkul_df.copy()
        else:
            st.warning("File **Data Matkul Dosen** belum diunggah.")

# ---- Tab 2: Cabang Ilmu ----
with tab2:
    st.header("🌿 Fuzzy Matching ke Cabang Ilmu (Prioritas Taxo-Folk)")
    if not st.session_state.run_analysis:
        st.info("Klik **Mulai Analisis** di atas untuk memulai.")
    else:
        if matkul_file is None or rumpun_file is None:
            st.warning("File Data Matkul atau Referensi Cabang Ilmu belum lengkap.")
        else:
            if hasil_matkul is None:
                matkul_df = _read_any(matkul_file)
                hasil_matkul = matkul_df.copy()

            rumpun_df = _read_any(rumpun_file)
            
            src_col_name = _find_col(hasil_matkul, ["NAMA MATKUL", "Nama Matkul", "Matkul", "mata kuliah", "mata_kuliah", "nama_matkul"])
            ref_col_name = _find_col(rumpun_df, ["Cabang Ilmu", "Cabang_Ilmu"])

            if not ref_col_name:
                st.error("Kolom **Cabang Ilmu** tidak ditemukan pada file Rumpun.")
            elif not src_col_name:
                st.error("Kolom identitas Mata Kuliah tidak ditemukan pada data referensi Dosen.")
            else:
                start = time.time()
                df_res = run_fuzzy_token_aware(
                    hasil_matkul, rumpun_df[ref_col_name],
                    src_col=src_col_name, topk_limit=int(topk_limit),
                    progress_title="Cabang Ilmu"
                )
                hasil_rumpun = pd.concat(
                    [hasil_matkul.reset_index(drop=True),
                     df_res.rename(columns={
                         "BestMatch": "Best Match Cabang Ilmu",
                         "Score": "Score Cabang Ilmu",
                         "Top3": "Top3 Cabang Ilmu",
                         "Top3Score": "Top3 Score Cabang Ilmu"
                     })], axis=1
                )
                st.success(f"Selesai (Cabang Ilmu) dalam {time.time()-start:.2f} detik.")
                st.dataframe(hasil_rumpun, use_container_width=True)

# ---- Tab 3: DDC ----
with tab3:
    st.header("📚 Fuzzy Matching DDC")
    if not st.session_state.run_analysis:
        st.info("Klik **Mulai Analisis** di atas untuk memulai.")
    else:
        if matkul_file is None or ddc_file is None:
            st.warning("File Data Matkul atau Referensi DDC belum lengkap.")
        else:
            if hasil_matkul is None:
                matkul_df = _read_any(matkul_file)
                hasil_matkul = matkul_df.copy()

            ddc_df = _read_any(ddc_file)
            src_col_name = _find_col(hasil_matkul, ["NAMA MATKUL", "Nama Matkul", "Matkul", "mata kuliah", "mata_kuliah", "nama_matkul"])
            ref_ddc_name = _find_col(ddc_df, ["DDC", "ddc"])

            if not ref_ddc_name:
                st.error("Kolom **DDC** tidak ditemukan pada file DDC.")
            elif not src_col_name:
                st.error("Kolom identitas Mata Kuliah tidak ditemukan pada data referensi Dosen.")
            else:
                start = time.time()
                df_res = run_fuzzy_token_aware(
                    hasil_matkul, ddc_df[ref_ddc_name],
                    src_col=src_col_name, topk_limit=int(topk_limit),
                    progress_title="DDC"
                )
                hasil_ddc = pd.concat(
                    [hasil_matkul.reset_index(drop=True),
                     df_res.rename(columns={
                         "BestMatch": "Best Match DDC",
                         "Score": "Score DDC",
                         "Top3": "Top3 DDC",
                         "Top3Score": "Top3 Score DDC"
                     })], axis=1
                )
                st.success(f"Selesai (DDC) dalam {time.time()-start:.2f} detik.")
                st.dataframe(hasil_ddc, use_container_width=True)

# ========================= 
# EXPORT GABUNGAN (1 SHEET) 
# =========================
if st.session_state.run_analysis and (matkul_file is not None) and (hasil_rumpun is not None) and (hasil_ddc is not None):
    st.markdown("---")
    st.subheader("📄 Hasil Gabungan (1 Sheet)")

    def _pick_col_smart(df: pd.DataFrame, candidates: list[str], out_name: str) -> pd.Series:
        norm_map = {_norm_header(c): c for c in df.columns}
        for cand in candidates:
            key = _norm_header(cand)
            if key in norm_map:
                s = df[norm_map[key]].astype(str).fillna("")
                s.name = out_name
                return s
        return pd.Series([""] * len(df), name=out_name, dtype="object")

    NAMA_DOSEN_ALIASES = ["Nama Dosen","NAMA DOSEN","Nama","Nama Lengkap","nama_dosen","nama dosen","DOSEN"]
    NIDN_ALIASES       = ["NIDN","nidn","No NIDN","no_nidn","NIDN Dosen"]
    MATKUL_ALIASES     = ["NAMA MATKUL","Nama Matkul","Matkul","mata kuliah","mata_kuliah","nama_matkul"]

    nama_dosen  = _pick_col_smart(hasil_matkul, NAMA_DOSEN_ALIASES, "Nama Dosen")
    nidn        = _pick_col_smart(hasil_matkul, NIDN_ALIASES, "NIDN")
    nama_matkul = _pick_col_smart(hasil_matkul, MATKUL_ALIASES, "NAMA MATKUL")

    combined = pd.concat([
        pd.DataFrame({
            "Nama Dosen": nama_dosen,
            "NIDN": nidn,
            "NAMA MATKUL": nama_matkul
        }).reset_index(drop=True),
        hasil_rumpun[["Best Match Cabang Ilmu", "Score Cabang Ilmu"]].reset_index(drop=True),
        hasil_ddc[["Best Match DDC", "Score DDC"]].reset_index(drop=True)
    ], axis=1)

    final_cols = ["Nama Dosen", "NIDN", "NAMA MATKUL", 
                  "Best Match Cabang Ilmu", "Score Cabang Ilmu", 
                  "Best Match DDC", "Score DDC"]
    
    st.dataframe(combined[final_cols], use_container_width=True)

    xlsx_bytes = to_excel_bytes(combined[final_cols], sheet_name="Gabungan_TaxoFolk_DDC")
    
    st.download_button(
        "📥 Download Hasil Gabungan (Excel, 1 Sheet)",
        data=xlsx_bytes,
        file_name="hasil_fuzzy_matkul_gabungan_cabangilmu_ddc.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    st.caption(
        "Berkas gabungan ditarik secara robust dan berisi 7 kolom identitas inti: **Nama Dosen**, **NIDN**, **NAMA MATKUL**, "
        "**Best Match Cabang Ilmu**, **Score Cabang Ilmu**, **Best Match DDC**, **Score DDC**."
    )
