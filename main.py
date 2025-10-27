# main.py — Orkestrasi Proses Takso-Folk (Tanpa Radio Button)
# -------------------------------------------------------------
# Sidebar: kategori → tombol langsung untuk tiap proses
# -------------------------------------------------------------

import io
import os
import runpy
import traceback
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

import streamlit as st

st.set_page_config(page_title="Orkestrasi Takso-Folk menggunakan Agentic LLM", page_icon="🗺️", layout="wide")
st.title("🗺️ Orkestrasi Proses Takso-Folk")
st.caption("Gabungan pipeline: Fuzzy Logic → Topic Map → Klasifikasi")

# ========================= Session State =========================
if "to_run" not in st.session_state:
    st.session_state.to_run = None
if "last_logs" not in st.session_state:
    st.session_state.last_logs = None

# ========================= Konfigurasi & Pemetaan =========================
BASE_DIR = Path(__file__).parent.resolve()

ALIAS = {
    "1_homebase_prodi.py": "Proses Homebase Program Studi",
    "2_rwy_pendidikan.py": "Proses Riwayat Pendidikan",
    "3_rwy_ajar.py": "Proses Riwayat Pengajaran",
    "4_litabmas.py": "Proses Judul Litabmas",
    "5_publikasi.py": "Proses Judul Publikasi",
    "6_scholar.py": "Proses Judul dari Google Scholar",
    "9f_agentic_with_cache.py": "Proses Klasifikasi AI LLM via API",
    }

GROUPS = [
    (
        "⚙️ Proses Fuzzy Logic",
        [
            "1_homebase_prodi.py",
            "2_rwy_pendidikan.py",
            "3_rwy_ajar.py",
        ],
    ),
    (
        "🧭 Proses Topic Map",
        [
            "4_litabmas.py",
            "5_publikasi.py",
            "6_scholar.py",
        ],
    ),
    (
        "🔎 Proses Klasifikasi",
        [
            "9f_agentic_with_cache.py",
        
        ],
    ),
]

# ========================= Utilitas Eksekusi Skrip =========================

def _resolve_script_path(name: str) -> Path:
    return BASE_DIR / name

def run_script(script_path: Path) -> str:
    buffer_out, buffer_err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(buffer_out), redirect_stderr(buffer_err):
            runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as e:
        print(f"[SystemExit] Kode keluar: {getattr(e, 'code', None)}", file=buffer_err)
    except Exception:
        traceback.print_exc(file=buffer_err)
    finally:
        out = buffer_out.getvalue()
        err = buffer_err.getvalue()
        logs = []
        if out.strip():
            logs.append("[STDOUT]\n" + out)
        if err.strip():
            logs.append("[STDERR]\n" + err)
        return "\n\n".join(logs).strip() or "(Tidak ada output)"

# ========================= Sidebar =========================
with st.sidebar:
    st.header("Kategori Proses")
    st.caption("Klik tombol di bawah untuk menjalankan proses.")

    for group_label, files in GROUPS:
        with st.expander(group_label, expanded=False):
            for f in files:
                label = ALIAS.get(f, f)
                fpath = _resolve_script_path(f)
                if fpath.exists():
                    if st.button(f"▶️ {label}", key=f"btn_{f}", use_container_width=True):
                        st.session_state.to_run = {"alias": label, "path": str(fpath)}
                        st.rerun()
                else:
                    st.warning(f"File tidak ditemukan: {f}")

    st.divider()
    if st.button("🧹 Bersihkan Output", use_container_width=True):
        st.session_state.to_run = None
        st.session_state.last_logs = None
        st.rerun()

# ========================= Area Utama (Output) =========================
output = st.container()
if st.session_state.to_run:
    alias = st.session_state.to_run["alias"]
    script_path = Path(st.session_state.to_run["path"])
    with output:
        st.subheader(f"▶️ {alias}")
        with st.spinner(f"Menjalankan: {alias} ..."):
            logs = run_script(script_path)
        st.session_state.last_logs = logs
        with st.expander(f"📜 Log — {alias}", expanded=False):
            st.code(logs)
elif st.session_state.last_logs:
    with output:
        st.info("Tidak ada proses yang berjalan. Menampilkan log terakhir.")
        with st.expander("📜 Log Terakhir", expanded=False):
            st.code(st.session_state.last_logs)
else:
    with output:
        st.info("Pilih salah satu proses di sidebar. Output akan muncul di sini.")
