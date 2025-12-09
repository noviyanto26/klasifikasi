# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import graphviz
import io, time, json, os, re, ast
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

# --- LLM Providers ---
from groq import Groq, APIError, BadRequestError
from openai import OpenAI
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import requests
from dotenv import load_dotenv

# --- PDF & Utils ---
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# --- RAG Libraries ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    st.error("Library RAG belum terinstall. Mohon jalankan: pip install sentence-transformers scikit-learn")
    st.stop()

# ==========================================================
# INIT
# ==========================================================
load_dotenv()
st.set_page_config(page_title="Agentic AI Classification - TaksoFolk", page_icon="🧠", layout="wide")
PG = "pg9aa_"  # Prefix state

for k, v in [
    (PG + "started", False),
    (PG + "df_hasil", None),
    (PG + "df_mapping", None),
    (PG + "excel_bytes", None),
    (PG + "pdf_bytes", None),
    (PG + "json_cache_bytes", None),
    (PG + "hasil_partial", []),
    (PG + "stop_requested", False),
    (PG + "current_provider_index", 0),
    (PG + "taxonomy_embeddings", None), # Cache untuk embedding taksonomi
    (PG + "taxonomy_list", []),         # List valid Cabang Ilmu
]:
    st.session_state.setdefault(k, v)

# ==========================================================
# RAG ENGINE & EMBEDDING MODEL
# ==========================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

class SimpleLocalRAG:
    def __init__(self, texts: List[str], sources: List[str]):
        self.model = load_embedding_model()
        self.texts = texts
        self.sources = sources
        if texts:
            self.embeddings = self.model.encode(texts, show_progress_bar=False)
        else:
            self.embeddings = None

    def retrieve(self, query: str, top_k: int = 5) -> str:
        if not self.texts or self.embeddings is None:
            return "Tidak ada data teks detail."
        
        query_vec = self.model.encode([query])
        similarities = cosine_similarity(query_vec, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score > 0.25: # Threshold relevansi
                txt = self.texts[idx]
                snippet = txt[:400] + "..." if len(txt) > 400 else txt
                results.append(f"- [{self.sources[idx]}] {snippet} (Relevansi: {score:.2f})")
        
        if not results: return "Tidak ditemukan bukti relevan di RAG."
        return "\n".join(results)

# ==========================================================
# TAXONOMY VALIDATOR
# ==========================================================
def prepare_taxonomy_cache(df_mapping: pd.DataFrame):
    """Mengindeks seluruh Cabang Ilmu yang valid dari file mapping."""
    if df_mapping is None: return
    
    # Cari kolom Cabang Ilmu
    df_mapping.columns = [c.strip() for c in df_mapping.columns]
    col_cabang = next((c for c in df_mapping.columns if c.lower() == "cabang ilmu"), None)
    
    if not col_cabang:
        # Fallback jika nama kolom beda, ambil kolom terakhir
        col_cabang = df_mapping.columns[-1]

    valid_list = df_mapping[col_cabang].dropna().unique().astype(str).tolist()
    valid_list = [x.strip() for x in valid_list if x.strip().lower() != 'nan']
    
    st.session_state[PG + "taxonomy_list"] = valid_list
    
    # Hitung embedding sekali saja untuk efisiensi
    model = load_embedding_model()
    st.session_state[PG + "taxonomy_embeddings"] = model.encode(valid_list, show_progress_bar=False)

def validate_and_correct_taxonomy(raw_field: str) -> str:
    """
    Memastikan output sesuai dengan daftar 'Cabang Ilmu'.
    """
    if not raw_field or str(raw_field).lower() == 'nan' or raw_field == "None":
        return None

    valid_list = st.session_state.get(PG + "taxonomy_list", [])
    valid_embeddings = st.session_state.get(PG + "taxonomy_embeddings", None)
    
    # Jika cache belum siap, coba ambil dari mapping langsung (safety net)
    if not valid_list and st.session_state.get(PG + "df_mapping") is not None:
        prepare_taxonomy_cache(st.session_state[PG + "df_mapping"])
        valid_list = st.session_state.get(PG + "taxonomy_list", [])
        valid_embeddings = st.session_state.get(PG + "taxonomy_embeddings", None)

    if not valid_list or valid_embeddings is None:
        return raw_field 

    raw_clean = raw_field.strip()

    # 1. Cek Exact Match (Case Insensitive)
    for item in valid_list:
        if item.lower() == raw_clean.lower():
            return item
            
    # 2. Semantic Search Strict
    model = load_embedding_model()
    query_vec = model.encode([raw_clean])
    similarities = cosine_similarity(query_vec, valid_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    
    if best_score > 0.5: 
        return valid_list[best_idx]
    
    if best_score > 0.4:
        return valid_list[best_idx]

    return raw_field

# ==========================================================
# JSON SANITIZER (ROBUST FIX)
# ==========================================================
def _extract_json_block(text: str) -> Optional[str]:
    if not text: return None
    # Coba cari blok kode markdown
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL)
    if m: return m.group(1)
    
    # Coba cari dari kurawal pembuka sampai penutup terakhir
    start_brace = text.find("{")
    end_brace = text.rfind("}")
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        return text[start_brace : end_brace + 1]
    
    return text

def _clean_json_text(text: str) -> str:
    """Membersihkan format JSON yang rusak sebelum diparsing."""
    # 1. Ganti smart quotes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    # 2. Hapus trailing commas
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # 3. Escape newlines dalam string
    text = re.sub(r'(?<!\\)\n', ' ', text)
    return text

def _loads_json_strict(text: str) -> Any:
    """Parser JSON yang sangat toleran terhadap kesalahan sintaks."""
    if not text: raise ValueError("Empty JSON response")

    # Langkah 1: Extract JSON murni
    cleaned_block = _extract_json_block(text) or text
    
    # Langkah 2: Coba JSON standard
    try:
        return json.loads(cleaned_block)
    except json.JSONDecodeError:
        pass

    # Langkah 3: Bersihkan teks dan coba lagi
    cleaned_text = _clean_json_text(cleaned_block)
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass

    # Langkah 4: Gunakan AST Literal Eval (Python Parser)
    # Ini sering berhasil jika LLM menggunakan single quote atau lupa escape double quote
    try:
        # Ubah null/true/false JSON ke None/True/False Python
        py_syntax = cleaned_text.replace("null", "None").replace("true", "True").replace("false", "False")
        return ast.literal_eval(py_syntax)
    except (ValueError, SyntaxError):
        pass

    # Langkah 5: Upaya terakhir - "Unterminated string" patch
    try:
        return json.loads(cleaned_text + '"}')
    except:
        pass

    raise ValueError(f"Gagal mem-parsing JSON setelah berbagai metode. Raw: {text[:100]}...")

# ==========================================================
# LLM WRAPPERS
# ==========================================================
MAX_OUT_TOKENS = 4096

def _call_openai_compatible(client, model, temp, system, user):
    resp = client.chat.completions.create(
        model=model,
        temperature=temp,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=MAX_OUT_TOKENS,
    )
    content = resp.choices[0].message.content or ""
    return _loads_json_strict(content)

def _call_gemini(client, model, temp, system, user):
    try:
        model_client = genai.GenerativeModel(model)
    except Exception as e:
        raise ProviderUnavailableError(f"Gagal memuat model Google: {model}. Error: {e}")

    # Prompt diperkeras agar JSON valid
    prompt = (
        f"{system}\n\n{user}\n\n"
        "IMPORTANT: Return VALID JSON ONLY. Escape any double quotes inside strings with backslash (e.g. \"text \\\"quote\\\" text\")."
    )
    
    try:
        resp = model_client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temp,
                response_mime_type="application/json",
                max_output_tokens=MAX_OUT_TOKENS,
            ),
        )
        content = resp.text or ""
    except ValueError:
        if hasattr(resp, 'prompt_feedback') and resp.prompt_feedback:
             # Blokir karena safety settings
            raise ProviderQuotaError(f"Google Safety Block: {resp.prompt_feedback}")
        raise ProviderUnavailableError("Google mengembalikan respons kosong/invalid.")
    except Exception as e:
        raise ProviderUnavailableError(f"Google Error: {str(e)}")

    return _loads_json_strict(content)

def _call_databricks(client_key, model_url, temp, system, user):
    headers = {
        "Authorization": f"Bearer {client_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": MAX_OUT_TOKENS,
        "temperature": temp,
    }
    try:
        response = requests.post(model_url, headers=headers, json=payload, timeout=60)
        if response.status_code == 401:
            raise ProviderUnavailableError("Databricks API Key (PAT) tidak valid atau kedaluwarsa.")
        if response.status_code == 403:
            raise ProviderQuotaError("Izin ditolak. Pastikan API Key (PAT) memiliki hak 'Can Query' pada endpoint.")
        if response.status_code == 429:
            raise ProviderQuotaError("Databricks quota/rate limit terlampaui (Error 429).")
        response.raise_for_status()
        resp_json = response.json()
        content = resp_json["choices"][0]["message"]["content"]
        if not content:
            raise ValueError("Respons Databricks kosong.")
        return _loads_json_strict(content)
    except requests.RequestException as e:
        raise ProviderUnavailableError(f"Gagal menghubungi Databricks: {e}")
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise ProviderUnavailableError(f"Gagal mem-parsing respons Databricks. Error: {e}")

class ProviderQuotaError(Exception): ...
class ProviderUnavailableError(Exception): ...

# ==========================================================
# UTILS & VISUAL
# ==========================================================
def _safe_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError, AttributeError):
        return 0.0

def format_eta(seconds: int) -> str:
    if seconds < 0:
        return "00:00"
    jam, sisa = divmod(seconds, 3600)
    menit, detik = divmod(sisa, 60)
    return f"{jam:02d}:{menit:02d}:{detik:02d}" if jam > 0 else f"{menit:02d}:{detik:02d}"

def taksofolk_mapping(bidang: Optional[str], df_mapping: pd.DataFrame) -> List[str]:
    """
    Memetakan bidang ilmu (Cabang Ilmu) ke hierarki:
    [Rumpun Ilmu, Pohon Ilmu 1, Pohon Ilmu 2, Cabang Ilmu]
    """
    if not bidang or str(bidang).lower() == "nan":
        return ["Tidak Diketahui"] * 4
    
    try:
        df_mapping.columns = [c.strip() for c in df_mapping.columns]
        
        col_rumpun = "Rumpun Ilmu"
        col_pohon1 = "Pohon Ilmu 1"
        col_pohon2 = "Pohon Ilmu 2"
        col_cabang = "Cabang Ilmu"
        
        cols_map = {c.lower(): c for c in df_mapping.columns}
        col_cabang = cols_map.get("cabang ilmu", df_mapping.columns[-1]) 
        col_rumpun = cols_map.get("rumpun ilmu", df_mapping.columns[0])
        col_pohon1 = cols_map.get("pohon ilmu 1", df_mapping.columns[1])
        col_pohon2 = cols_map.get("pohon ilmu 2", df_mapping.columns[2])

        mask = df_mapping[col_cabang].astype(str).str.strip().str.lower() == str(bidang).strip().lower()
        
        if mask.any():
            row = df_mapping[mask].iloc[0]
            return [
                str(row.get(col_rumpun, "Lainnya")),
                str(row.get(col_pohon1, "Lainnya")),
                str(row.get(col_pohon2, "Lainnya")),
                str(row.get(col_cabang, bidang))
            ]
        
        return ["Lainnya", "Lainnya", "Lainnya", str(bidang)]
        
    except Exception as e:
        return ["Error", "Error", str(e), str(bidang)]

def build_taksofolk_tree(nama_dosen: str, bidang1: Optional[str], bidang2: Optional[str], df_mapping: pd.DataFrame) -> graphviz.Digraph:
    dot = graphviz.Digraph()
    root_id = "root"
    dot.node(root_id, "Taksonomi Ilmu", style="filled", color="lightgrey", shape="doubleoctagon")

    def clean_id(text):
        if not text: return "unknown"
        return re.sub(r'[^a-zA-Z0-9]', '_', str(text)).strip().lower()

    bidang_unik = []
    if bidang1 and str(bidang1).lower() != 'nan':
        bidang_unik.append(bidang1)
    if bidang2 and str(bidang2).lower() != 'nan' and bidang2 != bidang1:
        bidang_unik.append(bidang2)

    colors = ["#FFD700", "#90EE90", "#ADD8E6", "#FFB6C1"] 
    
    for bidang in bidang_unik:
        levels = taksofolk_mapping(bidang, df_mapping)
        parent_id = root_id
        for i, level_name in enumerate(levels):
            if not level_name or str(level_name).lower() == 'nan': continue
            node_id = f"L{i}_{clean_id(level_name)}"
            is_last_node = (i == len(levels) - 1)
            shape = "ellipse" if is_last_node else "box"
            color = colors[i % len(colors)]
            dot.node(node_id, str(level_name), shape=shape, style="filled", color=color)
            dot.edge(parent_id, node_id)
            parent_id = node_id
    return dot

def export_trees_to_pdf(df_hasil: pd.DataFrame, df_mapping: pd.DataFrame) -> io.BytesIO:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 36
    for _, row in df_hasil.iterrows():
        nama = str(row.get("Lecturer Name", "")).strip()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, height - margin, f"Pohon Takso-Folk: {nama}")
        try:
            dot = build_taksofolk_tree(nama, row.get("Field of Science 1"), row.get("Field of Science 2"), df_mapping)
            img_data = io.BytesIO(dot.pipe(format="png"))
            img = ImageReader(img_data)
            iw, ih = img.getSize()
            scale = min((width - 2 * margin) / iw, (height - 2 * margin - 30) / ih, 1.0)
            dw, dh = iw * scale, ih * scale
            y_pos = height - margin - 30 - dh
            c.drawImage(img, margin, y_pos, width=dw, height=dh, preserveAspectRatio=True)
        except Exception:
            c.drawString(margin, height - margin - 24, "Gagal membuat visual.")
        c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ==========================================================
# DATA COLLECTION (Updated for RAG)
# ==========================================================
def collect_all_evidence(dosen: str, dfs: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, Any], List, Optional[SimpleLocalRAG]]:
    evidence = {}
    rag_texts = []
    rag_sources = []
    candidates = [] 

    def _collect_matches(df, col, source_name):
        found = []
        if df is None: return found
        df.columns = [c.strip() for c in df.columns]
        col_nama = next((c for c in df.columns if c.lower() == 'nama dosen'), None)
        if not col_nama: return found
        
        sub = df[df[col_nama].astype(str) == dosen]
        if sub.empty: return found
        
        keyword = col.lower().split()[0]
        col_score = next((c for c in df.columns if "score" in c.lower() and keyword in c.lower()), None)
        col_best = next((c for c in df.columns if "match" in c.lower() and keyword in c.lower()), None)
        
        if col_score and col_best:
            try:
                sub = sub.sort_values(by=col_score, ascending=False)
                seen_vals = set()
                for _, row in sub.iterrows():
                    val = row[col_best]
                    score = _safe_float(row[col_score])
                    
                    if score > 60.0 and val not in seen_vals:
                        found.append((val, score, source_name))
                        seen_vals.add(val)
            except:
                pass
        return found

    candidates.extend(_collect_matches(dfs["homebase"], "Cabang Ilmu", "homebase"))
    candidates.extend(_collect_matches(dfs["pendidikan"], "Cabang Ilmu", "pendidikan"))
    candidates.extend(_collect_matches(dfs["mengajar"], "Cabang Ilmu", "mengajar"))
    
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    evidence["homebase"] = next((v for v, s, src in candidates if src == "homebase"), None)
    evidence["pendidikan"] = next((v for v, s, src in candidates if src == "pendidikan"), None)
    evidence["mengajar"] = next((v for v, s, src in candidates if src == "mengajar"), None)

    def _get_tags_flexible(df_source, source_label):
        if df_source is None: return []
        col_nama = next((c for c in df_source.columns if c.lower() == 'nama dosen'), None)
        if not col_nama: return []
        sub = df_source[df_source[col_nama].astype(str) == dosen]
        if sub.empty: return []
        keywords = ['topik', 'topic', 'tag', 'tags', 'judul', 'title', 'keywords', 'publikasi', 'research']
        found_cols = [c for c in sub.columns if any(k in c.lower() for k in keywords)]
        found_tags = []
        if found_cols:
            target_col = found_cols[0] 
            raw_list = sub[target_col].dropna().astype(str).tolist()
            for item in raw_list:
                parts = re.split(r'[;,]', item)
                for p in parts:
                    clean_p = p.strip()
                    if len(clean_p) > 2: found_tags.append(clean_p)
        return sorted(list(set(found_tags)))

    evidence["publikasi_tags"] = _get_tags_flexible(dfs["publikasi"], "Publikasi")
    evidence["pengabdian_tags"] = _get_tags_flexible(dfs["pengabdian"], "Pengabdian")
    evidence["scholar_tags"] = _get_tags_flexible(dfs["scholar"], "Scholar")

    def _collect_text_rag(df, src_name):
        if df is not None:
            df.columns = [c.strip() for c in df.columns]
            col_nama = next((c for c in df.columns if c.lower() == 'nama dosen'), None)
            if col_nama:
                sub = df[df[col_nama].astype(str) == dosen]
                keywords = ['judul', 'title', 'abstract', 'deskripsi', 'topik', 'kalimat', 'publikasi', 'research']
                cols = [c for c in sub.columns if any(k in c.lower() for k in keywords)]
                if cols and not sub.empty:
                    for col in cols:
                        texts = sub[col].dropna().astype(str).tolist()
                        for t in texts:
                            if len(t) > 5:
                                rag_texts.append(t)
                                rag_sources.append(src_name)

    _collect_text_rag(dfs["publikasi"], "Publikasi")
    _collect_text_rag(dfs["pengabdian"], "Pengabdian")
    _collect_text_rag(dfs["scholar"], "Scholar")
    if "rag_data" in dfs and dfs["rag_data"] is not None:
        _collect_text_rag(dfs["rag_data"], "Data Tambahan")

    rag_engine = SimpleLocalRAG(rag_texts, rag_sources) if rag_texts else None
    
    return evidence, candidates, rag_engine

def get_all_dosen_safely(dfs: Dict[str, pd.DataFrame]):
    all_dosen_series = []
    for name in ["homebase", "pendidikan", "mengajar"]:
        df = dfs.get(name)
        if df is not None:
            col_nama = next((c for c in df.columns if c.strip().lower() == 'nama dosen'), None)
            if col_nama:
                all_dosen_series.append(df[col_nama])
            else:
                st.warning(f"File '{name}' tidak memiliki kolom 'Nama Dosen'.")
    return pd.concat(all_dosen_series).dropna().astype(str).unique() if all_dosen_series else []

# ==========================================================
# UI: PENGATURAN LLM
# ==========================================================
st.sidebar.header("⚙️ LLM Settings")

GITHUB_MODELS: List[str] = [
    "openai/gpt-4o-mini", "microsoft/phi-4-mini-instruct", "meta/meta-llama-3.1-8b-instruct",
    "meta/llama-3.3-70b-instruct", "deepseek/deepseek-v3-0324", "google/gemma-2-9b-it",
    "mistralai/mistral-7b-instruct", "ai21-labs/ai21-jamba-1.5-mini", "deepseek/deepseek-r1-0528",
    "cohere/cohere-command-r-08-2024", "ai21-labs/ai21-jamba-1.5-large", "core42/jais-30b-chat"
]

def _github_headers():
    return {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}

PROVIDER_CONFIG = {
    "OpenRouter": {
        "api_key_name": "OPENROUTER_API_KEY",
        "init_func": lambda key: OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1"),
        "call_func": _call_openai_compatible,
        "model": "meta-llama/llama-3-8b-instruct",
        "error_map": {
            (APIError, "insufficient_quota"): ProviderQuotaError, (APIError, "rate_limit_exceeded"): ProviderQuotaError,
            (APIError, "more credits"): ProviderQuotaError, (APIError, "Insufficient credits"): ProviderQuotaError,
        },
    },
    "Groq": {
        "api_key_name": "GROQ_API_KEY",
        "init_func": lambda key: Groq(api_key=key),
        "call_func": _call_openai_compatible,
        "model": "llama-3.1-8b-instant",
        "error_map": {
            (BadRequestError, "json_validate_failed"): ProviderUnavailableError, (APIError, "insufficient_quota"): ProviderQuotaError,
            (APIError, "tokens per day"): ProviderQuotaError,
        },
    },
    "Google": {
        "api_key_name": "GOOGLE_API_KEY",
        "init_func": lambda key: genai.configure(api_key=key),
        "call_func": _call_gemini,
        "model": "models/gemini-2.5-flash",
        "error_map": {
            (google_exceptions.ResourceExhausted, "free_tier_requests"): ProviderQuotaError,
            (google_exceptions.NotFound, "was not found"): ProviderUnavailableError,
            (google_exceptions.PermissionDenied, "does not have access"): ProviderUnavailableError,
        },
    },
    "GitHub": {
        "api_key_name": "GITHUB_API_KEY",
        "init_func": lambda key: OpenAI(
            api_key=key, base_url="https://models.github.ai/inference", default_headers=_github_headers(),
        ),
        "call_func": _call_openai_compatible,
        "model": GITHUB_MODELS[0],
        "error_map": {
            (APIError, "insufficient_quota"): ProviderQuotaError, (APIError, "Unknown model"): ProviderUnavailableError,
            (APIError, "unknown_model"): ProviderUnavailableError,
        },
    },
    "Databricks": {
        "api_key_name": "DATABRICKS_API_KEY",
        "init_func": lambda key: key,
        "call_func": _call_databricks,
        "model": "https://dbc-f6300dcd-6ffc.cloud.databricks.com/serving-endpoints/databricks-meta-llama-3-1-405b-instruct/invocations",
        "error_map": {},
    },
    "Ollama": {
        "api_key_name": "OLLAMA_API_KEY",
        "init_func": lambda key: OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
        "call_func": _call_openai_compatible,
        "model": "llama3.1",
        "check_func": lambda: requests.get("http://localhost:11434", timeout=2).ok,
        "error_map": {},
    },
}

ALL_POSSIBLE_PROVIDERS = ["Databricks", "GitHub", "Groq", "OpenRouter", "Google", "Ollama"]
default_available = []
for name in ALL_POSSIBLE_PROVIDERS:
    config = PROVIDER_CONFIG[name]
    api_key = os.getenv(config["api_key_name"]) or (st.secrets.get(config["api_key_name"]) if hasattr(st, "secrets") else None)
    is_available = bool(api_key) or name == "Ollama"
    if is_available:
        try:
            if "check_func" in config and not config["check_func"]():
                is_available = False
            else:
                config["client"] = config["init_func"](api_key)
        except Exception:
            is_available = False
    config["is_available"] = is_available
    if is_available:
        default_available.append(name)

selected_providers = st.sidebar.multiselect(
    "Select and Sort Provider Priorities",
    options=ALL_POSSIBLE_PROVIDERS,
    default=default_available,
    help="Program akan mencoba dari atas ke bawah jika terjadi error.",
)
FALLBACK_ORDER = selected_providers
AVAILABLE_PROVIDERS = [p for p in FALLBACK_ORDER if PROVIDER_CONFIG[p].get("is_available")]
st.sidebar.info(f"Active Fallback Sequence: {' → '.join(AVAILABLE_PROVIDERS) or 'Tidak ada'}")

OPENROUTER_MODELS = [
    "meta-llama/llama-3.3-70b-instruct", "google/gemini-pro-2.5", "openai/gpt-4o",
    "meta-llama/llama-3.3-70b-instruct:free", "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.3-8b-instruct:free", "nvidia/nemotron-nano-9b-v2:free",
    "deepseek/deepseek-chat-v3.1:free", "openai/gpt-oss-20b:free",
    "meituan/longcat-flash-chat:free", "alibaba/tongyi-deepresearch-30b-a3b:free",
]
GROQ_MODELS = [
    "allam-2-7b", "whisper-large-v3", "openai/gpt-oss-120b", "meta-llama/llama-prompt-guard-2-86m",
    "groq/compound-mini", "playai-tts", "groq/compound", "playai-tts-arabic",
    "meta-llama/llama-4-maverick-17b-128e-instruct", "qwen/qwen3-32b", "moonshotai/kimi-k2-instruct",
    "whisper-large-v3-turbo", "meta-llama/llama-4-scout-17b-16e-instruct", "meta-llama/llama-guard-4-12b",
    "meta-llama/llama-prompt-guard-2-22m", "llama-3.1-8b-instant", "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct-0905",
]
GOOGLE_MODELS = [
    "models/gemini-2.5-pro", "models/gemini-2.5-flash", "models/gemini-pro-latest",
    "models/gemini-flash-latest", "models/gemma-3-12b-it", "models/gemma-3-27b-it",
]

st.sidebar.subheader("Model Selection")
st.session_state[PG + "openrouter_model"] = st.sidebar.selectbox("OpenRouter Models", options=OPENROUTER_MODELS, key=PG + "_or_model_widget")
st.session_state[PG + "groq_model"] = st.sidebar.selectbox("Groq Models", options=GROQ_MODELS, key=PG + "_groq_model_widget")
st.session_state[PG + "github_model"] = st.sidebar.selectbox("GitHub Models", options=GITHUB_MODELS, key=PG + "_github_model_widget")
st.session_state[PG + "google_model"] = st.sidebar.selectbox("Google Models", options=GOOGLE_MODELS, key=PG + "_google_model_widget")

st.session_state[PG + "temp"] = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1, key=PG + "_temp_widget")
max_self_reflect = st.sidebar.slider("Max. Self-Reflect cycle", 0, 2, 1, 1, key=PG + "cycles")

# ==========================================================
# LOGIKA INFERENSI
# ==========================================================
def proses_dengan_ai(system_prompt: str, user_prompt: str, fallback_response: Dict) -> Dict:
    if not AVAILABLE_PROVIDERS:
        st.error("No AI provider selected or available.")
        st.stop()

    current_index = st.session_state.get(PG + "current_provider_index", 0)
    if current_index >= len(AVAILABLE_PROVIDERS):
        current_index = 0
        st.session_state[PG + "current_provider_index"] = 0

    last_error = None
    for i in range(len(AVAILABLE_PROVIDERS)):
        provider_index_to_try = (current_index + i) % len(AVAILABLE_PROVIDERS)
        provider_name = AVAILABLE_PROVIDERS[provider_index_to_try]
        config = PROVIDER_CONFIG[provider_name]
        
        try:
            st.toast(f"Trying out providers: {provider_name}...")
            model_to_use = config["model"]
            if provider_name == "OpenRouter": model_to_use = st.session_state.get(PG + "openrouter_model", config["model"])
            elif provider_name == "Groq": model_to_use = st.session_state.get(PG + "groq_model", config["model"])
            elif provider_name == "GitHub": model_to_use = st.session_state.get(PG + "github_model", config["model"])
            elif provider_name == "Google": model_to_use = st.session_state.get(PG + "google_model", config["model"])

            system_prompt_hard = (
                f"{system_prompt}\n\n"
                "OUTPUT RULES:\n- Jawab HANYA JSON valid.\n"
                "- TANPA markdown, TANPA code fence, TANPA penjelasan di luar JSON.\n"
                '- Gunakan tepat kunci: "final_field", "alternatives", "confidence", "reasoning", "supporting_sources" atau sesuai schema.'
            )
            
            call_kwargs = {"temp": st.session_state.get(PG + "temp", 0.2), "system": system_prompt_hard, "user": user_prompt}
            
            if provider_name == "Databricks":
                call_kwargs["client_key"] = config["client"]
                call_kwargs["model_url"] = model_to_use
            else:
                call_kwargs["client"] = config["client"]
                call_kwargs["model"] = model_to_use

            result = config["call_func"](**call_kwargs)
            st.toast(f"Succeed with {provider_name}!", icon="✅")
            
            display_model_name = model_to_use.split('/')[-1]
            if provider_name == "Databricks": display_model_name = model_to_use.split('/')[-2]
                
            result["_used_provider"] = f"{provider_name} ({display_model_name})"
            st.session_state[PG + "current_provider_index"] = provider_index_to_try
            return result

        except Exception as e:
            last_error = e
            msg = str(e)
            should_fallback = False
            status_402 = getattr(e, "status_code", None) == 402 or "code': 402" in msg or 'code": 402' in msg or " 402 " in msg
            
            if provider_name == "OpenRouter" and ("Insufficient credits" in msg or status_402): should_fallback = True
            if provider_name == "GitHub" and ("Unknown model" in msg or "unknown_model" in msg): should_fallback = True
            if isinstance(e, (ProviderQuotaError, ProviderUnavailableError)): should_fallback = True
            for (error_type, error_text), _ in config["error_map"].items():
                if isinstance(e, error_type) and (error_text is None or error_text in msg):
                    should_fallback = True
                    break
            if not should_fallback and isinstance(e, (json.JSONDecodeError, ValueError)): should_fallback = True
            
            if should_fallback:
                st.warning(f"⚠️ Error pada {provider_name}: {e}. Beralih...")
                continue
            raise e

    raise ProviderUnavailableError(f"All providers failed. Last error: {last_error}")

# ==========================================================
# AGENS (Updated for RAG)
# ==========================================================
def _ekstrak_nama_bidang(data: Any) -> Optional[str]:
    if isinstance(data, str) and data.strip(): return data
    if isinstance(data, dict): return data.get("field")
    if isinstance(data, list) and len(data) > 0: return _ekstrak_nama_bidang(data[0])
    return None

def agentic_plan(nama_dosen, candidates, evidence):
    user_prompt = (
        f"Buat rencana untuk klasifikasi bidang ilmu dosen '{nama_dosen}'. "
        f"Analisis kandidat {candidates} dan tags dari bukti {evidence}. "
        "Output HANYA JSON valid: {\"steps\": list, \"focus_terms\": list of strings}"
    )
    return proses_dengan_ai("Anda adalah AI perencana.", user_prompt, {"steps": ["Analisis manual"], "focus_terms": []})

def agentic_draft(nama_dosen, evidence, candidates, plan, rag_engine):
    # Format kandidat untuk prompt
    cand_text = "\n".join([f"- {l} (sumber: {s}, skor: {sc:.2f})" for l, sc, s in candidates])
    
    # Ambil daftar nama bidang unik dari kandidat untuk validasi ketat
    valid_names = list(set([l for l, sc, s in candidates]))
    valid_names_str = ", ".join([f"'{n}'" for n in valid_names])

    rag_context = ""
    if rag_engine and plan.get("focus_terms"):
        snippets = []
        for term in plan["focus_terms"]:
            res = rag_engine.retrieve(term, top_k=3)
            snippets.append(f"Query '{term}':\n{res}")
        rag_context = "\n\n".join(snippets)
    else:
        rag_context = "(Tidak ada data teks RAG atau mesin pasif)"

    user_prompt = (
        f"Nama Dosen: {nama_dosen}\n"
        f"DAFTAR KANDIDAT VALID (PILIH DARI SINI):\n{cand_text}\n\n"
        f"DATA TOPIK / TAGS (Hanya sebagai konteks):\n"
        f"- Publikasi: {evidence.get('publikasi_tags', [])}\n"
        f"- Pengabdian: {evidence.get('pengabdian_tags', [])}\n"
        f"- Scholar: {evidence.get('scholar_tags', [])}\n\n"
        f"BUKTI RAG:\n{rag_context}\n\n"
        "TUGAS: Tentukan 'Cabang Ilmu' (Final Field) dan Alternatifnya (Field 2).\n"
        "ATURAN SANGAT PENTING (STRICT RULES):\n"
        f"1. JANGAN membuat nama bidang ilmu baru. HANYA BOLEH MEMILIH dari: [{valid_names_str}].\n"
        "2. UNTUK 'alternatives' (Field 2): HANYA ISI jika ada BUKTI KUAT dosen tersebut memiliki kepakaran ganda (Interdisipliner). "
        "   Contoh: S1 Teknik, S2 Manajemen -> Maka Field 1=Teknik, Field 2=Manajemen.\n"
        "3. JIKA dosen MONO-DISIPLIN (misal: S1, S2, Mengajar, Publikasi semuanya tentang 'Hukum'), "
        "   MAKA 'alternatives' HARUS KOSONG ([]). JANGAN MEMAKSAKAN mengisi dengan hal yang tidak relevan (seperti 'Penginderaan Jauh' untuk dosen Hukum).\n"
        "4. Field 2 TIDAK BOLEH sama dengan Field 1.\n\n"
        "Jawab HANYA JSON valid:\n"
        '{"final_field": str, "alternatives": list, "confidence": number, "reasoning": str, "supporting_sources": dict}'
    )
    return proses_dengan_ai(
        "Anda adalah validator taksonomi yang ketat. Anda dilarang berhalusinasi.", user_prompt,
        {"final_field": "Gagal", "alternatives": [], "confidence": 0.0, "reasoning": "Gagal", "supporting_sources": {}},
    )

def agentic_critique(nama_dosen, draft):
    user_prompt = (
        f"Tinjau draf untuk '{nama_dosen}': {json.dumps(draft, ensure_ascii=False)}.\n"
        "Output HANYA JSON valid: {\"issues\": list, \"suggestions\": list}"
    )
    return proses_dengan_ai("Anda AI kritikus.", user_prompt, {"issues": [], "suggestions": []})

def agentic_finalize(nama_dosen, draft, critique):
    user_prompt = (
        f"Draf awal: {json.dumps(draft, ensure_ascii=False)}\n\n"
        f"Kritik: {json.dumps(critique, ensure_ascii=False)}\n\n"
        "Perbaiki draf berdasarkan kritik.\n"
        "CHECK KHUSUS 'alternatives' (Field 2):\n"
        "- Hapus Field 2 jika itu hanya sinonim dari 'final_field'.\n"
        "- Hapus Field 2 jika tidak didukung oleh data (Hallucination check).\n"
        "- Pastikan Field 2 benar-benar ada di daftar kandidat input atau standar DIKTI.\n"
        "Jawab HANYA JSON valid menggunakan struktur draf."
    )
    return proses_dengan_ai("Anda AI finalis yang skeptis terhadap data yang tidak relevan.", user_prompt, draft)

# ==========================================================
# INPUT DATA UTAMA
# ==========================================================
st.sidebar.header("📂 Upload Analysis File")
files = {
    name: st.sidebar.file_uploader(label, type=["xlsx"], key=f"{PG}{name}")
    for name, label in [
        ("homebase", "Homebase Dosen"), ("pendidikan", "Riwayat Pendidikan"),
        ("mengajar", "Riwayat Mengajar"), ("publikasi", "Publikasi"),
        ("pengabdian", "Pengabdian"), ("scholar", "Google Scholar"),
        ("mapping", "Mapping TaksoFolk"),
        ("rag_data", "Data Tambahan RAG (Opsional)")
    ]
}

st.sidebar.markdown("---")
st.sidebar.subheader("📤 (Optional) Continue Session")
cache_file = st.sidebar.file_uploader(
    "Upload Cache (cache_hasil.json)", type=["json"], key=PG + "cache_file",
    help="Upload file .json dari sesi sebelumnya untuk melanjutkan progres."
)
st.sidebar.markdown("---")

c1, c2 = st.sidebar.columns(2)
if c1.button("🚀 Start Analysis", key=PG + "start", use_container_width=True):
    st.session_state[PG + "started"] = True
    for k in [PG + "df_hasil", PG + "excel_bytes", PG + "pdf_bytes", PG + "json_cache_bytes"]:
        st.session_state[k] = None
    st.session_state[PG + "hasil_partial"] = []
    st.session_state[PG + "stop_requested"] = False
    st.session_state[PG + "current_provider_index"] = 0
    # Reset taxonomy cache
    st.session_state[PG + "taxonomy_embeddings"] = None
    st.session_state[PG + "taxonomy_list"] = []

if c2.button("🔄 Reset", key=PG + "reset", use_container_width=True):
    for k in list(st.session_state.keys()):
        if k.startswith(PG): del st.session_state[k]
    st.rerun()

st.title("🧠 Agentic AI Fuzzy Logic Classification - TaxoFolk + RAG Approach")
if not st.session_state.get(PG + "started"):
    st.info("Upload all files, then click **Start Analysis**.")
    st.stop()

# ==========================================================
# PROSES UTAMA
# ==========================================================
if st.session_state.get(PG + "df_hasil") is None:
    if not all([files["homebase"], files["mapping"]]): # Minimum required
        st.error("❌ Please upload at least 'Homebase' and 'Mapping TaksoFolk'.")
        st.stop()

    with st.spinner("Reading files & Indexing RAG..."):
        dfs = {name: pd.read_excel(file) for name, file in files.items() if file is not None}
        st.session_state[PG + "df_mapping"] = dfs["mapping"]
        all_dosen = get_all_dosen_safely(dfs)
        
        # --- PERSIAPAN TAXONOMY VALIDATOR ---
        if st.session_state.get(PG + "taxonomy_embeddings") is None:
             prepare_taxonomy_cache(dfs["mapping"])

    if len(all_dosen) == 0:
        st.error("No lecturer names were found.")
        st.stop()

    cached_df = pd.DataFrame()
    dosen_to_process = all_dosen
    
    if cache_file is not None:
        try:
            cached_df = pd.read_json(cache_file) 
            if "Lecturer Name" in cached_df.columns:
                processed_dosen = set(cached_df["Lecturer Name"].astype(str))
                dosen_to_process = [d for d in all_dosen if d not in processed_dosen]
                st.info(f"✅ Cache JSON ditemukan. {len(processed_dosen)} selesai, {len(dosen_to_process)} baru.")
            else:
                st.warning("Cache JSON tidak valid.")
        except Exception as e:
            st.warning(f"Gagal membaca cache JSON: {e}.")

    progress_bar = st.progress(0, text="Memulai analisis...")
    
    if st.button("🛑 Stop Processing", key=PG + "stop_btn"):
        st.session_state[PG + "stop_requested"] = True
        st.warning("Permintaan berhenti... Proses akan dihentikan dan menyimpan hasil parsial setelah dosen saat ini selesai.")
    
    start_time = time.time()

    if len(dosen_to_process) == 0:
         st.success("✅ Tidak ada dosen baru untuk diproses.")
         st.session_state[PG + "df_hasil"] = cached_df
         with st.spinner("Mempersiapkan file unduhan dari cache..."):
             out_xlsx = io.BytesIO()
             cached_df.to_excel(out_xlsx, index=False, engine="openpyxl")
             st.session_state[PG + "excel_bytes"] = out_xlsx.getvalue()
             st.session_state[PG + "pdf_bytes"] = export_trees_to_pdf(cached_df, dfs["mapping"]).getvalue()
             st.session_state[PG + "json_cache_bytes"] = cached_df.to_json(orient='records', indent=4).encode('utf-8')
         st.rerun()

    try:
        total_to_process = len(dosen_to_process)
        for i, dosen in enumerate(sorted(dosen_to_process), 1):
            if st.session_state[PG + "stop_requested"]:
                st.warning("Proses dihentikan oleh pengguna. Menyimpan hasil parsial...")
                break

            # 1. Collect Evidence + RAG Engine (Updated)
            evidence, candidates, rag_engine = collect_all_evidence(dosen, dfs)
            
            # 2. Agentic Flow (Updated with RAG)
            plan = agentic_plan(dosen, candidates, evidence)
            draft = agentic_draft(dosen, evidence, candidates, plan, rag_engine)

            final_decision = draft
            for _ in range(max_self_reflect):
                critique = agentic_critique(dosen, final_decision)
                final_decision = agentic_finalize(dosen, final_decision, critique)

            alts = final_decision.get("alternatives", [])
            
            # --- EKTRAKSI DAN KOREKSI OTOMATIS BERDASARKAN TAKSONOMI ---
            raw_field_1 = _ekstrak_nama_bidang(final_decision.get("final_field"))
            raw_field_2 = _ekstrak_nama_bidang(alts[0] if alts else None)
            
            # Koreksi otomatis ke taksonomi
            bidang_ilmu_1 = validate_and_correct_taxonomy(raw_field_1)
            bidang_ilmu_2 = validate_and_correct_taxonomy(raw_field_2)

            if bidang_ilmu_1 and bidang_ilmu_1 == bidang_ilmu_2: bidang_ilmu_2 = None

            confidence_val = final_decision.get("confidence")
            safe_confidence_score = _safe_float(confidence_val)
            if safe_confidence_score > 1.0: safe_confidence_score = safe_confidence_score / 100.0

            st.session_state[PG + "hasil_partial"].append({
                "Lecturer Name": dosen,
                "Field of Science 1": bidang_ilmu_1,
                "Field of Science 2": bidang_ilmu_2,
                "Dominant Source": final_decision.get("_used_provider", "N/A"),
                "Confidence Score": round(safe_confidence_score * 100, 2),
                "Reasoning Log": final_decision.get("reasoning", ""),
                "LLM JSON": json.dumps({"plan": plan, "final": final_decision}, ensure_ascii=False),
            })

            elapsed = time.time() - start_time
            eta_seconds = (total_to_process - i) * (elapsed / i) if i > 0 else 0
            progress_bar.progress(i / total_to_process, text=f"Menganalisis {i}/{total_to_process}: {dosen} | ETA: {format_eta(int(eta_seconds))}")

    except Exception as e:
        st.error(f"🛑 Proses dihentikan karena error: {e}")
        st.warning(f"Menyimpan hasil parsial...")
    
    finally:
        progress_bar.empty()
        df_new_results = pd.DataFrame(st.session_state[PG + "hasil_partial"]) if st.session_state[PG + "hasil_partial"] else pd.DataFrame()
        
        if 'cached_df' not in locals(): cached_df = pd.DataFrame() 
        
        if not df_new_results.empty or not cached_df.empty:
            with st.spinner("Menggabungkan hasil..."):
                if not cached_df.empty:
                    df_hasil = pd.concat([cached_df, df_new_results]).drop_duplicates(subset=["Lecturer Name"], keep="last").reset_index(drop=True)
                else:
                    df_hasil = df_new_results
                
                st.session_state[PG + "df_hasil"] = df_hasil
                st.success(f"✅ Hasil gabungan ({len(df_hasil)} dosen) telah diproses.")
                
                out_xlsx = io.BytesIO()
                df_hasil.to_excel(out_xlsx, index=False, engine="openpyxl")
                st.session_state[PG + "excel_bytes"] = out_xlsx.getvalue()
                st.session_state[PG + "pdf_bytes"] = export_trees_to_pdf(df_hasil, dfs["mapping"]).getvalue()
                st.session_state[PG + "json_cache_bytes"] = df_hasil.to_json(orient='records', indent=4).encode('utf-8')
        else:
            st.session_state[PG + "df_hasil"] = None
        
        st.session_state[PG + "stop_requested"] = False
        st.session_state[PG + "hasil_partial"] = []

# ==========================================================
# OUTPUT UI
# ==========================================================
df_hasil = st.session_state.get(PG + "df_hasil")
df_mapping = st.session_state.get(PG + "df_mapping")

if df_hasil is not None:
    tab1, tab2, tab3 = st.tabs(["📊 Results Table", "🌳 Taxo-Folk Tree", "📈 Statistics"])
    
    with tab1:
        st.subheader("📊 Analysis Results")
        st.dataframe(df_hasil)
        col1, col2 = st.columns(2) 
        with col1:
            if st.session_state.get(PG + "excel_bytes"):
                st.download_button("💾 Download Laporan (Excel)", st.session_state[PG + "excel_bytes"], "hasil_klasifikasi.xlsx", key=PG+"dl_excel", use_container_width=True)
        with col2:
            if st.session_state.get(PG + "json_cache_bytes"):
                st.download_button("📥 **Unduh Cache (JSON)**", st.session_state[PG + "json_cache_bytes"], "cache_hasil.json", key=PG+"dl_json_cache", use_container_width=True)

    with tab2:
        st.subheader("🌳 Taxo-Folk Tree per Lecturer")
        if st.session_state.get(PG + "pdf_bytes"):
            st.download_button("📥 Download All Trees (PDF)", st.session_state[PG + "pdf_bytes"], "pohon_taksofolk_parsial.pdf", key=PG+"dl_pdf")
        for _, row in df_hasil.iterrows():
            with st.expander(f"👨‍🏫 {row['Lecturer Name']}"):
                dot = build_taksofolk_tree(row["Lecturer Name"], row.get("Field of Science 1"), row.get("Field of Science 2"), df_mapping)
                st.graphviz_chart(dot)

    with tab3:
        st.subheader("📈 Confidence Score Statistics")
        if "Confidence Score" in df_hasil.columns:
            sc = pd.to_numeric(df_hasil["Confidence Score"], errors='coerce').dropna()
            if not sc.empty:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rata-rata", f"{sc.mean():.2f}%")
                c2.metric("Median", f"{sc.median():.2f}%")
                c3.metric("Min", f"{sc.min():.2f}%")
                c4.metric("Max", f"{sc.max():.2f}%")
                st.dataframe(sc.describe())
                try:
                    bins = pd.cut(sc, bins=range(0, 101, 10), right=True)
                    st.bar_chart(bins.value_counts().sort_index())
                except: pass
