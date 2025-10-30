# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import graphviz
import io, time, json, os, re
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

# ==========================================================
# INIT
# ==========================================================
load_dotenv()
st.set_page_config(page_title="Agentic AI Classification - TaksoFolk", page_icon="🧠", layout="wide")
PG = "pg9aa_"  # Prefix state
# --- MODIFIKASI 1: CACHE_FILE dihapus ---

for k, v in [
    (PG + "started", False),
    (PG + "df_hasil", None),
    (PG + "df_mapping", None),
    (PG + "excel_bytes", None),
    (PG + "pdf_bytes", None),
    (PG + "json_cache_bytes", None), # <<< MODIFIKASI 1: Ditambahkan
]:
    st.session_state.setdefault(k, v)

# ==========================================================
# JSON SANITIZER (mencegah: Unterminated string / output non-JSON)
# ==========================================================
def _extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    # 1) code-fence ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    # 2) blok dari { pertama ke } terakhir
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None

def _loads_json_strict(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = _extract_json_block(text)
        if not cleaned:
            raise
        # hapus koma buntut
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        # normalisasi kutip “ ” ‘ ’ → standar
        cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        return json.loads(cleaned)

# ==========================================================
# PEMBUNGKUS PANGGILAN MODEL
# ==========================================================
MAX_OUT_TOKENS = 4096  # agar output tidak kepotong & JSON utuh

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

# --- MODIFIKASI 1: Mengubah _call_gemini agar dinamis membuat model ---
def _call_gemini(client, model, temp, system, user):
    # 'client' dari init_func adalah None (setelah modifikasi di PROVIDER_CONFIG)
    # Kita menginisialisasi model di sini menggunakan string 'model' yang dipilih
    try:
        model_client = genai.GenerativeModel(model)
    except Exception as e:
        # Menangkap error jika nama model salah atau API key belum dikonfigurasi
        raise ProviderUnavailableError(f"Gagal memuat model Google: {model}. Error: {e}")

    prompt = (
        f"{system}\n\n{user}\n\n"
        "Jawab HANYA JSON valid. TANPA markdown, TANPA code fence, TANPA teks lain di luar JSON."
    )
    resp = model_client.generate_content( # Menggunakan model_client yang baru dibuat
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temp,
            response_mime_type="application/json",
            max_output_tokens=MAX_OUT_TOKENS,
        ),
    )
    content = resp.text or ""
    return _loads_json_strict(content)
# --- AKHIR MODIFIKASI 1 ---

class ProviderQuotaError(Exception): ...
class ProviderUnavailableError(Exception): ...

# ==========================================================
# UTIL & VISUAL (DIDEFINISIKAN SEBELUM DIPAKAI)
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
    if not bidang or str(bidang) == "nan":
        return ["Lainnya"] * 4
    try:
        mask = df_mapping["Level 3"].astype(str) == str(bidang)
        if mask.any():
            row = df_mapping[mask].iloc[0]
            return [str(row["Bidang"]), str(row["Level 1"]), str(row["Level 2"]), str(row["Level 3"])]
        return [f"{bidang} - Bidang", f"{bidang} - L1", f"{bidang} - L2", str(bidang)]
    except Exception:
        return [f"{bidang} - Bidang", f"{bidang} - L1", f"{bidang} - L2", str(bidang)]

def build_taksofolk_tree(nama_dosen: str, bidang1: Optional[str], bidang2: Optional[str], df_mapping: pd.DataFrame) -> graphviz.Digraph:
    dot = graphviz.Digraph()
    dot.node("root", "Bidang Ilmu", style="filled", color="lightgrey")

    def _add_branch(prefix: str, bidang: str):
        path = taksofolk_mapping(bidang, df_mapping)
        unique_path = []
        for level in path:
            if level and (not unique_path or unique_path[-1] != level):
                unique_path.append(level)
        if not unique_path or unique_path[-1] != bidang:
            unique_path.append(bidang)
        parent_id = "root"
        colors = ["lightpink", "lightblue", "lightgreen", "lightyellow"]
        for i, level_name in enumerate(unique_path):
            node_id = f"{prefix}_{i}"
            is_last_node = i == len(unique_path) - 1
            shape = "ellipse" if is_last_node else "box"
            color = "orange" if is_last_node else colors[i % len(colors)]
            dot.node(node_id, str(level_name), shape=shape, style="filled", color=color)
            dot.edge(parent_id, node_id)
            parent_id = node_id

    bidang_unik = []
    if bidang1:
        bidang_unik.append(bidang1)
    if bidang2 and bidang2 != bidang1:
        bidang_unik.append(bidang2)
    for i, bidang in enumerate(bidang_unik):
        _add_branch(chr(97 + i), bidang)
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

def collect_all_evidence(dosen: str, dfs: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, Any], List]:
    evidence = {}

    def _pick_first_match(df, col):
        if "Nama Dosen" not in df.columns:
            return None, 0.0
        sub = df[df["Nama Dosen"].astype(str) == dosen]
        if not sub.empty:
            return sub.iloc[0].get(f"Best Match {col}"), float(sub.iloc[0].get(f"Score {col}", 0.0))
        return None, 0.0

    evidence["homebase"], hb_score = _pick_first_match(dfs["homebase"], "Cabang Ilmu")
    evidence["pendidikan"], pd_score = _pick_first_match(dfs["pendidikan"], "Cabang Ilmu")
    evidence["mengajar"], mg_score = _pick_first_match(dfs["mengajar"], "Cabang Ilmu")

    def _get_tags(df_tag):
        if "Nama Dosen" in df_tag.columns and "Tag" in df_tag.columns:
            sub = df_tag[df_tag["Nama Dosen"].astype(str) == dosen]
            if not sub.empty:
                return sorted(list(set(sub["Tag"].dropna().astype(str).tolist())))
        return []

    evidence["publikasi_tags"] = _get_tags(dfs["publikasi"])
    evidence["pengabdian_tags"] = _get_tags(dfs["pengabdian"])
    evidence["scholar_tags"] = _get_tags(dfs["scholar"])

    candidates = sorted(
        [(evidence[k], s, k) for k, s in [("homebase", hb_score), ("pendidikan", pd_score), ("mengajar", mg_score)] if evidence[k]],
        key=lambda x: x[1],
        reverse=True,
    )
    return evidence, candidates

def get_all_dosen_safely(dfs: Dict[str, pd.DataFrame]):
    all_dosen_series = []
    for name in ["homebase", "pendidikan", "mengajar"]:
        df = dfs.get(name)
        if df is not None and "Nama Dosen" in df.columns:
            all_dosen_series.append(df["Nama Dosen"])
        else:
            st.warning(f"File '{name}' tidak memiliki kolom 'Nama Dosen'.")
    return pd.concat(all_dosen_series).dropna().astype(str).unique() if all_dosen_series else []

# ==========================================================
# UI: PENGATURAN LLM
# ==========================================================
st.sidebar.header("⚙️ LLM Settings")

# --- Daftar model GitHub (STATIS, sesuai ketentuan) ---
GITHUB_MODELS: List[str] = [
    "openai/gpt-4o-mini",                 # plug-and-play
    "microsoft/phi-4-mini-instruct",
    "meta/meta-llama-3.1-8b-instruct",
    "meta/llama-3.3-70b-instruct",
    "deepseek/deepseek-v3-0324",
    "google/gemma-2-9b-it",
    "mistralai/mistral-7b-instruct",
    "ai21-labs/ai21-jamba-1.5-mini",
    "deepseek/deepseek-r1-0528",
    "cohere/cohere-command-r-08-2024",
    "ai21-labs/ai21-jamba-1.5-large",
    "core42/jais-30b-chat"
]

# ==========================================================
# DEFINISI PROVIDER (tanpa DeepSeek provider)
# ==========================================================
def _github_headers():
    return {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

PROVIDER_CONFIG = {
    "OpenRouter": {
        "api_key_name": "OPENROUTER_API_KEY",
        "init_func": lambda key: OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1"),
        "call_func": _call_openai_compatible,
        "model": "meta-llama/llama-3-8b-instruct",
        "error_map": {
            (APIError, "insufficient_quota"): ProviderQuotaError,
            (APIError, "rate_limit_exceeded"): ProviderQuotaError,
            (APIError, "more credits"): ProviderQuotaError,
            (APIError, "Insufficient credits"): ProviderQuotaError,
        },
    },
    "Groq": {
        "api_key_name": "GROQ_API_KEY",
        "init_func": lambda key: Groq(api_key=key),
        "call_func": _call_openai_compatible,
        "model": "llama-3.1-8b-instant",
        "error_map": {
            (BadRequestError, "json_validate_failed"): ProviderUnavailableError,
            (APIError, "insufficient_quota"): ProviderQuotaError,
            (APIError, "tokens per day"): ProviderQuotaError,
        },
    },
    # --- MODIFIKASI 2: Mengubah init_func Google ---
    "Google": {
        "api_key_name": "GOOGLE_API_KEY",
        # init_func HANYA mengkonfigurasi API key. Client akan 'None'.
        # Model akan dibuat di dalam _call_gemini
        "init_func": lambda key: genai.configure(api_key=key),
        "call_func": _call_gemini,
        "model": "models/gemini-2.5-flash", # Default model jika tidak ada pilihan
        "error_map": {
            (google_exceptions.ResourceExhausted, "free_tier_requests"): ProviderQuotaError,
            (google_exceptions.NotFound, "was not found"): ProviderUnavailableError,
            (google_exceptions.PermissionDenied, "does not have access"): ProviderUnavailableError,
        },
    },
    # --- AKHIR MODIFIKASI 2 ---
    "GitHub": {
        "api_key_name": "GITHUB_API_KEY",
        "init_func": lambda key: OpenAI(
            api_key=key,
            base_url="https://models.github.ai/inference",
            default_headers=_github_headers(),
        ),
        "call_func": _call_openai_compatible,
        "model": GITHUB_MODELS[0],  # default "openai/gpt-4o-mini"
        "error_map": {
            (APIError, "insufficient_quota"): ProviderQuotaError,
            (APIError, "Unknown model"): ProviderUnavailableError,
            (APIError, "unknown_model"): ProviderUnavailableError,
        },
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

# Inisialisasi Klien & cek ketersediaan — urutan prioritas default:
ALL_POSSIBLE_PROVIDERS = ["GitHub", "Groq", "OpenRouter", "Google", "Ollama"]  # GitHub → Groq → OpenRouter → Google → Ollama
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

# Pilihan model per provider
OPENROUTER_MODELS = [
    # Model lama Anda jika masih ingin dipakai
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-pro-2.5",
    "openai/gpt-4o",
    # === Tambahan model free dari file Excel ===
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.3-8b-instruct:free",
    "nvidia/nemotron-nano-9b-v2:free",          # ✅ direkomendasikan utama
    "deepseek/deepseek-chat-v3.1:free",         # ✅ layak (fallback JSON)
    "openai/gpt-oss-20b:free",                  # ✅ layak (uji JSON)
    "meituan/longcat-flash-chat:free",          # ⚠ risiko JSON
    "alibaba/tongyi-deepresearch-30b-a3b:free", # ⚠ cenderung naratif, tidak ketat JSON
]
GROQ_MODELS = [
    "allam-2-7b",
    "whisper-large-v3",
    "openai/gpt-oss-120b",
    "meta-llama/llama-prompt-guard-2-86m",
    "groq/compound-mini",
    "playai-tts",
    "groq/compound",
    "playai-tts-arabic",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct",
    "whisper-large-v3-turbo",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-guard-4-12b",
    "meta-llama/llama-prompt-guard-2-22m",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "moonshotai/kimi-k2-instruct-0905",
]

# --- MODIFIKASI 3: Menambahkan daftar model Google ---
GOOGLE_MODELS = [
    "models/gemini-2.5-pro",
    "models/gemini-2.5-flash",
    "models/gemini-pro-latest",
    "models/gemini-flash-latest",
    "models/gemma-3-12b-it",
    "models/gemma-3-27b-it",
]
# --- AKHIR MODIFIKASI 3 ---

st.sidebar.subheader("Model Selection")
st.session_state[PG + "openrouter_model"] = st.sidebar.selectbox(
    "OpenRouter Models", options=OPENROUTER_MODELS, key=PG + "_or_model_widget"
)
st.session_state[PG + "groq_model"] = st.sidebar.selectbox(
    "Groq Models", options=GROQ_MODELS, key=PG + "_groq_model_widget"
)
st.session_state[PG + "github_model"] = st.sidebar.selectbox(
    "GitHub Models", options=GITHUB_MODELS, key=PG + "_github_model_widget"
)

# --- MODIFIKASI 4: Menambahkan selectbox Google ---
st.session_state[PG + "google_model"] = st.sidebar.selectbox(
    "Google Models", options=GOOGLE_MODELS, key=PG + "_google_model_widget"
)
# --- AKHIR MODIFIKASI 4 ---


st.sidebar.caption("Make sure API keys are set in .env or Streamlit Secrets.")
st.session_state[PG + "temp"] = st.sidebar.slider(
    "Temperature", 0.0, 1.0, 0.2, 0.1, key=PG + "_temp_widget",
    help=("""Set the model's creativity level:
- 0.0–0.2: Consistent and deterministic answers.
- 0.3–0.6: Balance consistency & variety.
- 0.7–1.0: Creative, but less stable.""")
)
max_self_reflect = st.sidebar.slider(
    "Max. Self-Reflect cycle", 0, 2, 1, 1, key=PG + "cycles",
    help="Higher values ​​have the potential to improve quality, but slow down the process.."
)

# ==========================================================
# LOGIKA INFERENSI + FALLBACK
# ==========================================================
def proses_dengan_ai(system_prompt: str, user_prompt: str, fallback_response: Dict) -> Dict:
    if not AVAILABLE_PROVIDERS:
        st.error("No AI provider selected or available.")
        st.stop()

    last_error = None
    for provider_name in AVAILABLE_PROVIDERS:
        config = PROVIDER_CONFIG[provider_name]
        try:
            st.toast(f"Trying out providers: {provider_name}...")
            
            # --- MODIFIKASI 5: Menambahkan logika untuk mengambil model Google ---
            model_to_use = config["model"]
            if provider_name == "OpenRouter":
                model_to_use = st.session_state.get(PG + "openrouter_model", config["model"])
            elif provider_name == "Groq":
                model_to_use = st.session_state.get(PG + "groq_model", config["model"])
            elif provider_name == "GitHub":
                model_to_use = st.session_state.get(PG + "github_model", config["model"])
            elif provider_name == "Google":
                model_to_use = st.session_state.get(PG + "google_model", config["model"])
            # --- AKHIR MODIFIKASI 5 ---

            # Perketat sistem prompt agar anti-teks tambahan
            system_prompt_hard = (
                f"{system_prompt}\n\n"
                "OUTPUT RULES:\n"
                "- Jawab HANYA JSON valid.\n"
                "- TANPA markdown, TANPA code fence, TANPA penjelasan di luar JSON.\n"
                '- Gunakan tepat kunci: "final_field", "alternatives", "confidence", "reasoning", "supporting_sources" '
                'atau sesuai schema yang diminta di prompt terkait.'
            )

            result = config["call_func"](
                client=config["client"],
                model=model_to_use,
                temp=st.session_state.get(PG + "temp", 0.2),
                system=system_prompt_hard,
                user=user_prompt,
            )
            st.toast(f"Berhasil dengan {provider_name}!", icon="✅")
            result["_used_provider"] = f"{provider_name} ({model_to_use.split('/')[-1]})"
            return result

        except Exception as e:
            last_error = e
            msg = str(e)
            should_fallback = False

            # === Deteksi khusus OpenRouter 402 (Insufficient credits) ===
            status_402 = getattr(e, "status_code", None) == 402 or "code': 402" in msg or 'code": 402' in msg or " 402 " in msg
            if provider_name == "OpenRouter" and ("Insufficient credits" in msg or status_402):
                st.warning(
                    "⚠️ OpenRouter: Insufficient credits (402). Switch to the next provider. "
                    "Top up your balance at https://openrouter.ai/settings/credits if you want to use OpenRouter."
                )
                should_fallback = True

            # === Deteksi khusus GitHub unknown model ===
            if provider_name == "GitHub" and ("Unknown model" in msg or "unknown_model" in msg):
                st.warning("⚠️ GitHub Models: Unknown model. Switch to the next provider / choose another model.")
                should_fallback = True

            # Mapping error -> fallback (kuota, rate limit, dsb.)
            if not should_fallback:
                for (error_type, error_text), mapped_exception in config["error_map"].items():
                    if isinstance(e, error_type) and (error_text is None or error_text in msg):
                        st.warning(f"Error pada {provider_name} ({mapped_exception.__name__}). Beralih...")
                        should_fallback = True
                        break

            # Respons bukan JSON
            if not should_fallback and isinstance(e, json.JSONDecodeError):
                st.warning(f"Provider {provider_name} returns a non-JSON response. Switch...")
                should_fallback = True

            if not should_fallback:
                # error lain yang tidak dipetakan -> hentikan
                raise e

    # Jika semua gagal
    raise ProviderUnavailableError(f"All providers failed. Last error: {last_error}")

# ==========================================================
# AGENS (PLAN → DRAFT → CRITIQUE → FINALIZE)
# ==========================================================
def _ekstrak_nama_bidang(data: Any) -> Optional[str]:
    if isinstance(data, dict):
        return data.get("field")
    if isinstance(data, str) and data.strip():
        return data
    return None

def agentic_plan(nama_dosen, candidates, evidence):
    user_prompt = (
        f"Buat rencana untuk klasifikasi bidang ilmu dosen '{nama_dosen}'. "
        f"Analisis kandidat {candidates} dan tags dari bukti {evidence}. "
        "Output HANYA JSON valid: {\"steps\": list, \"focus_terms\": list}"
    )
    system_prompt = "Anda adalah AI perencana."
    return proses_dengan_ai(system_prompt, user_prompt, {"steps": ["Analisis manual"], "focus_terms": []})

def agentic_draft(nama_dosen, evidence, candidates, plan):
    cand_text = "\n".join([f"- {l} (sumber: {s}, skor: {sc:.2f})" for l, sc, s in candidates])
    user_prompt = (
        f"Nama Dosen: {nama_dosen}\nKANDIDAT:\n{cand_text or '(Tidak ada)'}\nTAGS:\n"
        f"- Publikasi: {evidence.get('publikasi_tags', [])}\n"
        f"- Pengabdian: {evidence.get('pengabdian_tags', [])}\n"
        f"- Scholar: {evidence.get('scholar_tags', [])}\n\n"
        "Buat DRAF keputusan klasifikasi. Jawab HANYA JSON valid dengan kunci persis:\n"
        '{"final_field": str, "alternatives": list, "confidence": number, "reasoning": str, "supporting_sources": dict}'
    )
    system_prompt = "Anda adalah pakar klasifikasi taksonomi keilmuan."
    return proses_dengan_ai(
        system_prompt,
        user_prompt,
        {"final_field": "Gagal", "alternatives": [], "confidence": 0.0, "reasoning": "Gagal", "supporting_sources": {}},
    )

def agentic_critique(nama_dosen, draft):
    user_prompt = (
        f"Tinjau draf untuk '{nama_dosen}': {json.dumps(draft, ensure_ascii=False)}.\n"
        "Output HANYA JSON valid: {\"issues\": list, \"suggestions\": list}"
    )
    system_prompt = "Anda AI kritikus."
    return proses_dengan_ai(system_prompt, user_prompt, {"issues": [], "suggestions": []})

def agentic_finalize(nama_dosen, draft, critique):
    user_prompt = (
        f"Draf awal: {json.dumps(draft, ensure_ascii=False)}\n\n"
        f"Kritik: {json.dumps(critique, ensure_ascii=False)}\n\n"
        "Perbaiki draf berdasarkan kritik. Jawab HANYA JSON valid menggunakan struktur draf."
    )
    system_prompt = "Anda AI finalis."
    return proses_dengan_ai(system_prompt, user_prompt, draft)

# ==========================================================
# INPUT DATA UTAMA
# ==========================================================
st.sidebar.header("📂 Upload Analysis File")
files = {
    name: st.sidebar.file_uploader(label, type=["xlsx"], key=f"{PG}{name}")
    for name, label in [
        ("homebase", "Homebase Dosen"),
        ("pendidikan", "Riwayat Pendidikan"),
        ("mengajar", "Riwayat Mengajar"),
        ("publikasi", "Publikasi"),
        ("pengabdian", "Pengabdian"),
        ("scholar", "Google Scholar"),
        ("mapping", "Mapping TaksoFolk"),
    ]
}

# --- START: MODIFIKASI 2: PENAMBAHAN UPLOAD CACHE JSON ---
st.sidebar.markdown("---")
st.sidebar.subheader("📤 (Optional) Continue Session")
cache_file = st.sidebar.file_uploader(
    "Upload Cache (cache_hasil.json)", 
    type=["json"], 
    key=PG + "cache_file",
    help="Upload the .json file you downloaded from the previous session to continue progress.."
)
st.sidebar.markdown("---")
# --- END: MODIFIKASI 2 ---

c1, c2 = st.sidebar.columns(2)
if c1.button("🚀 Start Analysis", key=PG + "start", use_container_width=True):
    st.session_state[PG + "started"] = True
    for k in [PG + "df_hasil", PG + "excel_bytes", PG + "pdf_bytes", PG + "json_cache_bytes"]:
        st.session_state[k] = None
if c2.button("🔄 Reset", key=PG + "reset", use_container_width=True):
    for k in list(st.session_state.keys()):
        if k.startswith(PG):
            del st.session_state[k]
    st.rerun()

st.title("🧠 Agentic AI Fuzzy Logic Classification - TaxoFolk Approach")
if not st.session_state.get(PG + "started"):
    st.info("Upload all files, then click **Start Analysis**.")
    st.stop()

# ==========================================================
# PROSES UTAMA
# ==========================================================
if st.session_state.get(PG + "df_hasil") is None:
    if not all(files.values()):
        st.error("❌ Please upload all 7 files.")
        st.stop()

    with st.spinner("Reading files..."):
        dfs = {name: pd.read_excel(file) for name, file in files.items()}
        st.session_state[PG + "df_mapping"] = dfs["mapping"]
        all_dosen = get_all_dosen_safely(dfs)

    if not all_dosen.any():
        st.error("No lecturer names were found.")
        st.stop()

    # --- START: MODIFIKASI 3: LOGIKA PEMBACAAN CACHE UPLOAD ---
    cached_df = pd.DataFrame()
    dosen_to_process = all_dosen
    
    # Cek jika file cache di-upload (cache_file diambil dari state)
    if cache_file is not None:
        try:
            # Baca JSON langsung ke DataFrame
            cached_df = pd.read_json(cache_file) 
            if "Lecturer Name" in cached_df.columns:
                processed_dosen = set(cached_df["Lecturer Name"].astype(str))
                dosen_to_process = [d for d in all_dosen if d not in processed_dosen]
                st.info(f"✅ Cache JSON ditemukan. {len(processed_dosen)} dosen akan dilewati. {len(dosen_to_process)} dosen baru akan diproses.")
            else:
                st.warning("Cache JSON tidak valid (tidak ada 'Lecturer Name'). Memproses ulang.")
                dosen_to_process = all_dosen
        except Exception as e:
            st.warning(f"Gagal membaca cache JSON: {e}. Memproses ulang.")
            dosen_to_process = all_dosen
    else:
        st.info(f"Cache tidak di-upload. Memproses {len(dosen_to_process)} dosen dari awal.")

    hasil = [] # 'hasil' HANYA menampung data baru
    progress_bar = st.progress(0, text="Memulai analisis...")
    start_time = time.time()

    # Penanganan jika tidak ada yang perlu diproses (sudah di-cache semua)
    # (Perbaikan dari error sebelumnya: `if not dosen_to_process:` diubah ke `if len(...) == 0:`)
    if len(dosen_to_process) == 0:
         st.success("✅ Tidak ada dosen baru untuk diproses. Semua data diambil dari cache.")
         st.session_state[PG + "df_hasil"] = cached_df
         with st.spinner("Mempersiapkan file unduhan dari cache..."):
            out_xlsx = io.BytesIO()
            cached_df.to_excel(out_xlsx, index=False, engine="openpyxl")
            st.session_state[PG + "excel_bytes"] = out_xlsx.getvalue()
            st.session_state[PG + "pdf_bytes"] = export_trees_to_pdf(cached_df, dfs["mapping"]).getvalue()
            
            # Siapkan juga cache JSON untuk diunduh ulang
            json_string = cached_df.to_json(orient='records', indent=4)
            st.session_state[PG + "json_cache_bytes"] = json_string.encode('utf-8')
         st.rerun()
    # --- END: MODIFIKASI 3 ---

   try:
        total_to_process = len(dosen_to_process)
        for i, dosen in enumerate(sorted(dosen_to_process), 1):
            evidence, candidates = collect_all_evidence(dosen, dfs)
            plan = agentic_plan(dosen, candidates, evidence)
            draft = agentic_draft(dosen, evidence, candidates, plan)

            final_decision = draft
            for _ in range(max_self_reflect):
                critique = agentic_critique(dosen, final_decision)
                final_decision = agentic_finalize(dosen, final_decision, critique)

            alts = final_decision.get("alternatives", [])
            bidang_ilmu_1 = _ekstrak_nama_bidang(final_decision.get("final_field"))
            bidang_ilmu_2 = _ekstrak_nama_bidang(alts[0] if alts else None)
            if bidang_ilmu_1 and bidang_ilmu_1 == bidang_ilmu_2:
                bidang_ilmu_2 = None

            confidence_val = final_decision.get("confidence")
            safe_confidence_score = _safe_float(confidence_val)

            # --- START PERBAIKAN ---
            # Normalisasi skor jika LLM mengembalikan 80 (int) alih-alih 0.8 (float)
            if safe_confidence_score > 1.0:
                safe_confidence_score = safe_confidence_score / 100.0
            # --- END PERBAIKAN ---

            hasil.append(
                {
                    "Lecturer Name": dosen,
                    "Field of Science 1": bidang_ilmu_1,
                    "Field of Science 2": bidang_ilmu_2,
                    "Dominant Source": final_decision.get("_used_provider", "N/A"),
                    # Sekarang 'safe_confidence_score' PASTI antara 0.0 - 1.0
                    "Confidence Score": round(safe_confidence_score * 100, 2), 
                    "Reasoning Log": final_decision.get("reasoning", ""),
                    "LLM JSON": json.dumps({"plan": plan, "final": final_decision}, ensure_ascii=False),
                }
            )

            elapsed = time.time() - start_time
            eta_formatted = "Menghitung..."
            if i > 5:
                eta_seconds = (total_to_process - i) * (elapsed / i) if i > 0 else 0
                eta_formatted = format_eta(int(eta_seconds))

            progress_bar.progress(i / total_to_process, text=f"Menganalisis {i}/{total_to_process}: {dosen} | ETA: {eta_formatted}")

    except Exception as e:
        st.error(f"🛑 Proses dihentikan karena error: {e}")
        st.warning(f"Menyimpan hasil parsial untuk {len(hasil)} dosen yang baru diproses.")
    finally:
        # --- START: MODIFIKASI 4: BLOK FINALLY UNTUK CACHE JSON ---
        progress_bar.empty()
        
        df_new_results = pd.DataFrame(hasil) if hasil else pd.DataFrame()
        
        # 'cached_df' sudah didefinisikan di atas (dari file upload)
        if 'cached_df' not in locals():
            cached_df = pd.DataFrame() 
            
        # Periksa apakah ada hasil (baru atau lama) untuk diproses
        if not df_new_results.empty or not cached_df.empty:
            with st.spinner("Menggabungkan hasil baru dengan cache dan membuat file..."):
                
                # Gabungkan data lama (cache) dengan data baru (hasil)
                if not cached_df.empty:
                    df_hasil = pd.concat([cached_df, df_new_results]).drop_duplicates(
                        subset=["Lecturer Name"], keep="last"
                    ).reset_index(drop=True)
                else:
                    df_hasil = df_new_results
                
                # Set state session dengan hasil gabungan
                st.session_state[PG + "df_hasil"] = df_hasil
                
                st.success(f"✅ Hasil gabungan ({len(df_hasil)} dosen) telah diproses.")
                
                # Buat file download Excel (dari hasil gabungan)
                out_xlsx = io.BytesIO()
                df_hasil.to_excel(out_xlsx, index=False, engine="openpyxl")
                st.session_state[PG + "excel_bytes"] = out_xlsx.getvalue()
                
                # Buat file download PDF
                st.session_state[PG + "pdf_bytes"] = export_trees_to_pdf(df_hasil, dfs["mapping"]).getvalue()

                # Buat file CACHE JSON untuk diunduh
                json_string = df_hasil.to_json(orient='records', indent=4)
                st.session_state[PG + "json_cache_bytes"] = json_string.encode('utf-8')

        else:
            st.session_state[PG + "df_hasil"] = None
        # --- END: MODIFIKASI 4 ---

# ==========================================================
# OUTPUT UI
# ==========================================================
df_hasil = st.session_state.get(PG + "df_hasil")
df_mapping = st.session_state.get(PG + "df_mapping")

if df_hasil is not None:
    
    # --- MODIFIKASI: Menambahkan tab 'Statistics' ---
    tab_titles = ["📊 Results Table", "🌳 Taxo-Folk Tree", "📈 Statistics"]
    tab1, tab2, tab3 = st.tabs(tab_titles)
    
    with tab1:
        st.subheader("📊 Analysis Results")
        st.dataframe(df_hasil)
        
        # --- START: MODIFIKASI 5: TOMBOL DOWNLOAD CACHE JSON ---
        col1, col2 = st.columns(2) 
        with col1:
            if st.session_state.get(PG + "excel_bytes"):
                st.download_button(
                    "💾 Download Laporan (Excel)",
                    st.session_state[PG + "excel_bytes"],
                    "hasil_klasifikasi.xlsx",
                    key=PG + "dl_excel",
                    use_container_width=True
                )
        
        with col2:
            if st.session_state.get(PG + "json_cache_bytes"):
                st.download_button(
                    "📥 **Unduh Cache (JSON)**",
                    st.session_state[PG + "json_cache_bytes"],
                    "cache_hasil.json",
                    key=PG + "dl_json_cache",
                    help="Simpan file ini! Upload file ini di sesi berikutnya untuk melanjutkan progres.",
                    use_container_width=True
                )
        # --- END: MODIFIKASI 5 ---

    with tab2:
        st.subheader("🌳 Taxo-Folk Tree per Lecturer")
        if st.session_state.get(PG + "pdf_bytes"):
            st.download_button(
                "📥 Download All Trees (PDF)",
                st.session_state[PG + "pdf_bytes"],
                "pohon_taksofolk_parsial.pdf",
                key=PG + "dl_pdf",
            )
        for _, row in df_hasil.iterrows():
            with st.expander(f"👨‍🏫 {row['Lecturer Name']}"):
                dot = build_taksofolk_tree(row["Lecturer Name"], row.get("Field of Science 1"), row.get("Field of Science 2"), df_mapping)
                st.graphviz_chart(dot)

    # --- START: BLOK KODE BARU UNTUK TAB 3 ---
    with tab3:
        st.subheader("📈 Confidence Score Statistics")
        
        if "Confidence Score" not in df_hasil.columns:
            st.warning("Kolom 'Confidence Score' tidak ditemukan.")
        else:
            # Konversi ke numerik, paksa error ke NaN, lalu hapus NaN
            confidence_scores = pd.to_numeric(df_hasil["Confidence Score"], errors='coerce').dropna()
            
            if confidence_scores.empty:
                st.info("Tidak ada data 'Confidence Score' yang valid untuk dianalisis.")
            else:
                st.markdown("Ringkasan Statistik Deskriptif untuk **Confidence Score (%)**")
                
                # Tampilkan metrik utama
                cols_metrik = st.columns(4)
                cols_metrik[0].metric("Rata-rata", f"{confidence_scores.mean():.2f} %")
                cols_metrik[1].metric("Median", f"{confidence_scores.median():.2f} %")
                cols_metrik[2].metric("Minimum", f"{confidence_scores.min():.2f} %")
                cols_metrik[3].metric("Maksimum", f"{confidence_scores.max():.2f} %")
                
                # Tampilkan tabel describe()
                st.dataframe(confidence_scores.describe())

                st.markdown("---")
                st.subheader("Distribusi Confidence Score")
                
                try:
                    # Buat bin untuk histogram (misal: 0-10, 10-20, ..., 90-100)
                    # Kita gunakan pd.cut
                    bins = pd.cut(confidence_scores, bins=range(0, 101, 10), right=True)
                    
                    # Hitung jumlah di setiap bin
                    hist_data = bins.value_counts().sort_index()
                    
                    # Ubah nama index agar lebih mudah dibaca di chart
                    hist_data.index = hist_data.index.astype(str)
                    
                    # Ubah nama Series agar ada label di chart
                    hist_data.name = "Jumlah Dosen"
                    
                    st.bar_chart(hist_data)
                    st.caption("Histogram yang menunjukkan jumlah dosen per rentang skor kepercayaan (interval 10%).")

                except Exception as e:
                    st.error(f"Gagal membuat histogram: {e}")
    # --- END: BLOK KODE BARU UNTUK TAB 3 ---
