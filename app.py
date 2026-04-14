import json
import hmac
import requests
import streamlit as st
from pathlib import Path
import base64
import time
from db_optimized import build_context, start_DB, warmup, rebuild_DB
import threading
import os

current_dir = Path(__file__).resolve()
image_path = current_dir.parent / "images"
folder_path = current_dir.parent / "docs"
CONFIG_FILE = current_dir.parent / "config.json"
ACCESS_COUNT_FILE = current_dir.parent / "access_count.txt"
MAX_OLLAMA_CONCURRENCY = int(os.getenv("MAX_OLLAMA_CONCURRENCY", "2"))
_OLLAMA_SEM = threading.Semaphore(MAX_OLLAMA_CONCURRENCY)

# Config loading/saving
def load_config() -> dict:
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"system_prompt": "", "initial_message": ""}

def save_config(data: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_prompt() -> str:
    return load_config().get("system_prompt", "")

def load_initial_message() -> str:
    return load_config().get("initial_message", "")

def get_current_prompt():
    if st.session_state.get("_prompt_reload_flag"):
        st.session_state._prompt_reload_flag = False
        st.session_state.current_prompt = load_prompt()
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = load_prompt()
    return st.session_state.current_prompt

def get_current_initial_message():
    if st.session_state.get("im_reload_flag"):
        st.session_state.im_reload_flag = False
        st.session_state.current_initial_message = load_initial_message()
    if "current_initial_message" not in st.session_state:
        st.session_state.current_initial_message = load_initial_message()
    return st.session_state.current_initial_message

SETTINGS_PASSWORD = st.secrets["admin_password"] ["value"]

# General page-config --> Always needs to be the first st.-command!
st.set_page_config(page_title="Dialogos BNE", page_icon="🦙", layout="centered", initial_sidebar_state="collapsed")

with st.sidebar.expander("⚙️ Einstellungen (Admin)", expanded=False):
    pw = st.text_input("Passwort (Einstellungen werden erst nach korrekter Eingabe sichtbar)", type="password", key="prompt_pw")
    if pw and hmac.compare_digest(pw.encode("utf-8"), str(SETTINGS_PASSWORD).encode("utf-8")):
        st.write("*Aufrufe:*", st.session_state.get("access_count", 0))
        new_count = st.number_input("Zähler anpassen", min_value=0,
            value=st.session_state.get("access_count", 0), step=1, key="access_count_input")
        if st.button("Zähler speichern"):
            try:
                with open(ACCESS_COUNT_FILE, "w") as f:
                    f.write(str(new_count))
                st.session_state.access_count = new_count
                st.success("Zähler aktualisiert.")
            except Exception as e:
                st.error(f"Fehler: {e}")
        st.markdown("**System-Prompt bearbeiten:**")
        if "prompt_edit_buffer" not in st.session_state:
            st.session_state.prompt_edit_buffer = load_prompt()
        new_prompt = st.text_area("System-Prompt", st.session_state.prompt_edit_buffer, height=200,
                                  key="prompt_edit_area")
        if st.button("Prompt speichern"):
            try:
                cfg = load_config()
                cfg["system_prompt"] = new_prompt
                save_config(cfg)
                st.success("Prompt gespeichert! Änderungen sind sofort aktiv.")
                st.session_state.prompt_edit_buffer = new_prompt
                st.session_state._prompt_reload_flag = True
                st.session_state.pop("current_prompt", None)
            except Exception as e:
                st.error(f"Fehler beim Speichern: {e}")
        st.markdown("**Initiale Nachricht bearbeiten:**")
        if "initial_m_edit_buffer" not in st.session_state:
            st.session_state.initial_m_edit_buffer = load_initial_message()
        new_message = st.text_area("Initial Message", st.session_state.initial_m_edit_buffer, height=200,
                                   key="initial_message_edit_area")
        if st.button("Nachricht speichern"):
            try:
                cfg = load_config()
                cfg["initial_message"] = new_message
                save_config(cfg)
                st.success("Änderung gespeichert! Änderungen sind sofort aktiv.")
                st.session_state.initial_m_edit_buffer = new_message
                st.session_state.im_reload_flag = True
                st.session_state.pop("current_initial_message", None)
            except Exception as e:
                st.error(f"Fehler beim Speichern: {e}")

        st.markdown("**Dokumente zur Wissensbasis hinzufügen:**")
        uploaded_files = st.file_uploader(
            "PDF-, TXT- oder DOCX-Dateien hochladen (mehrere möglich)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key="doc_uploader",
        )
        if uploaded_files:
            if st.button("Hochladen & Datenbank aktualisieren"):
                saved_names = []
                errors = []
                for uf in uploaded_files:
                    try:
                        # Strip any directory components from the filename
                        safe_name = Path(uf.name).name
                        # Enforce allowed extensions (defense-in-depth beyond Streamlit's type filter)
                        if Path(safe_name).suffix.lower() not in {".pdf", ".txt", ".docx"}:
                            errors.append(f"{uf.name}: Dateityp nicht erlaubt.")
                            continue
                        dest = folder_path / safe_name
                        with open(dest, "wb") as out_f:
                            out_f.write(uf.getbuffer())
                        saved_names.append(safe_name)
                    except Exception as e:
                        errors.append(f"{uf.name}: {e}")
                if saved_names:
                    with st.spinner("Datenbank wird neu aufgebaut …"):
                        try:
                            rebuild_DB(folder_path)
                            st.success(
                                f"{len(saved_names)} Datei(en) gespeichert und Datenbank aktualisiert: "
                                + ", ".join(saved_names)
                            )
                        except Exception as e:
                            st.error(f"Fehler beim Neuaufbau der Datenbank: {e}")
                for err in errors:
                    st.error(f"Fehler beim Speichern: {err}")

        st.markdown("**Aktuell in der Wissensbasis:**")
        try:
            doc_files = sorted([
                f for f in os.listdir(folder_path)
                if f.endswith((".pdf", ".txt", ".docx"))
            ])
            if doc_files:
                for fname in doc_files:
                    st.caption(f"📄 {fname}")
            else:
                st.caption("Keine Dateien vorhanden.")
        except Exception as e:
            st.error(f"Fehler beim Lesen des Ordners: {e}")

        st.markdown("**Datei löschen:**")
        try:
            doc_files_del = sorted([
                f for f in os.listdir(folder_path)
                if f.endswith((".pdf", ".txt", ".docx"))
            ])
            if doc_files_del:
                file_to_delete = st.selectbox(
                    "Datei auswählen", doc_files_del, key="file_to_delete"
                )
                if st.button("Löschen & Datenbank aktualisieren"):
                    try:
                        os.remove(folder_path / Path(file_to_delete).name)
                        with st.spinner("Datenbank wird neu aufgebaut …"):
                            rebuild_DB(folder_path)
                        st.success(f'"{file_to_delete}" gelöscht und Datenbank aktualisiert.')
                    except Exception as e:
                        st.error(f"Fehler beim Löschen: {e}")
            else:
                st.caption("Keine Dateien vorhanden.")
        except Exception as e:
            st.error(f"Fehler beim Lesen des Ordners: {e}")
    elif pw:
        st.error("Falsches Passwort.")

def _ollama_get(url, timeout=5):
    return requests.get(url, timeout=timeout)

def _ollama_post(url, payload, timeout=120, stream=False):
    headers = {"Content-Type": "application/json"}
    return requests.post(url, data=json.dumps(payload), headers=headers, timeout=timeout, stream=stream)

def ollama_ready(base_url: str, model: str) -> bool:
    base = base_url.rstrip('/')
    try:
        r = _ollama_get(f"{base}/api/version", timeout=3)
        if not r.ok:
            return False
    except Exception:
        return False

    # Check model availability; /api/show returns 200 when model exists (and lazily loads metadata)
    try:
        r = _ollama_post(f"{base}/api/show", {"name": model}, timeout=10)
        return r.ok
    except Exception:
        return False

def ollama_warmup(base_url: str, model: str, timeout: int = 300) -> bool:

   # Do a tiny non-streaming /api/generate call to load weights/kv-cache, so streaming chat won’t 503.
    base = base_url.rstrip('/')
    payload = {
        "model": model,
        "prompt": " ",
        "stream": False,
        "options": {"num_predict": 1}
    }
    try:
        r = _ollama_post(f"{base}/api/generate", payload, timeout=timeout, stream=False)
        return r.ok
    except Exception:
        return False

@st.cache_resource(show_spinner=False)
def _init_backend():
    start_DB(folder_path)
    warmup(load_ce=True)
    return True

_init_backend()

@st.cache_resource
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)

    css = """
    <style>
    :root {
      --background-color: transparent !important;
      --secondary-background-color: transparent !important;
      --primary-background-color: transparent !important;
    }

    /* Main background */
    [data-testid="stAppViewContainer"] {
      background-image: url("data:image/png;base64,___B64___") !important;
      background-size: cover !important;
      background-position: center center !important;
      background-repeat: no-repeat !important;
      background-attachment: fixed !important;
    }
    /* Clear other layers */
    .stApp, html, body { background: transparent !important; }
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
      background: transparent !important;
      backdrop-filter: none !important;
      box-shadow: none !important;
    }
    /* Remove Streamlit's overlay fades */
    [data-testid="stAppViewContainer"]::before,
    [data-testid="stAppViewContainer"]::after {
      content: none !important;
      background: none !important;
    }

    /* Reduce container padding */
    main .block-container {
      padding-top: 0 !important;
      padding-bottom: 0 !important;
    }
    /* Optional: hide footer/badge */
    footer, a.viewerBadge_container__1QSob, .viewerBadge_link__1S137 {
      display: none !important;
    }
    /* Bottom frame wrappers */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"]{
      background: transparent !important;
      background-image: none !important;   /* kills any gradient */
      box-shadow: none !important;
      border: 0 !important;
      backdrop-filter: none !important;
    }

    /* Just in case Streamlit adds decorative pseudo-elements here */
    [data-testid="stBottom"]::before,
    [data-testid="stBottom"]::after{
      content: none !important;
      background: none !important;
      background-image: none !important;
      box-shadow: none !important;
      border: 0 !important;
    }

    /* The element-container that hosts the chat input inside stBottom */
    [data-testid="stBottom"] .stElementContainer,
    [data-testid="stBottom"] .stChatInput{
      background: transparent !important;
      border: 0 !important;
      box-shadow: none !important;
    }

    </style>
    """

    st.markdown(css.replace("___B64___", bin_str), unsafe_allow_html=True)

set_png_as_page_bg(image_path / 'background.png')
st.title("Dialogos BNE")

# -------- Model config via env (no UI controls) --------
MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")  # e.g., "llama3", "llama3:8b-instruct"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# ----- Optional other AI's on a different location (not tested) -------
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")   # "ollama" or "openai"
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "none")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "default")

def ollama_chat(model: str, base_url: str, system_prompt: str, messages,
                temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 512, stream: bool = True):
    base = base_url.rstrip('/')
    chat_url = f"{base}/api/chat"

    # Preflight: ensure server+model, then warm once per model
    warmed_key = f"__ollama_warmed::{model}"
    if not ollama_ready(base, model):
        raise RuntimeError(f"Ollama-Server/Modell nicht erreichbar (Server: {base}, Modell: {model}). "
                           f"Starte den Server mit `ollama serve` und prüfe, ob `{model}` installiert ist (`ollama list`).")

    if not st.session_state.get(warmed_key, False):
        if ollama_warmup(base, model):
            st.session_state[warmed_key] = True

    payload = {
        "model": model,
        "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
        }
    }
    # Retry loop for transient 5xx / connection errors on the initial request
    max_attempts = 3
    backoff = 2.0
    attempt = 0
    while True:
        attempt += 1
        try:
            if stream:
                # Use a finite timeout to avoid hanging forever; streaming yields lines progressively
                with _ollama_post(chat_url, payload, timeout=300, stream=True) as r:
                    # If server needs a second chance on first byte, raise_for_status() will trigger retry
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if "message" in obj and "content" in obj["message"]:
                            yield obj["message"]["content"]
                        if obj.get("done"):
                            break
                break  # success, exit retry loop
            else:
                r = _ollama_post(chat_url, payload, timeout=300, stream=False)
                r.raise_for_status()
                data = r.json()
                return data.get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            status = getattr(e.response, "status_code", None)
            transient = status in (500, 502, 503, 504) or isinstance(e, (
            requests.exceptions.ConnectionError, requests.exceptions.Timeout))
            if attempt < max_attempts and transient:
                time.sleep(backoff)
                backoff *= 1.5
                continue
            # If it’s a non-transient error or we exhausted retries, rethrow
            raise
# Here is the openai method !-------
def openai_chat(model, base_url, api_key, system_prompt, messages,
                temperature=0.7, top_p=0.9, max_tokens=512, stream=True):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    max_attempts, backoff, attempt = 3, 2.0, 0
    while True:
        attempt += 1
        try:
            r = requests.post(url, json=payload, headers=headers,
                              stream=stream, timeout=300)
            r.raise_for_status()
            if stream:
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                break
            else:
                return r.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            status = getattr(e.response, "status_code", None)
            transient = status in (500, 502, 503, 504) or isinstance(
                e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))
            if attempt < max_attempts and transient:
                time.sleep(backoff)
                backoff *= 1.5
                continue
            raise

if "messages" not in st.session_state:
    st.session_state.messages = []
if "topic" not in st.session_state:
    st.session_state.topic = None
if "background_notes" not in st.session_state:
    st.session_state.background_notes = ""

def ensure_starter_message():
    if len(st.session_state.messages) == 0:
        initial_text = get_current_initial_message()
        st.session_state.messages.append({
            "role": "assistant",
            "content": initial_text
        })

ensure_starter_message()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def read_access_count() -> int:
    try:
        if ACCESS_COUNT_FILE.exists():
            with open(ACCESS_COUNT_FILE, "r", encoding="utf-8") as f:
                return int((f.read().strip() or "0"))
        return 0
    except Exception:
        return -1

def increment_access_count():
    try:
        if ACCESS_COUNT_FILE.exists():
            with open(ACCESS_COUNT_FILE, "r") as f:
                count = int(f.read().strip() or "0")
        else:
            count = 0
        count += 1
        with open(ACCESS_COUNT_FILE, "w") as f:
            f.write(str(count))
        return count
    except Exception:
        return -1

# Increment exactly once per user session/tab
if "access_count_initialized" not in st.session_state:
    st.session_state.access_count_initialized = True
    st.session_state.access_count = increment_access_count()
else:
    # on reruns: do NOT increment; just keep current or refresh from file
    st.session_state.access_count = read_access_count()

user_input = st.chat_input("Nachricht eingeben …")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Topic & retrieval
    if st.session_state.topic is None:
        st.session_state.topic = user_input.strip()
        ctx, cites = build_context(
            st.session_state.topic,
            variants=3, first_stage_k=6, final_k=5, max_chars_per_passage=700, use_cross_encoder=True
        )
        st.session_state.background_notes = ctx or "(keine Treffer – stelle generische Fragen auf Basis des Themas)"
    else:
        composite_query = f"{st.session_state.topic}\n\nLernenden-Turn: {user_input}"
        ctx, cites = build_context(
            composite_query,
            variants=2, first_stage_k=5, final_k=5, max_chars_per_passage=700, use_cross_encoder=True
        )
        if ctx:
            st.session_state.background_notes = ctx

    # Build system prompt (inject hidden notes)
    system_prompt = get_current_prompt().format(background_notes=st.session_state.background_notes)

    # LLM call (stream to chat)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        reply = ""
        acquired_immediately = _OLLAMA_SEM.acquire(blocking=False)
        if not acquired_immediately:
            placeholder.info("Das Modell ist gerade ausgelastet. Bitte kurz warten …")
            _OLLAMA_SEM.acquire()
        try:
            placeholder.empty()
            try:
                if LLM_BACKEND == "openai":
                    _gen = openai_chat(
                        model=OPENAI_MODEL,
                        base_url=OPENAI_API_URL,
                        api_key=OPENAI_API_KEY,
                        system_prompt=system_prompt,
                        messages=st.session_state.messages,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_tokens=MAX_TOKENS,
                        stream=True,
                    )
                else:
                    _gen = ollama_chat(
                        model=MODEL,
                        base_url=OLLAMA_URL,
                        system_prompt=system_prompt,
                        messages=st.session_state.messages,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_tokens=MAX_TOKENS,
                        stream=True,
                    )
                for chunk in _gen:
                    reply += chunk
                    placeholder.markdown(reply)
                assistant_reply = reply.strip()
            except Exception as e:
                print(f"[ERROR] LLM call failed: {e}")
                assistant_reply = (
                    "Entschuldige, der KI-Server ist momentan nicht erreichbar. "
                    "Bitte versuche es in einem Moment erneut."
                )
                placeholder.markdown(assistant_reply)
        finally:
            _OLLAMA_SEM.release()

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.rerun()