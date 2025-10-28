import os
import json
import requests
import streamlit as st
from pathlib import Path
import base64
#from db import build_context
from db_optimized import build_context, start_DB, warmup


current_dir = Path(__file__).resolve()
image_path = current_dir.parent / "images"
folder_path = current_dir.parent / "docs"

st.set_page_config(page_title="Dialogos BNE", page_icon="🦙", layout="centered")

import time

def _ollama_get(url, timeout=5):
    return requests.get(url, timeout=timeout)

def _ollama_post(url, payload, timeout=120, stream=False):
    headers = {"Content-Type": "application/json"}
    return requests.post(url, data=json.dumps(payload), headers=headers, timeout=timeout, stream=stream)

def ollama_ready(base_url: str, model: str) -> bool:
    """Check server is reachable and model is available."""
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
    """
    Do a tiny non-streaming /api/generate call to load weights/kv-cache, so streaming chat won’t 503.
    """
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
    # Ensure index exists and models are in-memory
    start_DB(folder_path)
    return True

_init_backend()

warmup(load_ce=True)
st.session_state.last_warmup_ce = True

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

set_png_as_page_bg(image_path/'background.png')
st.title("Dialogos BNE")

# -------- Model config via env (no UI controls) --------
MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")                 # e.g., "llama3", "llama3:8b-instruct"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# -------- Initial System Prompt (German, Socratic) --------
INITIAL_SYSTEM_PROMPT_DE = """Rolle: Du bist ein geduldiger, sokratischer Tutor. Du hilfst Lernenden,
indem du gezielte Fragen stellst, statt Lösungen vorzugeben. Du führst durch kleine,
logische Schritte, prüfst (Miss-)Verständnisse und bietest kurze, konkrete Hinweise.

Sprache & Stil:
- Sprich Deutsch, freundlich und klar.
- Stelle jeweils nur EINE fokussierte Frage pro Antwort (max. 2–3 Sätze).
- Bestätige richtige Zwischenschritte knapp; bei Irrtümern stelle Nachfragen.
- Vermeide lange Monologe, Aufzählungslisten und direkte Lösungen.

Verwendung von Hintergrundwissen:
- Du erhältst **Hintergrundnotizen** aus einer lokalen Datenbank (RAG). Sie sind nur für dich.
- Nutze sie, um deine nächste Frage gezielt zu wählen, aber:
  * Zitiere oder verrate diese Notizen nicht wörtlich.
  * Erfinde keine Inhalte außerhalb der Notizen.
  * Gib keine langen Zusammenfassungen aus den Notizen wieder.
  * Formuliere Fragen so, dass der/die Lernende die Einsichten selbst entdeckt.
- Wenn ausdrücklich nach Quellen gefragt wird („Woher weißt du das?“), gib nur Dateinamen/Labels an, nicht die Passagen.

Themenstart:
- Beginne immer mit: „Welches Thema möchtest du heute üben?“
- Die erste Nutzerantwort definiert das Thema. Danach richtest du jede Frage an diesem Thema aus.
- Wenn keine oder wenige Notizen gefunden werden, bleibe bei sokratischen Fragen auf Basis des Themas.

Hinweise bei Lösungswunsch:
- Wenn die Lernenden direkt nach der Lösung fragen, erinnere kurz an die Methode und biete 2–3 präzise Hinweise.
- Gib keine vollständigen Lösungen.

WICHTIG:
- Keine Halluzinationen. Wenn dir etwas unklar ist, frage nach.
- Bleibe strikt im sokratischen Modus (Fragen-basiert).
- Eine Antwort = eine fokussierte Frage, ggf. ein sehr kurzer Hinweis.

<BEGIN HINTERGRUNDNOTIZEN – NICHT OFFENLEGEN>
{background_notes}
<END HINTERGRUNDNOTIZEN>
"""

# -------- Ollama chat helper (streaming) --------
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
        else:
            # We still proceed; warmup is best-effort
            pass

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
            # Status code if available
            status = getattr(e.response, "status_code", None)
            transient = status in (500, 502, 503, 504) or isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))
            if attempt < max_attempts and transient:
                time.sleep(backoff)
                backoff *= 1.5
                continue
            # If it’s a non-transient error or we exhausted retries, rethrow
            raise

# -------- Session state --------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "topic" not in st.session_state:
    st.session_state.topic = None
if "background_notes" not in st.session_state:
    st.session_state.background_notes = ""
if "last_cites" not in st.session_state:
    st.session_state.last_cites = []
if "show_sources" not in st.session_state:
    st.session_state.show_sources = False

# -------- Starter assistant turn --------
def ensure_starter_message():
    if len(st.session_state.messages) == 0:
        # Short help is given via /help, to keep the first turn focused & socratic.
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hallo, ich bin dein sokratischer Gesprächspartner. Ich werde dich bei deinem Gedankengang zu "
                       "Themen im Bereich 'Nachhaltigkeit' begleiten. Über welches Thema möchtest du dich mit mir "
                       "austauschen?"
        })

ensure_starter_message()

# -------- Render history (chat-only) --------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -------- Slash-command handler --------
def handle_command(cmd: str) -> bool:
    """Returns True if a command was handled (i.e., don't call the LLM)."""
    parts = cmd.strip().split()
    head = parts[0].lower()

# -------- Chat input (the only UI) --------
user_input = st.chat_input("Nachricht eingeben …")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Commands short-circuit the LLM
    if user_input.strip().startswith("/"):
        handled = handle_command(user_input)
        if handled:
            st.rerun()

    # Topic & retrieval
    if st.session_state.topic is None:
        st.session_state.topic = user_input.strip()
        ctx, cites = build_context(
            st.session_state.topic,
            variants=3, first_stage_k=6, final_k=5, max_chars_per_passage=700, use_cross_encoder=True
        )
        st.session_state.background_notes = ctx or "(keine Treffer – stelle generische Fragen auf Basis des Themas)"
        st.session_state.last_cites = cites
    else:
        composite_query = f"{st.session_state.topic}\n\nLernenden-Turn: {user_input}"
        ctx, cites = build_context(
            composite_query,
            variants=2, first_stage_k=5, final_k=5, max_chars_per_passage=700, use_cross_encoder=True
        )
        if ctx:
            st.session_state.background_notes = ctx
            st.session_state.last_cites = cites

    # Build system prompt (inject hidden notes)
    system_prompt = INITIAL_SYSTEM_PROMPT_DE.format(background_notes=st.session_state.background_notes)

    # LLM call (stream to chat)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        reply = ""
        try:
            for chunk in ollama_chat(
                model=MODEL,
                base_url=OLLAMA_URL,
                system_prompt=system_prompt,
                messages=st.session_state.messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                stream=True
            ):
                reply += chunk
                placeholder.markdown(reply)
            assistant_reply = reply.strip()
        except Exception as e:
            assistant_reply = f"Entschuldige, beim Aufruf des lokalen Modells ist ein Fehler aufgetreten: `{e}`"
            placeholder.markdown(assistant_reply)

    # Save assistant turn
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

    # Optional: source labels via /sources on (never reveal note content)
    if st.session_state.show_sources and st.session_state.last_cites:
        labels = []
        for c in st.session_state.last_cites:
            src = c.get("source") or c.get("id")
            labels.append(f"- {c.get('label')} — `{src}` (Score: {c.get('score'):.3f})")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "**Quellen (Labels/Dateinamen):**\n" + "\n".join(labels)
        })

    st.rerun()
