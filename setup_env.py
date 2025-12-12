#!/usr/bin/env python3
import os
import platform
import subprocess
import sys
import shutil
from pathlib import Path

import time
import socket

def _is_port_open(host: str, port: int, timeout=0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port)); return True
        except Exception:
            return False

def ensure_ollama_server():
    """
    Ensure the ollama server is running locally.
    Tries systemd (if available) then falls back to spawning `ollama serve` in the background.
    Waits until port 11434 is accepting connections (max ~12s).
    """
    if not shutil.which("ollama"):
        print("⚠️ Ollama CLI not found; cannot start server.")
        return False

    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    host_ip, host_port = (host.split(":")[0], int(host.split(":")[1]))
    if _is_port_open(host_ip, host_port):
        print(f"✓ Ollama server already running at {host}")
        return True

    # Try systemd user service (if available)
    if shutil.which("systemctl"):
        # Try user service first (no sudo)
        try:
            subprocess.run(["systemctl", "--user", "start", "ollama"], check=False)
        except Exception:
            pass
        # Try system service as a fallback (requires sudo; best-effort)
        if not _is_port_open(host_ip, host_port):
            subprocess.run(["sudo", "systemctl", "start", "ollama"], check=False)

    # If still not up, spawn a background server (WSL friendly)
    if not _is_port_open(host_ip, host_port):
        print("Starting `ollama serve` in the background…")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for readiness
    for _ in range(24):  # ~12s max
        if _is_port_open(host_ip, host_port):
            print(f"✓ Ollama server ready at {host_ip}:{host_port}")
            return True
        time.sleep(0.5)

    print("⚠️ Could not start or reach the Ollama server. You can start it manually with `ollama serve`.")
    return False

# ----------------- helpers -----------------
def run(cmd, check=True):
    print(f"\n>>> {cmd}")
    return subprocess.run(cmd, shell=True, check=check)

def have_module(mod_name):
    try:
        __import__(mod_name)
        return True
    except Exception:
        return False

def pip_install(pkg_import_name, pip_spec=None):
    """
    pkg_import_name: name used in 'import ...'
    pip_spec: what to pass to pip install (defaults to pkg_import_name)
    """
    if pip_spec is None:
        pip_spec = pkg_import_name
    if have_module(pkg_import_name.replace("-", "_")):
        print(f"✓ {pip_spec} already installed")
        return
    print(f"Installing {pip_spec} ...")
    run(f"{sys.executable} -m pip install -q {pip_spec}")

# ----------------- start -----------------
system = platform.system().lower()
print(f"Detected OS: {system}")

# Ensure pip exists
pip_ok = shutil.which("pip") or have_module("pip")
if not pip_ok and "linux" in system:
    print("pip not found — installing via apt...")
    run("sudo apt update")
    run("sudo apt install -y python3-pip")
elif not pip_ok and "windows" in system:
    print("pip not found on Windows. Install Python from https://www.python.org/downloads/ with 'Add to PATH' enabled.")
    sys.exit(1)

# Upgrade pip (best effort)
run(f"{sys.executable} -m pip install --upgrade pip", check=False)

# Core packages (safe on both OSes)
core_packages = [
    ("streamlit", None),
    ("langchain", None),
    ("ollama", None),
    ("huggingface_hub", None),
    ("sentence_transformers", "sentence-transformers"),
    ("transformers", None),
    ("accelerate", None),
    ("faiss", "faiss-cpu"),
    ("chromadb", None)
]
optional_packages = [
    ("plotly", None),
    ("fitz", "pymupdf"),
]

for mod, spec in core_packages + optional_packages:
    pip_install(mod, spec)

# Ollama install / check
if "windows" in system:
    print("\n⚠️  On Windows, install Ollama manually if not present:")
    print("    https://ollama.com/download/windows")
else:
    if shutil.which("ollama"):
        print("✓ Ollama already installed")
        # --- Llama3 7B model pull via Ollama ---
        # --- Verify model presence (works for both new & existing installs) ---
        print("\nChecking for Llama3 8B model...")
        # NEW: ensure server is running
        if ensure_ollama_server():
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if "llama3:8b" in result.stdout:
                print("✓ Llama3 8B model already available.")
            else:
                print("⬇️  Pulling Llama3 8B model (this may take a while)…")
                run("ollama pull llama3:8b", check=False)
                # Re-check
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                if "llama3:8b" in result.stdout:
                    print("✓ Llama3 8B model ready.")
                else:
                    print("⚠️ Pull did not complete. Try manually: `ollama pull llama3:8b`")
        else:
            print("⚠️ Skipping model pull because the Ollama server isn’t reachable.")

    else:
        print("Installing Ollama for Linux...")
        run("curl -fsSL https://ollama.com/install.sh | sh", check=False)
        if shutil.which("ollama"):
            print("✓ Ollama installed")
            # --- Llama3 7B model pull via Ollama ---
            # --- Verify model presence (works for both new & existing installs) ---
            print("\nChecking for Llama3 8B model...")
            # NEW: ensure server is running
            if ensure_ollama_server():
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                if "llama3:8b" in result.stdout:
                    print("✓ Llama3 8B model already available.")
                else:
                    print("⬇️  Pulling Llama3 8B model (this may take a while)…")
                    run("ollama pull llama3:8b", check=False)
                    # Re-check
                    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                    if "llama3:8b" in result.stdout:
                        print("✓ Llama3 8B model ready.")
                    else:
                        print("⚠️ Pull did not complete. Try manually: `ollama pull llama3:8b`")
            else:
                print("⚠️ Skipping model pull because the Ollama server isn’t reachable.")

        else:
            print("⚠️ Ollama install could not be confirmed. You can install manually: https://ollama.com")

# Model downloads
print("\nPreparing local Hugging Face models directory...")
models = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "models/cross-encoder-ms-marco-MiniLM-L-6-v2",
    "sentence-transformers/all-MiniLM-L6-v2": "models/all-MiniLM-L6-v2",
}
pip_install("huggingface_hub")
from huggingface_hub import snapshot_download

for repo_id, dest in models.items():
    dest_path = Path(dest)
    if dest_path.exists() and any(dest_path.iterdir()):
        print(f"✓ Model already present: {dest}")
        continue
    print(f"⬇️  Downloading {repo_id} → {dest}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=dest,
        local_dir_use_symlinks=False,
        resume_download=True,
        tqdm_class=None,
    )
    print(f"✓ Downloaded: {dest}")

# Torch + CUDA check
print("\nChecking PyTorch & CUDA:")
if have_module("torch"):
    import torch
    print(f"✓ torch {torch.__version__} — CUDA available: {torch.cuda.is_available()}")
else:
    print("⚠️ PyTorch not installed via pip. Follow https://pytorch.org/ to install the CUDA build.")

print("\n✅ Setup complete. You can now run:")
print("   streamlit run app.py")
