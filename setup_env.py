#!/usr/bin/env python3
import os
import platform
import subprocess
import sys
from pathlib import Path

# ----- Utility helpers -----
def run(cmd):
    """Run a command and stream output."""
    print(f"\n>>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def pip_install(pkg):
    """Install a pip package only if missing."""
    try:
        __import__(pkg.split("==")[0].split(">=")[0].replace("-", "_"))
        print(f"✓ {pkg} already installed")
    except ImportError:
        print(f"Installing {pkg} ...")
        run(f"{sys.executable} -m pip install -q {pkg}")

# ----- Step 1: Basic info -----
system = platform.system().lower()
print(f"Detected OS: {system}")

# Ensure pip itself is ready
run(f"{sys.executable} -m ensurepip --upgrade")
run(f"{sys.executable} -m pip install --upgrade pip")

# ----- Step 2: Core dependencies -----
core_packages = [
    "streamlit",
    "langchain",
    "ollama",
    "huggingface_hub",
    "sentence-transformers",
    "transformers",
    "accelerate",
    "faiss-cpu"  # safe fallback; works even with CUDA available
]
optional_packages = ["plotly", "pymupdf"]

for pkg in core_packages + optional_packages:
    pip_install(pkg)

# ----- Step 3: Ollama install instructions -----
if "windows" in system:
    print("\n⚠️  On Windows, make sure you've installed Ollama manually:")
    print("    https://ollama.com/download/windows")
else:
    print("\nInstalling Ollama for Linux (if not installed)...")
    if not shutil.which("ollama"):
        run("curl -fsSL https://ollama.com/install.sh | sh")
    else:
        print("✓ Ollama already installed")

# ----- Step 4: Download models if missing -----
models = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "models/cross-encoder-ms-marco-MiniLM-L-6-v2",
    "sentence-transformers/all-MiniLM-L6-v2": "models/all-MiniLM-L6-v2"
}

run(f"{sys.executable} -m pip install -q hf-transfer")  # makes hf download available
for repo, dest in models.items():
    dest_path = Path(dest)
    if dest_path.exists():
        print(f"✓ Model already downloaded: {dest}")
    else:
        print(f"⬇️  Downloading model: {repo}")
        run(f"hf download {repo} --local-dir \"{dest}\"")

# ----- Step 5: Verify CUDA -----
print("\nChecking CUDA availability in PyTorch:")
try:
    import torch
    print(f"✓ Torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("⚠️  PyTorch not installed — please install the CUDA version manually from pytorch.org.")

# ----- Step 6: Done -----
print("\n✅ Environment setup complete!")
print("You can now run:")
print("   streamlit run app.py")
