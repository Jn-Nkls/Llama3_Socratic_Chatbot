#!/usr/bin/env python3
from pathlib import Path
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": MODELS_DIR / "all-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": MODELS_DIR / "cross-encoder-ms-marco-MiniLM-L-6-v2",
}
def download_model(repo_id: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        print(f"✓ Model already present: {repo_id} -> {dest}")
        return
    print(f"⬇ Downloading {repo_id} -> {dest}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    print(f"✓ Downloaded: {repo_id}")
def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for repo_id, dest in MODELS.items():
        download_model(repo_id, dest)
if __name__ == "__main__":
    main()