from pathlib import Path
from huggingface_hub import snapshot_download

base = Path("models"); base.mkdir(exist_ok=True)
snapshot_download("sentence-transformers/all-MiniLM-L6-v2", local_dir=base/"all-MiniLM-L6-v2")
snapshot_download("cross-encoder/ms-marco-MiniLM-L-6-v2", local_dir=base/"cross-encoder-ms-marco-MiniLM-L-6-v2")
print("Done.")
