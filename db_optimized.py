import os
import sys, multiprocessing as mp
import threading
from pathlib import Path
import multiprocessing as mp
from functools import lru_cache
import chromadb
import fitz  # PyMuPDF (import as fitz is faster to start)
import numpy as np
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from concurrent.futures import ThreadPoolExecutor

# -------------------- Global config & singletons --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
IS_WINDOWS = sys.platform.startswith("win")
torch.set_grad_enabled(False)  # SPEEDUP: no gradients anywhere
if device == "cuda":
    torch.backends.cudnn.benchmark = True  # SPEEDUP: better kernels after warmup

current_dir = Path(__file__).resolve()
folder_path = current_dir.parent / "docs"
db_path = current_dir.parent / ".chroma"
db_path.mkdir(exist_ok=True)

# One client/collection for the whole process (avoid re-creating)
_client = chromadb.PersistentClient(path=str(db_path))
_collection = _client.get_or_create_collection(name="my_docs")

# SentenceTransformer is expensive—load once
# SPEEDUP: larger batch + normalized outputs for better ANN behavior
_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
_model.max_seq_length = 256  # SPEEDUP: cap length—enough for chunks, less work

# Cross-encoder (optional) is heavy—lazy init
_reranker_lock = threading.Lock()
_reranker = None

def is_ce_loaded() -> bool:
    """Utility so the app can introspect CE status if needed."""
    print(_reranker)
    return _reranker is not None

# -------------------- Fast PDF → text (no disk writes) --------------------
def _pdf_to_text_one(pdf_path: Path) -> tuple[str, dict]:
    """Extract text from a single PDF file into memory (no temp .txt)."""
    doc = fitz.open(pdf_path)  # faster alias
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    text = "\n".join(parts)
    return text, {"source": pdf_path.name}

def _txt_to_text_one(txt_path: Path) -> tuple[str, dict]:
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read(), {"source": txt_path.name}

def load_all_texts(in_folder: Path) -> tuple[list[str], list[dict]]:
    """Load .pdf and .txt into memory, parallelizing PDFs."""
    texts, metas = [], []
    pdfs = []
    txts = []
    for fn in os.listdir(in_folder):
        p = in_folder / fn
        if fn.endswith(".pdf"):
            pdfs.append(p)
        elif fn.endswith(".txt"):
            txts.append(p)

    # SPEEDUP: parallel PDF extraction (CPU-bound, scales with cores)
    if pdfs:
        used_parallel = False
        if not IS_WINDOWS and len(pdfs) > 1:
            try:
                with mp.Pool(processes=min(len(pdfs), max(mp.cpu_count() - 1, 1))) as pool:
                    for text, meta in pool.map(_pdf_to_text_one, pdfs):
                        texts.append(text);
                        metas.append(meta)
                used_parallel = True
            except Exception:
                used_parallel = False

        if not used_parallel:
            # Windows-safe fallback: threads (or sequential if only 1 file)
            if len(pdfs) > 1:
                with ThreadPoolExecutor(max_workers=min(len(pdfs), max(os.cpu_count() - 1, 1))) as ex:
                    for text, meta in ex.map(_pdf_to_text_one, pdfs):
                        texts.append(text);
                        metas.append(meta)
            else:
                for p in pdfs:
                    text, meta = _pdf_to_text_one(p)
                    texts.append(text);
                    metas.append(meta)

    # Fast sequential for small .txt files
    for p in txts:
        text, meta = _txt_to_text_one(p)
        texts.append(text)
        metas.append(meta)

    return texts, metas

# -------------------- Chunking --------------------
@lru_cache(maxsize=1)  # splitter is pure config → cache
def _splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

def chunk_documents(texts: list[str], metas: list[dict]):
    docs = _splitter().create_documents(texts, metadatas=metas)
    chunk_texts = [d.page_content for d in docs]
    chunk_metas = []
    for i, d in enumerate(docs):
        m = dict(d.metadata)
        m["chunk_index"] = i
        chunk_metas.append(m)
    ids = [
        f"{m.get('source','doc')}:{m.get('start_index', 0)}:{m.get('chunk_index', i)}"
        for i, m in enumerate(chunk_metas)
    ]
    return chunk_texts, chunk_metas, ids

# -------------------- Embedding (batched, normalized) --------------------
def embed_texts(texts: list[str]) -> np.ndarray:
    # SPEEDUP: bigger batch, no progress bar, normalize for better ANN
    return _model.encode(
        texts,
        batch_size=128 if device == "cuda" else 64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

# -------------------- Retrieval & Reranking --------------------
def retrieve_candidates(queries, top_k_per_query=5):
    seen, candidates = set(), []
    for q in queries:
        q_emb = embed_texts([q])  # uses normalized vec
        res = _collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=top_k_per_query,
            include=["documents", "metadatas", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            if _id not in seen:
                seen.add(_id)
                candidates.append({"id": _id, "doc": doc, "meta": meta})
                candidates.append({"id": _id, "doc": doc, "meta": meta, "distance": float(dist)})
    return candidates

def _should_use_ce(cands, topk=5, top1_sim_thresh=0.82, gap_thresh=0.05, std_thresh=0.03):
    """
   Decide whether to run the cross-encoder.
    Heuristics:
      - If top-1 similarity is high and gap to #2 is big -> SKIP CE (easy query).
      - If similarities are bunched (low std) -> RUN CE (ambiguous).
    """
    # Work with the top-k by distance (ascending)
    have = [c for c in cands if c.get("distance") is not None]
    if not have:
        return True  # no distances -> play safe, run CE
    top = sorted(have, key=lambda x: x["distance"])[:max(2, topk)]
    sims = [1.0 - c["distance"] for c in top]
    if len(sims) < 2:
        return False
    top1, top2 = sims[0], sims[1]
    gap = top1 - top2
    if top1 >= top1_sim_thresh and gap >= gap_thresh:
        print("no ce")
        return False  # confident hit -> skip CE
    # If the top-k are very similar to each other, CE can help
    import numpy as _np
    if _np.std(sims) <= std_thresh:
        print("ce")
        return True
    # Default: run CE for safety on borderline cases
    print("default ce")
    return True

def ensure_reranker_loaded():
    global _reranker
    with _reranker_lock:
        if _reranker is None:
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
            if device == "cuda":
                try:
                    _reranker.model.half()
                except Exception:
                    pass

def rerank(query: str, candidates, top_k=5, enable=True):
    if not enable or not candidates:
        # simple cosine prefilter using the bi-encoder (cheap)
        return candidates[:top_k]
    ensure_reranker_loaded()
    pairs = [(query, c["doc"]) for c in candidates]
    scores = _reranker.predict(pairs)  # batched internally
    order = np.argsort(scores)[::-1]
    return [{**candidates[i], "score": float(scores[i])} for i in order[:top_k]]

def build_context(
    query: str,
    variants=0,                # SPEEDUP: default 0 (no rewrites)
    first_stage_k=8,           # pull a few more, still cheap
    final_k=5,
    max_chars_per_passage=700,
    use_cross_encoder=True,    # can toggle off for more speed
):
    cands = retrieve_candidates(query, top_k_per_query=first_stage_k)
    # GATE: only run CE when ambiguous (and only if caller allows CE)
    ce_enabled = use_cross_encoder and _should_use_ce(cands, topk=final_k)
    ranked = rerank(query, cands, top_k=final_k, enable=ce_enabled)

    if not ranked:
        return "", []

    ctx_lines, cites = [], []
    for i, r in enumerate(ranked, 1):
        src = r["meta"].get("source") if isinstance(r["meta"], dict) else None
        snippet = r["doc"].strip().replace("\n", " ")
        if len(snippet) > max_chars_per_passage:
            snippet = snippet[:max_chars_per_passage] + "…"
        cite = f"[{i}: {src or r['id']}]"
        ctx_lines.append(f"{cite} {snippet}")
        cites.append({"label": cite, "id": r["id"], "source": src, "score": float(r.get("score", 0.0))})
    return "\n".join(ctx_lines), cites

def warmup(load_ce: bool = True):
    """Preload models, build kernels, and warm ANN to kill cold-start lag."""
    # 1) Warm sentence-transformer + CUDA kernels
    _ = embed_texts(["warmup text"])
    # 2) Warm vector DB query path
    try:
        _collection.query(query_embeddings=_[:1].tolist(), n_results=1, include=["documents"])
    except Exception:
        pass
    # 3) Optionally warm cross-encoder --> Not necessary anymore --> always true rn
    if load_ce:
        ensure_reranker_loaded()
        try:
            _ = _reranker.predict([("warmup", "warmup")])
        except Exception:
            pass
    if device == "cuda":
        import torch as _torch
        _torch.cuda.synchronize()
    return True

def start_DB(in_folder: Path):
    if not os.listdir(folder_path):
        print("Folder is empty")
        try:
            _client.delete_collection(name="my_docs")
        except Exception:
            pass
        global _collection
        _collection = _client.get_or_create_collection(name="my_docs")
        try:
            _client.persist()
        except Exception:
            pass
    else:
        print("Folder is not empty")
        texts, metas = load_all_texts(in_folder)  # parallel PDFs, no .txt writes
        chunk_texts, chunk_metas, ids = chunk_documents(texts, metas)
        embeddings = embed_texts(chunk_texts)  # big batches, normalized
        _collection.upsert(
            ids=ids,
            documents=chunk_texts,
            metadatas=chunk_metas,
            embeddings=embeddings,
        )
        try:
            _client.persist()
        except Exception:
            pass


# -------------------- Small demo path (optional) --------------------
if __name__ == "__main__":
    print("Building/refreshing DB...")
    start_DB(folder_path)
    print("DB ready.")
