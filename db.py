import os
from pathlib import Path
import chromadb
import pymupdf
import numpy as np
import plotly.express as px
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from sklearn.decomposition import PCA

device = "cuda" if torch.cuda.is_available() else "cpu"
current_dir = Path(__file__).resolve()
folder_path = current_dir.parent / "docs"
db_path = current_dir.parent / ".chroma"
db_path.mkdir(exist_ok=True)
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
client = chromadb.PersistentClient(path=str(db_path))
collection = client.get_or_create_collection(name="my_docs")

print('DB')

def load_and_chunk(folder_path: Path):
    texts, metas = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            path = folder_path / filename
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
            texts.append(txt)
            metas.append({"source": filename})

    # Good defaults for German/paragraph-ish text:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,          # ~800–1200 is a sweet spot
        chunk_overlap=150,        # 100–200 keeps continuity
        length_function=len,      # simple & fast; switch to token counter if you like
        add_start_index=True,
        separators=[
            "\n\n",               # paragraphs
            "\n",                 # lines
            ". ", "? ", "! ",     # sentence-ish
            " ",                  # words
            ""                    # fallback to characters
        ],
    )

    # Create LangChain Documents so we get start indices in metadata
    docs = splitter.create_documents(texts, metadatas=metas)

    # Flatten to plain strings + metadatas for Chroma
    chunk_texts = [d.page_content for d in docs]
    chunk_metas = []
    for i, d in enumerate(docs):
        m = dict(d.metadata)  # {'source': ..., 'start_index': ...}
        m["chunk_index"] = i
        chunk_metas.append(m)

    # Stable, human-readable IDs: "<file>:<start_index>"
    ids = [
        f"{m.get('source','doc')}:{m.get('start_index', 0)}:{m.get('chunk_index', i)}"
        for i, m in enumerate(chunk_metas)
    ]

    print(f"Split {len(texts)} files into {len(chunk_texts)} chunks.")
    return chunk_texts, chunk_metas, ids

def pdf_to_txt(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            doc = pymupdf.open(os.path.join(folder_path, filename))  # open a document
            out = open(os.path.join(folder_path, filename[:-4])+".txt", "wb") # create a text output
            for page in doc:  # iterate the document pages
                text = page.get_text().encode("utf8")  # get plain text (is in UTF-8)
                out.write(text)  # write text of page
                out.write(bytes((12,)))  # write page delimiter (form feed 0x0C)
            out.close()
    return load_and_chunk(folder_path)
def visualize_embeddings(embeddings, labels):
    # ensure array
    X = np.asarray(embeddings)
    n_samples, n_features = X.shape
    if n_samples == 0:
        print("No embeddings to visualize.")
        return

    # pick components safely
    n_components = min(3, n_samples, n_features)

    pca = PCA(n_components=n_components)
    vis = pca.fit_transform(X)

    if n_components >= 3:
        fig = px.scatter_3d(
            x=vis[:, 0], y=vis[:, 1], z=vis[:, 2],
            text=labels,
            labels={"x": "PCA 1", "y": "PCA 2", "z": "PCA 3"},
            title="PCA of Embeddings (3D)",
        )
    elif n_components == 2:
        fig = px.scatter(
            x=vis[:, 0], y=vis[:, 1],
            text=labels,
            labels={"x": "PCA 1", "y": "PCA 2"},
            title="PCA of Embeddings (2D)",
        )
    else:  # n_components == 1
        fig = px.scatter(
            x=vis[:, 0], y=[0]*n_samples,
            text=labels,
            labels={"x": "PCA 1", "y": ""},
            title="PCA of Embeddings (1D)",
        )

    fig.update_traces(marker=dict(size=6))
    fig.show()


# -------- Query Rewriter (multi-query expansion) --------
_rewriter = None
def rewrite_query_multi(query: str, num_variants: int = 3):
    """Return [original, v1, v2, ...] short German rewrites to broaden recall."""
    if num_variants <= 0:
        return [query]
    global _rewriter
    if _rewriter is None:
        _rewriter = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device=device,
            max_new_tokens=64
        )
    prompt = (
        "Schreibe {n} unterschiedliche, kurze Suchanfragen auf Deutsch, "
        "die die gleiche Absicht haben wie die Originalanfrage.\n"
        f"Original: {query}\n"
        "Gib jede Variante auf einer neuen Zeile aus."
    )
    out = _rewriter(prompt)[0]["generated_text"]
    variants = []
    for line in [s.strip("-• \t") for s in out.splitlines()]:
        if line and line.lower() != query.lower() and line not in variants:
            variants.append(line)
    return [query] + variants[:num_variants]

# -------- First-stage retrieval from Chroma (vector search) --------
def retrieve_candidates(queries, top_k_per_query=5):
    seen, candidates = set(), []
    for q in queries:
        q_emb = model.encode([q]).tolist()
        res = collection.query(
            query_embeddings=q_emb,
            n_results=top_k_per_query,
            include=["documents", "metadatas"]
        )
        ids  = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        for _id, doc, meta in zip(ids, docs, metas):
            if _id not in seen:
                seen.add(_id)
                candidates.append({"id": _id, "doc": doc, "meta": meta})
    return candidates

# -------- Second-stage reranking (cross-encoder) --------
_reranker = None
def rerank(query: str, candidates, top_k=5):
    """Cross-encode (query, passage) pairs and return top_k best."""
    if not candidates:
        return []
    global _reranker
    if _reranker is None:
        # Multilingual cross-encoder trained on MS MARCO
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    pairs = [(query, c["doc"]) for c in candidates]
    scores = _reranker.predict(pairs)  # higher = better
    order = np.argsort(scores)[::-1]
    ranked = []
    for i in order[:top_k]:
        ranked.append({**candidates[i], "score": float(scores[i])})
    print(ranked)
    return ranked

# -------- One-call helper you can use instead of get_relevant_docs --------
# -------- Build a short context string for the LLM --------
def build_context(query: str, variants=3, first_stage_k=6, final_k=5, max_chars_per_passage=700):
    qs = rewrite_query_multi(query, num_variants=variants)
    cands = retrieve_candidates(qs, top_k_per_query=first_stage_k)
    ranked = rerank(query, cands, top_k=final_k)

    if not ranked:
        return "", []

    ctx_lines = []
    cites = []
    for i, r in enumerate(ranked, 1):
        src = r["meta"].get("source") if isinstance(r["meta"], dict) else None
        snippet = r["doc"].strip().replace("\n", " ")
        if len(snippet) > max_chars_per_passage:
            snippet = snippet[:max_chars_per_passage] + "…"
        cite = f"[{i}: {src or r['id']}]"
        ctx_lines.append(f"{cite} {snippet}")
        cites.append({"label": cite, "id": r["id"], "source": src, "score": r["score"]})
    context = "\n".join(ctx_lines)
    return context, cites

def start_DB(folder_path):
    client = chromadb.PersistentClient(path=str(db_path))
    # Build chunks
    chunk_texts, chunk_metas, ids = pdf_to_txt(folder_path)

    # Embed *strings*
    embeddings = model.encode(chunk_texts, show_progress_bar=True, batch_size=32)

    # Upsert with metadatas
    collection.upsert(
        ids=ids,
        documents=chunk_texts,
        metadatas=chunk_metas,
        embeddings=embeddings
    )

    try:
        client.persist()
    except Exception:
        pass

query_text = "Wie wirken sich bewaffnete Konflikte auf die mentale Gesundheit aus?"

# Zur Visualisierung:
client = chromadb.PersistentClient(path=str(db_path))
collection = client.get_or_create_collection(name="my_docs")
result = collection.get(include=["embeddings"])
embeddings = result["embeddings"]
labels = result["ids"]
#visualize_embeddings(embeddings, labels)

