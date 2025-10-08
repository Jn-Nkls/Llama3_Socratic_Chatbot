import argparse
import statistics as stats
import time
from pathlib import Path

import numpy as np

from db_optimized import (
    start_DB,
    build_context,
    folder_path,  # from db_optimized
    device as _device,
)
try:
    import torch
except Exception:
    torch = None

# ----------------------------
# Helpers
# ----------------------------
def now():
    return time.perf_counter()

def percentile(values, p):
    if not values:
        return float("nan")
    arr = np.sort(np.array(values, dtype=float))
    idx = int(np.ceil((p / 100.0) * len(arr))) - 1
    idx = max(0, min(idx, len(arr) - 1))
    return float(arr[idx])

def gpu_stats():
    if torch is None:
        return {}
    if _device == "cuda" and torch.cuda.is_available():
        return {
            "gpu_mem_alloc_MB": torch.cuda.memory_allocated() / (1024 ** 2),
            "gpu_mem_reserved_MB": torch.cuda.memory_reserved() / (1024 ** 2),
        }
    return {}

def print_block(title, lines):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    for line in lines:
        print(line)

# ----------------------------
# Benchmark
# ----------------------------
def build_queries(qfile: Path | None):
    if qfile and qfile.exists():
        return [q.strip() for q in qfile.read_text(encoding="utf-8").splitlines() if q.strip()]
    # Fallback toy set (edit to match your domain)
    return [
        "Wer ist der Autor von Frankenstein?",
        "Wie wirken sich bewaffnete Konflikte auf die mentale Gesundheit aus?",
        "Nenne die Kernpunkte der Vorlesung zur Suchalgorithmen.",
        "Was ist der Unterschied zwischen Precision@k und Recall@k?",
        "Erkläre die Funktionsweise von FAISS in wenigen Sätzen.",
        "Welche Parameter beeinflussen die Latenz eines RAG-Systems am meisten?",
        "Wie groß sollte die Chunkgröße typischerweise sein?",
        "Was ist der Nutzen eines Cross-Encoders beim Reranking?",
        "Wie kann man die Einbettungsdurchsatzrate erhöhen?",
        "Welche Rolle spielt Normalisierung von Embeddings?",
    ]

def time_ingestion(docs_dir: Path):
    t0 = now()
    start_DB(docs_dir)
    t1 = now()
    return t1 - t0

def bench_queries(queries, runs, warmup, cfg):
    latencies = []
    # Warmup (builds CUDA kernels, caches, ANN pages, etc.)
    for i in range(warmup):
        _ = build_context(
            queries[i % len(queries)],
            variants=cfg["variants"],
            first_stage_k=cfg["first_stage_k"],
            final_k=cfg["final_k"],
            max_chars_per_passage=cfg["max_chars"],
            use_cross_encoder=cfg["use_cross_encoder"],
        )

    # Timed runs
    t_batch_start = now()
    for i in range(runs):
        q = queries[i % len(queries)]
        t0 = now()
        _ctx, _cites = build_context(
            q,
            variants=cfg["variants"],
            first_stage_k=cfg["first_stage_k"],
            final_k=cfg["final_k"],
            max_chars_per_passage=cfg["max_chars"],
            use_cross_encoder=cfg["use_cross_encoder"],
        )
        t1 = now()
        latencies.append(t1 - t0)
    t_batch_end = now()

    total_time = t_batch_end - t_batch_start
    qps = runs / total_time if total_time > 0 else float("inf")

    summary = {
        "runs": runs,
        "mean_ms": 1000 * (sum(latencies) / len(latencies)),
        "median_ms": 1000 * stats.median(latencies),
        "p95_ms": 1000 * percentile(latencies, 95),
        "p99_ms": 1000 * percentile(latencies, 99),
        "min_ms": 1000 * min(latencies),
        "max_ms": 1000 * max(latencies),
        "qps": qps,
        **gpu_stats(),
    }
    return summary, latencies

def main():
    parser = argparse.ArgumentParser(description="Speed test for db_optimized RAG backend")
    parser.add_argument("--docs", type=str, default=str(folder_path), help="Path to docs folder")
    parser.add_argument("--queries", type=str, default="", help="Optional path to a queries.txt (one per line)")
    parser.add_argument("--runs", type=int, default=30, help="Number of measured queries")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup queries (not timed)")
    parser.add_argument("--first-stage-k", type=int, default=8, help="Candidates per query from ANN")
    parser.add_argument("--final-k", type=int, default=5, help="Final passages to keep")
    parser.add_argument("--variants", type=int, default=0, help="Query rewrite variants (0 = off)")
    parser.add_argument("--use-ce", action="store_true", help="Enable cross-encoder reranker")
    parser.add_argument("--no-ingest", action="store_true", help="Skip ingestion step")
    parser.add_argument("--max-chars", type=int, default=700, help="Max chars per passage in context")
    args = parser.parse_args()

    docs_dir = Path(args.docs)
    if not args.no_ingest:
        t_ingest = time_ingestion(docs_dir)
        print_block("Ingestion", [
            f"Docs dir:        {docs_dir}",
            f"Ingestion time:  {t_ingest:.2f} s",
            f"Device:          {_device}",
        ])
    else:
        print_block("Ingestion", [
            f"Docs dir:        {docs_dir}",
            "Ingestion skipped (--no-ingest)",
            f"Device:          {_device}",
        ])

    queries = build_queries(Path(args.queries) if args.queries else None)

    cfg_fast = {
        "variants": args.variants,
        "first_stage_k": args.first_stage_k,
        "final_k": args.final_k,
        "max_chars": args.max_chars,
        "use_cross_encoder": args.use_ce,
    }

    print_block("Config", [
        f"Queries:         {len(queries)} in pool",
        f"Runs:            {args.runs}  (warmup={args.warmup})",
        f"First-stage k:   {cfg_fast['first_stage_k']}",
        f"Final k:         {cfg_fast['final_k']}",
        f"Query variants:  {cfg_fast['variants']}",
        f"Cross-encoder:   {cfg_fast['use_cross_encoder']}",
    ])

    summary, lats = bench_queries(queries, runs=args.runs, warmup=args.warmup, cfg=cfg_fast)

    lines = [
        f"mean:    {summary['mean_ms']:.1f} ms",
        f"median:  {summary['median_ms']:.1f} ms",
        f"p95:     {summary['p95_ms']:.1f} ms",
        f"p99:     {summary['p99_ms']:.1f} ms",
        f"min/max: {summary['min_ms']:.1f} / {summary['max_ms']:.1f} ms",
        f"QPS:     {summary['qps']:.2f}",
    ]
    if "gpu_mem_alloc_MB" in summary:
        lines += [
            f"GPU alloc:   {summary['gpu_mem_alloc_MB']:.1f} MB",
            f"GPU reserved:{summary['gpu_mem_reserved_MB']:.1f} MB",
        ]
    print_block("Latency & Throughput", lines)

    # Optional: write raw latencies to a CSV for plotting
    out_csv = Path("rag_speed_latencies.csv")
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("latency_seconds\n")
        for x in lats:
            f.write(f"{x:.6f}\n")
    print(f"\nSaved raw latencies to {out_csv.resolve()}")
    print("You can plot them (histogram/CDF) in your notebook or Streamlit app.")

if __name__ == "__main__":
    main()
