from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def load_index(index_dir: Path) -> tuple[np.ndarray, list[dict[str, str]], dict[str, object]]:
    embeddings = np.load(index_dir / "embeddings.npy")
    records = json.loads((index_dir / "records.json").read_text(encoding="utf-8"))
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    return embeddings, records, metadata


def top_k_search(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    records: list[dict[str, str]],
    k: int,
) -> list[dict[str, object]]:
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = embeddings @ query_vec

    k = min(k, len(records))
    indices = np.argpartition(-scores, k - 1)[:k]
    ranked = sorted(indices, key=lambda i: float(scores[i]), reverse=True)

    results: list[dict[str, object]] = []
    for idx in ranked:
        record = records[int(idx)]
        results.append(
            {
                "id": record["id"],
                "split": record["split"],
                "score": float(scores[idx]),
                "question": record["question"],
                "answer": record["answer"],
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-k QA passages for a query.")
    parser.add_argument("query", type=str, help="User question to retrieve against.")
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/retrieval_index"), help="Index directory.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    args = parser.parse_args()

    embeddings, records, metadata = load_index(args.index_dir)
    model = SentenceTransformer(str(metadata["model_name"]))

    results = top_k_search(args.query, model, embeddings, records, args.top_k)

    print(json.dumps({"query": args.query, "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
