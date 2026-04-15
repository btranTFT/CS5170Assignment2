from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_qa_pairs(data_dir: Path, include_test: bool) -> list[dict[str, str]]:
    splits = ["train"] if not include_test else ["train", "test"]
    records: list[dict[str, str]] = []

    for split in splits:
        q_path = data_dir / split / "questions.txt"
        a_path = data_dir / split / "reference_answers.txt"

        questions = read_lines(q_path)
        answers = read_lines(a_path)
        if len(questions) != len(answers):
            raise ValueError(f"Line count mismatch in split '{split}': {len(questions)} != {len(answers)}")

        for i, (question, answer) in enumerate(zip(questions, answers), start=1):
            records.append(
                {
                    "id": f"{split}-{i}",
                    "split": split,
                    "question": question,
                    "answer": answer,
                    "passage": f"Question: {question}\nAnswer: {answer}",
                }
            )

    return records


def save_index(index_dir: Path, records: list[dict[str, str]], embeddings: np.ndarray, model_name: str) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    np.save(index_dir / "embeddings.npy", embeddings)

    metadata = {
        "model_name": model_name,
        "num_records": len(records),
        "dimension": int(embeddings.shape[1]),
    }
    (index_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (index_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")



def main() -> None:
    parser = argparse.ArgumentParser(description="Build dense retrieval index for QA records.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing train/test QA files.")
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/retrieval_index"), help="Output index directory.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Include test split in the retrieval index (disabled by default).",
    )
    args = parser.parse_args()

    records = load_qa_pairs(args.data_dir, include_test=args.include_test)
    model = SentenceTransformer(args.model_name)
    passages = [r["passage"] for r in records]
    embeddings = model.encode(passages, convert_to_numpy=True, normalize_embeddings=True)

    save_index(args.index_dir, records, embeddings, args.model_name)

    print("Index build complete.")
    print(f"Records: {len(records)}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"Saved to: {args.index_dir}")


if __name__ == "__main__":
    main()
