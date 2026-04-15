from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.rag.retrieve import load_index, top_k_search


def read_questions(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_prompt(question: str, retrieved: list[dict[str, object]]) -> str:
    context_blocks = []
    for item in retrieved:
        context_blocks.append(f"- Candidate answer: {item['answer']} (score={item['score']:.4f})")

    context = "\n".join(context_blocks)
    return (
        "You are a factual NBA QA system. Use only the context below to answer. "
        "If unsure, return the most likely short answer from context.\n\n"
        f"Question: {question}\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def generate_answers(
    questions: list[str],
    index_dir: Path,
    top_k: int,
    use_reader: bool,
    reader_model: str,
) -> tuple[list[str], list[dict[str, object]]]:
    embeddings, records, metadata = load_index(index_dir)
    retrieval_model = SentenceTransformer(str(metadata["model_name"]))

    generator = None
    if use_reader:
        from transformers import pipeline

        generator = pipeline("text2text-generation", model=reader_model)

    answers: list[str] = []
    logs: list[dict[str, object]] = []

    for question in questions:
        retrieved = top_k_search(question, retrieval_model, embeddings, records, top_k)

        if generator is None:
            answer = str(retrieved[0]["answer"]).split(";")[0].strip()
        else:
            prompt = build_prompt(question, retrieved)
            output = generator(prompt, max_new_tokens=32, do_sample=False)[0]["generated_text"]
            answer = output.strip().splitlines()[0]
            if not answer:
                answer = str(retrieved[0]["answer"]).split(";")[0].strip()

        answers.append(answer)
        logs.append(
            {
                "question": question,
                "predicted_answer": answer,
                "retrieved": retrieved,
            }
        )

    return answers, logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end RAG inference over a question file.")
    parser.add_argument("--questions-file", type=Path, required=True, help="Input questions file (one question per line).")
    parser.add_argument("--output-file", type=Path, required=True, help="Output answers file (one answer per line).")
    parser.add_argument(
        "--retrieval-log-file",
        type=Path,
        required=True,
        help="JSON file storing retrieved contexts and predictions.",
    )
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/retrieval_index"), help="Retrieval index directory.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval depth.")
    parser.add_argument(
        "--reader-model",
        type=str,
        default="google/flan-t5-base",
        help="Hugging Face reader model for answer generation.",
    )
    parser.add_argument(
        "--no-reader",
        action="store_true",
        help="Disable generator and use top retrieved answer as fallback reader.",
    )
    args = parser.parse_args()

    questions = read_questions(args.questions_file)
    answers, logs = generate_answers(
        questions=questions,
        index_dir=args.index_dir,
        top_k=args.top_k,
        use_reader=not args.no_reader,
        reader_model=args.reader_model,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.retrieval_log_file.parent.mkdir(parents=True, exist_ok=True)

    args.output_file.write_text("\n".join(answers) + "\n", encoding="utf-8")
    args.retrieval_log_file.write_text(json.dumps(logs, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Answered {len(answers)} questions.")
    print(f"Answers: {args.output_file}")
    print(f"Retrieval log: {args.retrieval_log_file}")


if __name__ == "__main__":
    main()
