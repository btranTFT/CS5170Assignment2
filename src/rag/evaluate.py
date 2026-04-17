from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def read_lines(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLES.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1

    overlap = 0
    for t in ref_tokens:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_recall(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not ref_tokens:
        return 1.0
    if not pred_tokens:
        return 0.0

    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1

    overlap = 0
    for t in ref_tokens:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1

    return overlap / len(ref_tokens)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def best_score(prediction: str, reference_line: str) -> tuple[float, float, float]:
    references = [r.strip() for r in reference_line.split(";") if r.strip()]
    if not references:
        references = [""]

    em = max(exact_match(prediction, ref) for ref in references)
    f1 = max(token_f1(prediction, ref) for ref in references)
    recall = max(answer_recall(prediction, ref) for ref in references)
    return em, f1, recall


def evaluate(predictions: list[str], references: list[str]) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError(f"Prediction/reference mismatch: {len(predictions)} != {len(references)}")

    em_scores: list[float] = []
    f1_scores: list[float] = []
    recall_scores: list[float] = []

    for pred, ref in zip(predictions, references):
        em, f1, recall = best_score(pred, ref)
        em_scores.append(em)
        f1_scores.append(f1)
        recall_scores.append(recall)

    n = len(predictions)
    return {
        "count": float(n),
        "answer_recall": sum(recall_scores) / n,
        "exact_match": sum(em_scores) / n,
        "token_f1": sum(f1_scores) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QA predictions with EM and token-level F1.")
    parser.add_argument("--predictions", type=Path, required=True, help="Predicted answers file.")
    parser.add_argument("--references", type=Path, required=True, help="Reference answers file.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save metrics JSON.")
    args = parser.parse_args()

    predictions = read_lines(args.predictions)
    references = read_lines(args.references)
    metrics = evaluate(predictions, references)

    print(json.dumps(metrics, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
