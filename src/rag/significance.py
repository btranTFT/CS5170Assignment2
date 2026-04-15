from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.rag.evaluate import best_score, read_lines


def paired_scores(preds: list[str], refs: list[str]) -> tuple[list[float], list[float]]:
    em_scores: list[float] = []
    f1_scores: list[float] = []
    for pred, ref in zip(preds, refs):
        em, f1 = best_score(pred, ref)
        em_scores.append(em)
        f1_scores.append(f1)
    return em_scores, f1_scores


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def paired_bootstrap_pvalue(
    a_scores: list[float], b_scores: list[float], n_samples: int = 10000, seed: int = 42
) -> dict[str, float]:
    if len(a_scores) != len(b_scores):
        raise ValueError("Score lists must have equal length for paired testing.")

    rng = random.Random(seed)
    n = len(a_scores)
    observed = mean(a_scores) - mean(b_scores)

    count = 0
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        delta = mean([a_scores[i] for i in idx]) - mean([b_scores[i] for i in idx])
        if abs(delta) >= abs(observed):
            count += 1

    pvalue = (count + 1) / (n_samples + 1)
    return {"observed_delta": observed, "p_value": pvalue}


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap significance test for two QA systems.")
    parser.add_argument("--pred-a", type=Path, required=True, help="Prediction file for system A.")
    parser.add_argument("--pred-b", type=Path, required=True, help="Prediction file for system B.")
    parser.add_argument("--references", type=Path, required=True, help="Reference answers file.")
    parser.add_argument("--samples", type=int, default=10000, help="Bootstrap samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    pred_a = read_lines(args.pred_a)
    pred_b = read_lines(args.pred_b)
    refs = read_lines(args.references)

    if not (len(pred_a) == len(pred_b) == len(refs)):
        raise ValueError(
            f"Line count mismatch: A={len(pred_a)} B={len(pred_b)} refs={len(refs)}"
        )

    em_a, f1_a = paired_scores(pred_a, refs)
    em_b, f1_b = paired_scores(pred_b, refs)

    results = {
        "exact_match": paired_bootstrap_pvalue(em_a, em_b, n_samples=args.samples, seed=args.seed),
        "token_f1": paired_bootstrap_pvalue(f1_a, f1_b, n_samples=args.samples, seed=args.seed),
    }

    print(json.dumps(results, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
