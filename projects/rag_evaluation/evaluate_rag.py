"""Evaluate this RAG app with a small, inspectable golden dataset.

The app model is GPT-4o-mini. TruLens uses GPT-5 as an LLM judge for
faithfulness: is the generated answer supported by the retrieved context?
Precision and recall are intentionally implemented here with simple token
overlap so students can see exactly how those scores are calculated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

EVALUATION_DIR = ROOT / "projects" / "rag_evaluation"
DATASET_PATH = EVALUATION_DIR / "golden_dataset.json"
RESULTS_PATH = EVALUATION_DIR / "results.json"


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """Load and validate the golden Q&A examples before making API calls."""
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("The golden dataset must be a non-empty JSON list.")

    required = {"id", "question", "reference_answer"}
    for row in dataset:
        if not required.issubset(row):
            missing = ", ".join(sorted(required - row.keys()))
            raise ValueError(f"Dataset row is missing: {missing}")
    return dataset


def tokens(text: str) -> set[str]:
    """Normalize words for a deliberately easy-to-understand overlap metric."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def precision_recall(answer: str, reference: str) -> tuple[float, float]:
    """Compare answer content with the reference answer using set overlap."""
    answer_tokens = tokens(answer)
    reference_tokens = tokens(reference)
    if not answer_tokens or not reference_tokens:
        return 0.0, 0.0

    overlap = answer_tokens & reference_tokens
    precision = len(overlap) / len(answer_tokens)
    recall = len(overlap) / len(reference_tokens)
    return precision, recall


def context_from_trace(retrieval: dict[str, Any]) -> str:
    """Rebuild the same evidence shown in the Streamlit retrieval expander."""
    vector_context = [
        item["document"]
        for item in retrieval.get("vector", [])
        if item.get("document")
    ]
    graph_context = [
        f"{fact['subject']} {fact['predicate']} {fact['object']}"
        for fact in retrieval.get("graph", [])
    ]
    return "\n".join(vector_context + graph_context)


def faithfulness_score(judge: Any, context: str, answer: str) -> tuple[float, str]:
    """Ask TruLens' GPT-5 provider to score evidence support from 0 to 1."""
    result = judge.groundedness_measure_with_cot_reasons(
        source=context,
        statement=answer,
    )
    if isinstance(result, tuple):
        score, reason = result
    else:
        score, reason = result, ""
    return float(score), str(reason)


def run_evaluation(limit: int | None = None) -> list[dict[str, Any]]:
    from projects.rag_chat.rag import call_llm_with_query
    from trulens.providers.openai import OpenAI as TruLensOpenAI

    load_dotenv(ROOT / ".env")
    dataset = load_dataset()[:limit]
    judge_model = "gpt-5"
    judge = TruLensOpenAI(model_engine=judge_model)
    results = []

    for row in dataset:
        response = call_llm_with_query(row["question"])
        answer = response["answer"]
        retrieval = response["retrieval"]
        context = context_from_trace(retrieval)
        precision, recall = precision_recall(answer, row["reference_answer"])
        faithfulness, reason = faithfulness_score(judge, context, answer)
        results.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": answer,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "faithfulness": round(faithfulness, 4),
                "faithfulness_reason": reason,
            }
        )

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def print_results(results: list[dict[str, Any]]) -> None:
    print("\nRAG evaluation results (0 = poor, 1 = strong)")
    print("-" * 72)
    print(f"{'ID':<16} {'Precision':>10} {'Recall':>10} {'Faithfulness':>14}")
    print("-" * 72)
    for row in results:
        print(
            f"{row['id']:<16} {row['precision']:>10.2f} "
            f"{row['recall']:>10.2f} {row['faithfulness']:>14.2f}"
        )
    averages = {
        metric: sum(row[metric] for row in results) / len(results)
        for metric in ("precision", "recall", "faithfulness")
    }
    print("-" * 72)
    print(
        f"{'AVERAGE':<16} {averages['precision']:>10.2f} "
        f"{averages['recall']:>10.2f} {averages['faithfulness']:>14.2f}"
    )
    print(f"\nDetailed results saved to {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N golden examples while learning.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check the golden dataset without calling the RAG app or judge.",
    )
    args = parser.parse_args()
    dataset = load_dataset()
    if args.validate_only:
        print(f"Dataset is valid: {len(dataset)} examples.")
        return
    print_results(run_evaluation(args.limit))


if __name__ == "__main__":
    main()
