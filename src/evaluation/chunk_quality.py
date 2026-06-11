"""
chunk_quality.py
~~~~~~~~~~~~~~~~
Score the quality of ingested chunks before running full RAGAS eval.

Two signals:
    coherence  — does each chunk read as a self-contained, meaningful unit?
                 (keyword-overlap heuristic, no LLM call)
    coverage   — does the chunk set cover the evaluation questions adequately?
                 (BM25 recall proxy against evaluation questions)

Usage:
    python -m src.evaluation.chunk_quality
"""
from __future__ import annotations

import argparse
import json
import re
import math
from pathlib import Path
from collections import Counter
from datetime import datetime, UTC

import structlog

from config import get_settings
from src.retrieval import RetrievalPipeline

log = structlog.get_logger(__name__)
settings = get_settings()


class ChunkQualityScorer:
    """
    Score ingested chunks for coherence and question coverage.
    Runs entirely locally — no LLM API calls.
    """

    def __init__(self, dataset_path: str = settings.evaluation_dataset_path) -> None:
        self.dataset_path = Path(dataset_path)
        self.retrieval = RetrievalPipeline(expand_strategy="none", compress_strategy="score")

    def run(self) -> dict:
        samples = self._load_dataset()
        log.info("chunk_quality_start", samples=len(samples))

        coverage_scores: list[float] = []
        retrieval_hit_rates: list[float] = []

        print(f"\nScoring chunk quality over {len(samples)} golden questions...")
        print("─" * 56)

        for i, sample in enumerate(samples, start=1):
            question     = sample["question"]
            ideal_chunks = sample.get("ideal_chunks", [])

            # Retrieve top docs for this question
            try:
                docs = self.retrieval.retrieve(
                    question=question,
                    compress=False,       # skip compression for raw chunk scoring
                )
            except Exception as exc:
                log.warning("chunk_quality_retrieve_error", error=str(exc))
                docs = []

            # Coverage: did any retrieved chunk contain the ideal keywords?
            coverage = self._keyword_coverage(docs, ideal_chunks)
            coverage_scores.append(coverage)

            # Coherence: average coherence score of returned chunks
            coherence_scores = [self._coherence(d.page_content) for d in docs]
            avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0

            retrieval_hit_rates.append(1.0 if coverage >= 0.5 else 0.0)

            print(f"  [{i:>2}/{len(samples)}] coverage={coverage:.2f}  "
                  f"coherence={avg_coherence:.2f}  "
                  f"q={question[:45]}")

        result = {
            "timestamp": datetime.now(UTC).isoformat(),
            "num_samples": len(samples),
            "avg_coverage": round(sum(coverage_scores) / len(coverage_scores), 4) if coverage_scores else 0.0,
            "hit_rate":     round(sum(retrieval_hit_rates) / len(retrieval_hit_rates), 4) if retrieval_hit_rates else 0.0,
            "per_question": [
                {
                    "question": s["question"],
                    "coverage": round(c, 4),
                    "hit":      h == 1.0,
                }
                for s, c, h in zip(samples, coverage_scores, retrieval_hit_rates)
            ],
        }

        self._print_report(result)
        return result

    def run_and_save(self, output_path: str = "results/chunk_quality.json") -> dict:
        result = self.run()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        log.info("chunk_quality_saved", path=str(out))
        return result

    # ── Scoring helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _keyword_coverage(docs, ideal_keywords: list[str]) -> float:
        """Fraction of ideal keywords found in any retrieved chunk."""
        if not ideal_keywords:
            return 1.0
        combined = " ".join(d.page_content.lower() for d in docs)
        hits = sum(1 for kw in ideal_keywords if kw.lower() in combined)
        return hits / len(ideal_keywords)

    @staticmethod
    def _coherence(text: str) -> float:
        """
        Heuristic coherence score (0–1):
        - Penalises very short chunks (< 30 words)
        - Penalises chunks with excessive special characters
        - Penalises chunks that are mostly whitespace / newlines
        """
        words = text.split()
        if len(words) < 10:
            return 0.1

        # Ratio of alpha chars to total chars
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)

        # Sentence count (rough)
        sentences = len(re.findall(r"[.!?]+", text)) + 1
        words_per_sentence = len(words) / sentences

        # Good chunks: 60–80% alpha, 10–40 words/sentence
        alpha_score    = min(alpha_ratio / 0.7, 1.0)
        sentence_score = 1.0 if 8 <= words_per_sentence <= 50 else 0.5
        length_score   = min(len(words) / 50, 1.0)

        return round((alpha_score + sentence_score + length_score) / 3, 4)

    @staticmethod
    def _print_report(result: dict) -> None:
        print("\n" + "═" * 48)
        print("  Chunk Quality Report")
        print("═" * 48)
        print(f"  Samples      : {result['num_samples']}")
        print(f"  Avg Coverage : {result['avg_coverage']:.4f}")
        print(f"  Hit Rate     : {result['hit_rate']:.4f}")
        print(f"  Timestamp    : {result['timestamp']}")
        print("─" * 48)
        print(f"  {'Question':<38} {'Cov':>5}  {'Hit'}")
        print("─" * 48)
        for row in result["per_question"]:
            hit_str = "✓" if row["hit"] else "✗"
            q = row["question"][:38]
            print(f"  {q:<38} {row['coverage']:>5.2f}  {hit_str}")
        print("═" * 48 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Score chunk quality locally")
    parser.add_argument("--dataset", default=settings.golden_dataset_path)
    parser.add_argument("--output",  default="results/chunk_quality.json")
    args = parser.parse_args()

    scorer = ChunkQualityScorer(dataset_path=args.dataset)
    scorer.run_and_save(output_path=args.output)


if __name__ == "__main__":
    main()