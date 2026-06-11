"""
latency_bench.py
~~~~~~~~~~~~~~~~
Measure p50 / p95 / p99 query latency locally.

Usage:
    python -m src.evaluation.latency_bench
    python -m src.evaluation.latency_bench --runs 50 --output results/latency.json
"""
from __future__ import annotations

import argparse
import json
import time
import statistics
from datetime import datetime, UTC
from pathlib import Path

import structlog

from config import get_settings
from src.generation.Rag_chain import RAGChain

log = structlog.get_logger(__name__)
settings = get_settings()

# Default questions used if no dataset provided
_DEFAULT_QUESTIONS = [
    "What is the main contribution of the Attention Is All You Need paper?",
    "What is the purpose of positional encoding in transformers?",
    "Why is hybrid search better than dense-only search?",
    "What are diffusion probabilistic models inspired by?",
    "What is a diffusion probabilistic model?",
]


class LatencyBench:
    """
    Runs N queries and reports p50, p95, p99 latency breakdowns for:
        - retrieval only
        - generation only
        - end-to-end (retrieval + generation)
    """

    def __init__(self) -> None:
        self.chain = RAGChain()

    def run(self, questions: list[str], runs: int = 20) -> dict:
        # Repeat questions to fill `runs` slots
        repeated = (questions * ((runs // len(questions)) + 1))[:runs]

        retrieval_times: list[float] = []
        generation_times: list[float] = []
        e2e_times: list[float] = []
        errors = 0

        print(f"\nRunning {runs} queries for latency benchmark...")
        print("─" * 40)

        for i, question in enumerate(repeated, start=1):
            try:
                # ── End-to-end ────────────────────────────────────────────────
                t_start = time.perf_counter()

                # ── Retrieval phase ───────────────────────────────────────────
                t_ret_start = time.perf_counter()
                docs = self.chain.retrieval.retrieve(question=question)
                t_ret_end = time.perf_counter()

                # ── Generation phase ──────────────────────────────────────────
                t_gen_start = time.perf_counter()
                self.chain._generate(question, docs, chat_history=None)
                t_gen_end = time.perf_counter()

                t_end = time.perf_counter()

                ret_ms = (t_ret_end - t_ret_start) * 1000
                gen_ms = (t_gen_end - t_gen_start) * 1000
                e2e_ms = (t_end - t_start) * 1000

                retrieval_times.append(ret_ms)
                generation_times.append(gen_ms)
                e2e_times.append(e2e_ms)

                print(f"  [{i:>3}/{runs}] e2e={e2e_ms:>7.1f}ms  "
                      f"ret={ret_ms:>7.1f}ms  gen={gen_ms:>7.1f}ms")

            except Exception as exc:
                log.warning("bench_query_error", question=question[:50], error=str(exc))
                errors += 1

        result = {
            "runs": runs,
            "errors": errors,
            "timestamp": datetime.now(UTC).isoformat(),
            "retrieval_ms":  self._percentiles(retrieval_times),
            "generation_ms": self._percentiles(generation_times),
            "e2e_ms":        self._percentiles(e2e_times),
        }

        self._print_report(result)
        return result

    def run_and_save(
        self,
        questions: list[str] | None = None,
        runs: int = 20,
        output_path: str = "results/latency.json",
    ) -> dict:
        qs = questions or _DEFAULT_QUESTIONS
        result = self.run(qs, runs=runs)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        log.info("latency_results_saved", path=str(out))
        return result

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _percentiles(times: list[float]) -> dict:
        if not times:
            return {}
        s = sorted(times)
        n = len(s)
        return {
            "p50":  round(statistics.median(s), 1),
            "p95":  round(s[int(n * 0.95)], 1),
            "p99":  round(s[int(n * 0.99)], 1),
            "mean": round(statistics.mean(s), 1),
            "min":  round(s[0], 1),
            "max":  round(s[-1], 1),
        }

    @staticmethod
    def _print_report(result: dict) -> None:
        print("\n" + "═" * 54)
        print("  Latency Benchmark Report")
        print("═" * 54)
        print(f"  Runs    : {result['runs']}   Errors: {result['errors']}")
        print(f"  Time    : {result['timestamp']}")
        print("─" * 54)
        print(f"  {'Phase':<16} {'p50':>8} {'p95':>8} {'p99':>8} {'mean':>8}")
        print("─" * 54)
        for phase in ("retrieval_ms", "generation_ms", "e2e_ms"):
            p = result[phase]
            label = phase.replace("_ms", "")
            print(f"  {label:<16} {p['p50']:>7.1f}ms {p['p95']:>7.1f}ms "
                  f"{p['p99']:>7.1f}ms {p['mean']:>7.1f}ms")
        print("═" * 54 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Latency benchmark for RAG pipeline")
    parser.add_argument("--runs",   type=int, default=20,
                        help="Number of queries to run (default: 20)")
    parser.add_argument("--dataset", default=None,
                        help="Optional path to evaluation dataset JSON to source questions from")
    parser.add_argument("--output", default="results/latency.json",
                        help="Where to save results (default: %(default)s)")
    args = parser.parse_args()

    questions = None
    if args.dataset:
        data = json.loads(Path(args.dataset).read_text())
        questions = [d["question"] for d in data]

    bench = LatencyBench()
    bench.run_and_save(questions=questions, runs=args.runs, output_path=args.output)


if __name__ == "__main__":
    main()