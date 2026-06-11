"""
Run RAGAS evaluation locally against your evaluation dataset.

Usage:
    python -m src.evaluation.ragas_runner
"""
from __future__ import annotations
 
import argparse
import asyncio
import os
import json
import time
from pathlib import Path
from datetime import datetime, UTC
import torch
from pydantic import BaseModel
 
from ragas import EvaluationDataset, SingleTurnSample, experiment
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

import litellm
import structlog
 
from config import get_settings
from src.generation.Rag_chain import RAGChain
 
log = structlog.get_logger(__name__)
settings = get_settings()

# Experiment result schema
class RAGEvalResult(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    context_precision: float

class RagasRunner:
    """
    Runs RAGAS evaluation on a evaluation dataset.
 
    Metrics computed:
        faithfulness       — are claims in the answer supported by context?
        answer_relevancy   — does the answer address the question?
        context_recall     — does retrieved context cover the ground truth?
        context_precision  — is the retrieved context precise (not noisy)?
    """

    def __init__(self, dataset_path: str = settings.evaluation_dataset_path) -> None:
        self.dataset_path = Path(dataset_path)
        self.chain = RAGChain()
        # RAGAS uses its own LLM/embeddings instances for metric computation
        os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key.get_secret_value())

        self._ragas_llm = llm_factory(
            f"groq/{settings.llm_model}",
            provider= "litellm",
            client= litellm.acompletion,
        )

        self._ragas_embeddings = HuggingFaceEmbeddings(
            model = settings.embedding_model,
            device = "cuda" if torch.cuda.is_available() else "cpu",  
            normalize_embeddings = True,
            batch_size = 32,
        )
        self._faithfulness = Faithfulness(llm = self._ragas_llm)
        self._answer_relevancy = AnswerRelevancy(
            llm= self._ragas_llm,
            embeddings= self._ragas_embeddings
        )
        self._context_recall = ContextRecall(llm=self._ragas_llm)
        self._context_precision = ContextPrecision(llm = self._ragas_llm)

    def run(self) -> dict:
        """
        Run evaluation and return a result dict with per-metric scores.
        """
        return asyncio.run(self._run_async())
    
    def run_and_save(self, output_path: str = "results/ragas_result.json") -> dict:
        """Run evaluation and persist results to *output_path*."""
        result = self.run()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        log.info("ragas_results_saved", path=str(out))
        return result
    
    async def _run_async(self) -> dict:
        raw_samples = self._load_dataset()
        log.info("ragas_run_start", samples= len(raw_samples))

        # Run the RAG chain on every sample to collect responses + contexts.
        ragas_samples = await self._collect_samples(raw_samples)

        row_results: list[RAGEvalResult] = []
        t0 = time.perf_counter()

        for i, row in enumerate(ragas_samples, start=1):
            log.info("ragas_scoring", index=i, total=len(ragas_samples))
            try:
                faith = await self._faithfulness.ascore(
                    user_input=row.user_input,
                    response=row.response,
                    retrieved_contexts=row.retrieved_contexts,
                )
                relevancy = await self._answer_relevancy.ascore(
                    user_input=row.user_input,
                    response=row.response,
                )
                recall = await self._context_recall.ascore(
                    user_input=row.user_input,
                    response=row.response,
                    retrieved_contexts=row.retrieved_contexts,
                    reference=row.reference,
                )
                precision = await self._context_precision.ascore(
                    user_input=row.user_input,
                    response=row.response,
                    retrieved_contexts=row.retrieved_contexts,
                    reference=row.reference,
                )
                row_results.append(RAGEvalResult(
                    faithfulness=float(faith.value),
                    answer_relevancy=float(relevancy.value),
                    context_recall=float(recall.value),
                    context_precision=float(precision.value),
                ))
            except Exception as exc:
                log.warning("ragas_scoring_error", index=i, error=str(exc))
 
        elapsed = round(time.perf_counter() - t0, 3)
        scores = self._aggregate_scores(row_results)              

        summary ={
            "scores": scores,
            "num_samples": len(raw_samples),
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now(UTC).isoformat(),
            "llm_model": settings.llm_model,
            "thresholds": {
                "faithfulness": settings.eval_faithfulness_threshold,
                "answer_relevancy": settings.eval_answer_relevancy_threshold,
                "context_recall": settings.eval_context_recall_threshold,
            },
            "passed": self._check_thresholds(scores),
        }
        self._print_report(summary)
        return summary
    
    async def _collect_samples(self, raw_samples: list[dict]) -> list[SingleTurnSample]:
        """
        Run the RAG chain on each raw sample and return a list of SingleTurnSamples.
        """
        results: list[SingleTurnSample] = []
        for i, sample in enumerate(raw_samples, start=1):
            question: str = sample["user_input"]
            ground_truth: str = sample["reference"]
 
            log.info("ragas_chain_query", index=i, total=len(raw_samples),
                     question=question[:60])
            try:
                response = self.chain.query(question=question)
                contexts = [src.content for src in response.sources]
                answer = response.answer
            except Exception as exc:
                log.warning("ragas_chain_error", question=question[:60], error=str(exc))
                contexts = []
                answer = ""
 
            results.append(SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            ))

            await asyncio.sleep(6.5)
        return results
    

    @staticmethod
    def _aggregate_scores(exp_results) -> dict:
        """Average per-metric scores across all experiment rows."""
        rows = list(exp_results)  # experiment() returns an iterable of RAGEvalResult
        n = len(rows)
        if n == 0:
            return {k: 0.0 for k in RAGEvalResult.model_fields}
 
        totals: dict[str, float] = {k: 0.0 for k in RAGEvalResult.model_fields}
        for row in rows:
            for field in RAGEvalResult.model_fields:
                totals[field] += getattr(row, field)
 
        return {k: round(v / n, 4) for k, v in totals.items()}


    def _load_dataset(self) -> list[dict]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found at {self.dataset_path}. "
                "Add Q&A pairs to evaluation/evaluation_dataset.json"
            )
        return json.loads(self.dataset_path.read_text(encoding="utf-8-sig"))

        
    def _check_thresholds(self, scores: dict) -> dict:
        """Return pass/fail per metric against configured thresholds."""
        return {
                "faithfulness":     scores["faithfulness"]     >= settings.eval_faithfulness_threshold,
                "answer_relevancy": scores["answer_relevancy"] >= settings.eval_answer_relevancy_threshold,
                "context_recall":   scores["context_recall"]   >= settings.eval_context_recall_threshold,
            }
        
    @staticmethod
    def _print_report(summary: dict) -> None:
        scores = summary["scores"]
        passed = summary["passed"]
        thresholds = summary["thresholds"]

        print("\n" + "═" * 52)
        print("  RAGAS Evaluation Report")
        print("═" * 52)
        print(f"  Samples   : {summary['num_samples']}")
        print(f"  Elapsed   : {summary['elapsed_seconds']}s")
        print(f"  Timestamp : {summary['timestamp']}")
        print("─" * 52)
        print(f"  {'Metric':<24} {'Score':>7}  {'Threshold':>9}  {'Status'}")
        print("─" * 52)

        for metric, score in scores.items():
            threshold = thresholds.get(metric, "—")
            status    = "✓ PASS" if passed.get(metric, True) else "✗ FAIL"
            thresh_str = f"{threshold:.2f}" if isinstance(threshold, float) else str(threshold)
            print(f"  {metric:<24} {score:>7.4f}  {thresh_str:>9}  {status}")
 
        print("═" * 52 + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description= "Run RAGAS evalustion locally")
    parser.add_argument(
        "--dataset",
        default= settings.evaluation_dataset_path,
        help="Path to evaluation dataset JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="results/ragas_latest.json",
        help="Where to save results JSON (default: %(default)s)",
    )
    args = parser.parse_args()

    runner = RagasRunner(dataset_path= args.dataset)
    runner.run_and_save(output_path= args.output)

if __name__ == "__main__":
    main()
