import asyncio
import inspect
import json
import os
import re
import time
from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge
from engine.vector_store import build_index, build_index_v1
from agent.main_agent import MainAgent

class ExpertEvaluator:
    def __init__(self):
        self.retrieval_evaluator = RetrievalEvaluator()
        self.ragas_available = False
        self.ragas_reason = ""
        self.sample_cls = None
        self.faithfulness_metric = None
        self.answer_relevancy_metric = None
        self._setup_ragas()

    def _setup_ragas(self) -> None:
        try:
            try:
                from ragas import SingleTurnSample as RagasSingleTurnSample
            except ImportError:
                from ragas.dataset_schema import SingleTurnSample as RagasSingleTurnSample

            try:
                from ragas.metrics.collections import Faithfulness as RagasFaithfulness
                from ragas.metrics.collections import AnswerRelevancy as RagasAnswerRelevancy
            except ImportError:
                from ragas.metrics import Faithfulness as RagasFaithfulness
                from ragas.metrics import ResponseRelevancy as RagasAnswerRelevancy

            try:
                from ragas.embeddings.base import embedding_factory
            except ImportError:
                from ragas.embeddings import embedding_factory

            from ragas.llms import llm_factory
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            llm = llm_factory("gpt-4o-mini", client=client)
            embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

            self.sample_cls = RagasSingleTurnSample
            self.faithfulness_metric = RagasFaithfulness(llm=llm)
            self.answer_relevancy_metric = RagasAnswerRelevancy(llm=llm, embeddings=embeddings)
            self.ragas_available = True
            self.ragas_reason = "ragas"
        except Exception as exc:
            self.ragas_available = False
            self.ragas_reason = f"fallback:{exc.__class__.__name__}"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)
        return [token for token in tokens if len(token) > 2]

    @staticmethod
    def _jaccard_score(left: str, right: str) -> float:
        left_tokens = set(ExpertEvaluator._tokenize(left))
        right_tokens = set(ExpertEvaluator._tokenize(right))
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = left_tokens & right_tokens
        union = left_tokens | right_tokens
        return len(overlap) / len(union)

    def _faithfulness_score(self, answer: str, contexts: list[str]) -> float:
        if not answer or not contexts:
            return 0.0

        answer_sentences = [segment.strip() for segment in re.split(r"[.!?]+", answer) if segment.strip()]
        if not answer_sentences:
            return 0.0

        supported = 0
        for sentence in answer_sentences:
            sentence_score = max(self._jaccard_score(sentence, context) for context in contexts)
            if sentence_score >= 0.18:
                supported += 1
        return round(supported / len(answer_sentences), 4)

    def _relevancy_score(self, question: str, answer: str) -> float:
        if not question or not answer:
            return 0.0
        base = self._jaccard_score(question, answer)
        answer_tokens = self._tokenize(answer)
        question_tokens = set(self._tokenize(question))
        if not answer_tokens:
            return 0.0
        coverage = sum(1 for token in question_tokens if token in answer_tokens) / max(len(question_tokens), 1)
        return round(min(1.0, (base * 0.6) + (coverage * 0.4)), 4)

    @staticmethod
    def _coerce_ragas_value(score) -> float:
        if score is None:
            return 0.0
        if hasattr(score, "score"):
            score = score.score
        if hasattr(score, "value"):
            score = score.value
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0

    async def _call_ragas_metric(self, metric, question: str, answer: str, contexts: list[str]):
        if hasattr(metric, "ascore"):
            try:
                params = inspect.signature(metric.ascore).parameters
                kwargs = {}
                if "user_input" in params:
                    kwargs["user_input"] = question
                if "response" in params:
                    kwargs["response"] = answer
                if "retrieved_contexts" in params:
                    kwargs["retrieved_contexts"] = contexts
                return await metric.ascore(**kwargs)
            except (TypeError, ValueError):
                # Older ragas variants may expose ascore with a sample object instead.
                pass

        if hasattr(metric, "single_turn_ascore"):
            sample = self.sample_cls(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )
            return await metric.single_turn_ascore(sample)

        raise AttributeError(f"Unsupported ragas metric API: {type(metric).__name__}")

    async def _score_with_ragas(self, question: str, answer: str, contexts: list[str]) -> dict:
        faithfulness_result = await self._call_ragas_metric(
            self.faithfulness_metric,
            question,
            answer,
            contexts,
        )
        relevancy_result = await self._call_ragas_metric(
            self.answer_relevancy_metric,
            question,
            answer,
            contexts,
        )

        faithfulness = round(self._coerce_ragas_value(faithfulness_result), 4)
        relevancy = round(self._coerce_ragas_value(relevancy_result), 4)
        return {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "backend": "ragas",
        }

    async def score(self, case, resp):
        answer = resp.get("answer", "")
        contexts = resp.get("contexts") or []
        if not isinstance(contexts, list):
            contexts = [str(contexts)]

        metadata = resp.get("metadata") or {}
        retrieved_ids = (
            metadata.get("retrieved_ids")
            or resp.get("retrieved_ids")
            or case.get("retrieved_ids")
            or []
        )
        expected_ids = case.get("expected_retrieval_ids") or []

        hit_rate = self.retrieval_evaluator.calculate_hit_rate(expected_ids, retrieved_ids) if expected_ids else 0.0
        mrr = self.retrieval_evaluator.calculate_mrr(expected_ids, retrieved_ids) if expected_ids else 0.0

        if self.ragas_available:
            try:
                ragas_scores = await self._score_with_ragas(case.get("question", ""), answer, contexts)
                faithfulness = ragas_scores["faithfulness"]
                relevancy = ragas_scores["relevancy"]
                backend = ragas_scores["backend"]
            except Exception:
                faithfulness = self._faithfulness_score(answer, contexts)
                relevancy = self._relevancy_score(case.get("question", ""), answer)
                backend = "fallback"
        else:
            faithfulness = self._faithfulness_score(answer, contexts)
            relevancy = self._relevancy_score(case.get("question", ""), answer)
            backend = "fallback"

        combined_score = round((faithfulness * 0.45) + (relevancy * 0.35) + (hit_rate * 0.2), 4)

        return {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "combined_score": combined_score,
            "backend": backend,
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": mrr,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
            },
        }

def build_regression_gate(v1_summary: dict, v2_summary: dict) -> dict:
    v1_metrics = v1_summary.get("metrics", {})
    v2_metrics = v2_summary.get("metrics", {})

    deltas = {
        "avg_score": round(v2_metrics.get("avg_score", 0.0) - v1_metrics.get("avg_score", 0.0), 4),
        "hit_rate": round(v2_metrics.get("hit_rate", 0.0) - v1_metrics.get("hit_rate", 0.0), 4),
        "agreement_rate": round(v2_metrics.get("agreement_rate", 0.0) - v1_metrics.get("agreement_rate", 0.0), 4),
        "faithfulness": round(v2_metrics.get("faithfulness", 0.0) - v1_metrics.get("faithfulness", 0.0), 4),
        "relevancy": round(v2_metrics.get("relevancy", 0.0) - v1_metrics.get("relevancy", 0.0), 4),
    }

    improved_metrics = [name for name, delta in deltas.items() if delta > 0]
    regressed_metrics = [name for name, delta in deltas.items() if delta < 0]
    unchanged_metrics = [name for name, delta in deltas.items() if delta == 0]

    if deltas["avg_score"] < 0:
        decision = "ROLLBACK"
        reason = "Bản V2 kém hơn V1 theo điểm tổng hợp."
    elif deltas["avg_score"] > 0:
        decision = "APPROVE"
        if regressed_metrics:
            reason = f"Bản V2 cải thiện điểm tổng hợp nhưng vẫn có chỉ số giảm: {', '.join(regressed_metrics)}."
        else:
            reason = "Bản V2 tốt hơn V1 trên benchmark hiện tại."
    elif regressed_metrics:
        decision = "BLOCK RELEASE"
        reason = "Bản V2 không cải thiện được điểm tổng hợp và còn có chỉ số kém hơn V1."
    else:
        decision = "BLOCK RELEASE"
        reason = "Điểm tổng hợp của V2 không tốt hơn V1."

    return {
        "decision": decision,
        "reason": reason,
        "baseline_version": v1_summary.get("metadata", {}).get("version"),
        "candidate_version": v2_summary.get("metadata", {}).get("version"),
        "delta": deltas,
        "improved_metrics": improved_metrics,
        "regressed_metrics": regressed_metrics,
        "unchanged_metrics": unchanged_metrics,
    }

async def run_benchmark_with_results(agent_variant: str, report_version: str):
    print(f"🚀 Khởi động Benchmark cho {report_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("[ERR] Thieu data/golden_set.jsonl. Hay chay 'python data/synthetic_gen.py' truoc.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("[ERR] File data/golden_set.jsonl rong. Hay tao it nhat 1 test case.")
        return None, None

    runner = BenchmarkRunner(
        MainAgent(version=agent_variant),
        ExpertEvaluator(),
        LLMJudge(),
    )
    results = await runner.run_all(dataset)
    performance_summary = runner.summarize_results(results)

    total = len(results)
    avg_score = sum(r["judge"]["final_score"] for r in results) / total
    avg_hit_rate = sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total
    avg_mrr = sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total
    avg_agreement = sum(r["judge"]["agreement_rate"] for r in results) / total
    avg_faithfulness = sum(r["ragas"]["faithfulness"] for r in results) / total
    avg_relevancy = sum(r["ragas"]["relevancy"] for r in results) / total
    total_tokens_used = sum(r.get("tokens_used", 0) for r in results)
    total_generation_cost = sum(r.get("generation_cost_usd", 0.0) for r in results)
    total_judge_cost = sum(r.get("judge_cost_usd", 0.0) for r in results)
    total_cost_estimate = sum(r.get("total_cost_usd", 0.0) for r in results)
    avg_latency = performance_summary["avg_latency"]

    summary = {
        "metadata": {
            "version": report_version,
            "agent_variant": agent_variant,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluator_backend": getattr(runner.evaluator, "ragas_reason", "unknown"),
        },
        "metrics": {
            "avg_score": avg_score,
            "hit_rate": avg_hit_rate,
            "mrr": avg_mrr,
            "agreement_rate": avg_agreement,
            "faithfulness": avg_faithfulness,
            "relevancy": avg_relevancy,
            "avg_latency": avg_latency,
            "pass_rate": performance_summary["pass_rate"],
            "total_tokens_used": total_tokens_used,
            "total_generation_cost_usd": total_generation_cost,
            "total_judge_cost_usd": total_judge_cost,
            "total_cost_estimate": total_cost_estimate,
        }
    }
    return results, summary


async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version, version)
    return summary


async def main():
    build_index_v1()
    build_index()

    _, v1_summary = await run_benchmark_with_results("v1", "Agent_V1_Base")
    v2_results, v2_summary = await run_benchmark_with_results("v2", "Agent_V2_Optimized")
    
    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    regression = build_regression_gate(v1_summary, v2_summary)
    delta = regression["delta"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")
    print(f"Decision: {regression['decision']}")
    print(f"Reason: {regression['reason']}")

    os.makedirs("reports", exist_ok=True)
    v2_summary["regression"] = regression
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    if regression["decision"] == "APPROVE":
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    elif regression["decision"] == "ROLLBACK":
        print("❌ QUYẾT ĐỊNH: HOÀN TÁC BẢN CẬP NHẬT (ROLLBACK)")
    else:
        print("[BLOCK] QUYET DINH: TU CHOI (BLOCK RELEASE)")


if __name__ == "__main__":
    asyncio.run(main())
