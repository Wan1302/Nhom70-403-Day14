import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from agent.main_agent import MainAgent


class PerCaseEvaluator:
    """Wrapper để tính Hit Rate & MRR per-case từ retrieved_ids của agent."""

    def __init__(self):
        self._eval = RetrievalEvaluator(top_k=5)

    async def score(self, case: dict, response: dict) -> dict:
        expected_ids = case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("retrieved_ids", [])

        if expected_ids and expected_ids != ["out_of_context"]:
            hit_rate = self._eval.calculate_hit_rate(expected_ids, retrieved_ids)
            mrr = self._eval.calculate_mrr(expected_ids, retrieved_ids)
        else:
            hit_rate = 0.0
            mrr = 0.0

        return {
            "faithfulness": 1.0,
            "relevancy": 1.0,
            "retrieval": {"hit_rate": hit_rate, "mrr": mrr},
        }


async def run_benchmark_with_results(agent_version: str):
    print(f"[START] Khoi dong Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("[ERR] Thieu data/golden_set.jsonl. Hay chay 'python data/synthetic_gen.py' truoc.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("[ERR] File data/golden_set.jsonl rong. Hay tao it nhat 1 test case.")
        return None, None

    version = "v2" if "V2" in agent_version.upper() else "v1"
    runner = BenchmarkRunner(MainAgent(version=version), PerCaseEvaluator(), LLMJudge())
    results = await runner.run_all(dataset)

    total = len(results)
    summary = {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total,
        },
    }
    return results, summary


async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary


async def main():
    v1_summary = await run_benchmark("Agent_V1_Base")

    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")

    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n[RESULT] --- KET QUA SO SANH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']:.2f}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']:.2f}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    if delta > 0:
        print("[OK] QUYET DINH: CHAP NHAN BAN CAP NHAT (APPROVE)")
    else:
        print("[BLOCK] QUYET DINH: TU CHOI (BLOCK RELEASE)")


if __name__ == "__main__":
    asyncio.run(main())
