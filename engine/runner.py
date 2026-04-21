import asyncio
import time
from typing import Any, Dict, List


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    @staticmethod
    def _extract_tokens_used(response: Dict[str, Any]) -> int:
        metadata = response.get("metadata") or {}
        tokens_used = metadata.get("tokens_used", 0)
        try:
            return int(tokens_used)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _estimate_cost(cls, tokens_used: int, model_name: str | None = None) -> float:
        # Fallback estimate if the agent does not return an explicit cost.
        if tokens_used <= 0:
            return 0.0

        model_name = (model_name or "").lower()
        if "gpt-4" in model_name:
            return round(tokens_used * 0.00003, 6)
        if "claude" in model_name:
            return round(tokens_used * 0.000025, 6)
        return round(tokens_used * 0.00002, 6)

    async def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()

        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time

        ragas_scores = await self.evaluator.score(test_case, response)
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case["expected_answer"],
        )

        metadata = response.get("metadata") or {}
        model_name = metadata.get("model")
        tokens_used = self._extract_tokens_used(response)
        generation_cost_usd = self._safe_float(metadata.get("cost_usd"))
        if generation_cost_usd <= 0:
            generation_cost_usd = self._estimate_cost(tokens_used, model_name)

        judge_cost_usd = self._safe_float(judge_result.get("judge_cost_usd"))
        total_cost_usd = round(generation_cost_usd + judge_cost_usd, 6)

        return {
            "test_case": test_case["question"],
            "expected_answer": test_case.get("expected_answer"),
            "agent_response": response["answer"],
            "contexts": response.get("contexts", []),
            "latency": latency,
            "latency_ms": round(latency * 1000, 2),
            "tokens_used": tokens_used,
            "generation_cost_usd": generation_cost_usd,
            "judge_cost_usd": judge_cost_usd,
            "total_cost_usd": total_cost_usd,
            "model": model_name,
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": "fail" if judge_result["final_score"] < 3 else "pass",
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        """
        Run the benchmark in batches to keep concurrency bounded.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        results: List[Dict[str, Any]] = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results

    @staticmethod
    def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a small summary object from raw benchmark results.
        This is optional today, but useful for future reporting.
        """
        total = len(results)
        if total == 0:
            return {
                "total": 0,
                "avg_latency": 0.0,
                "total_tokens_used": 0,
                "total_cost_estimate": 0.0,
                "pass_rate": 0.0,
            }

        avg_latency = sum(item.get("latency", 0.0) for item in results) / total
        total_tokens_used = sum(int(item.get("tokens_used", 0)) for item in results)
        total_generation_cost = sum(float(item.get("generation_cost_usd", 0.0)) for item in results)
        total_judge_cost = sum(float(item.get("judge_cost_usd", 0.0)) for item in results)
        total_cost_estimate = sum(float(item.get("total_cost_usd", 0.0)) for item in results)
        pass_rate = sum(1 for item in results if item.get("status") == "pass") / total

        return {
            "total": total,
            "avg_latency": avg_latency,
            "total_tokens_used": total_tokens_used,
            "total_generation_cost_usd": total_generation_cost,
            "total_judge_cost_usd": total_judge_cost,
            "total_cost_estimate": total_cost_estimate,
            "pass_rate": pass_rate,
        }
