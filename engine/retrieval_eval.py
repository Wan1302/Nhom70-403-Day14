from typing import List, Dict
from engine.vector_store import search


class RetrievalEvaluator:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        results = []

        for case in dataset:
            question = case.get("question", "")
            expected_ids = case.get("expected_retrieval_ids", [])

            # Bỏ qua hard cases không có expected retrieval
            if not expected_ids or expected_ids == ["out_of_context"]:
                continue

            retrieved = search(question, top_k=self.top_k * 3)
            seen, retrieved_ids = set(), []
            for r in retrieved:
                af = r["article_file"]
                if af not in seen:
                    seen.add(af)
                    retrieved_ids.append(af)
                if len(retrieved_ids) == self.top_k:
                    break

            hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=self.top_k)
            mrr = self.calculate_mrr(expected_ids, retrieved_ids)

            results.append({
                "question": question,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
                "hit_rate": hit_rate,
                "mrr": mrr
            })

        if not results:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0, "total": 0, "details": []}

        avg_hit_rate = sum(r["hit_rate"] for r in results) / len(results)
        avg_mrr = sum(r["mrr"] for r in results) / len(results)

        return {
            "avg_hit_rate": round(avg_hit_rate, 4),
            "avg_mrr": round(avg_mrr, 4),
            "total": len(results),
            "details": results
        }
