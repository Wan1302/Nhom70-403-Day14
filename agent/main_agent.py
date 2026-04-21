"""
RAG Agent using ChromaDB retrieval + OpenAI generation.
V1: top_k=3, basic system prompt
V2: top_k=5, stricter system prompt (anti-hallucination)
"""
import asyncio
import os
import time
from typing import Dict
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rag.retriever import Retriever

load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_API_KEY") or os.getenv("OPENAI_API_KEY")
GENERATION_MODEL = "gpt-4o-mini"

# Pricing per 1M tokens (gpt-4o-mini)
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60

SYSTEM_PROMPTS = {
    "v1": (
        "You are a helpful assistant. Answer the question based on the provided context. "
        "Be concise and factual."
    ),
    "v2": (
        "You are a precise assistant. Answer the question using ONLY the information in the provided context. "
        "If the context does not contain enough information to answer, respond exactly: "
        "'I don't have enough information to answer this.' "
        "Do not guess, invent, or add information beyond what is in the context. "
        "Be concise."
    ),
}


class MainAgent:
    def __init__(self, version: str = "v1"):
        if version not in ("v1", "v2"):
            raise ValueError("version must be 'v1' or 'v2'")
        self.version = version
        self.name = f"RAGAgent-{version}"
        top_k = 3 if version == "v1" else 5
        self._retriever = Retriever(top_k=top_k)
        self._client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def query(self, question: str) -> Dict:
        t0 = time.perf_counter()

        # Step 1: Retrieve relevant documents
        retrieved = await self._retriever.retrieve(question)
        context_text = "\n\n".join(
            f"[Source: {r['doc_id']}]\n{r['content']}" for r in retrieved
        )
        retrieved_ids = [r["doc_id"] for r in retrieved]

        # Step 2: Generate answer with LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[self.version]},
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}",
            },
        ]

        response = await self._client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=300,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        usage = response.usage
        cost_usd = (
            usage.prompt_tokens * INPUT_COST_PER_M
            + usage.completion_tokens * OUTPUT_COST_PER_M
        ) / 1_000_000

        return {
            "answer": response.choices[0].message.content.strip(),
            "contexts": [r["content"] for r in retrieved],
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "model": GENERATION_MODEL,
                "version": self.version,
                "tokens_used": usage.total_tokens,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cost_usd": round(cost_usd, 6),
                "latency_ms": round(latency_ms, 1),
            },
        }


if __name__ == "__main__":
    async def test():
        for ver in ("v1", "v2"):
            agent = MainAgent(version=ver)
            print(f"\n=== {agent.name} ===")
            resp = await agent.query("Was Abraham Lincoln the sixteenth President of the United States?")
            print("Answer:", resp["answer"])
            print("Retrieved:", resp["retrieved_ids"])
            print("Cost: $", resp["metadata"]["cost_usd"])
            print("Latency:", resp["metadata"]["latency_ms"], "ms")

    asyncio.run(test())
