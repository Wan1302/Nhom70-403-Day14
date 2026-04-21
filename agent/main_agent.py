"""
RAG Agent using ChromaDB retrieval + OpenAI generation.
V1: fixed-size chunking (200 chars), top_k=5, strict anti-hallucination prompt
V2: paragraph-based chunking, top_k=5, strict anti-hallucination prompt
Only difference: chunking strategy (fixed-size vs semantic paragraph).
"""
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.vector_store import search, search_v1

load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_API_KEY") or os.getenv("OPENAI_API_KEY")
GENERATION_MODEL = "gpt-4o-mini"

# Pricing per 1M tokens (gpt-4o-mini)
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60

# TỐI ƯU HÓA: Phân rõ Role, Rules khắt khe và dùng cấu trúc chặn Ảo giác
_STRICT_PROMPT = """
<system_role>
You are an elite, highly precise QA Agent. Your sole purpose is to answer user questions based STRICTLY on the provided retrieved context.
</system_role>

<instructions>
1. Carefully read the information inside the <context> tags provided by the user.
2. Formulate your answer using ONLY facts explicitly stated in that context.
3. If the answer cannot be found in the context, you MUST strictly fallback and output EXACTLY this phrase: "I don't have enough information to answer this."
</instructions>

<constraints>
- NEVER use your internal knowledge to answer.
- NEVER guess, hallucinate, or infer details not present in the text.
- DO NOT add conversational filler (e.g., "Based on the context...", "Here is the answer..."). Just output the direct answer.
</constraints>
"""


class MainAgent:
    def __init__(self, version: str = "v1"):
        if version not in ("v1", "v2"):
            raise ValueError("version must be 'v1' or 'v2'")
        self.version = version
        self.name = f"RAGAgent-{version}"
        # Both use top_k=5; only difference is chunking strategy
        self._top_k = 5
        self._search_fn = search_v1 if version == "v1" else search
        self._client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def query(self, question: str) -> Dict:
        t0 = time.perf_counter()

        # Step 1: Retrieve relevant documents
        retrieved = await asyncio.to_thread(self._search_fn, question, self._top_k)
        context_text = "\n\n".join(
            f"[Source: {r['article_file']}]\n{r['text']}" for r in retrieved
        )
        retrieved_ids = [r["article_file"] for r in retrieved]

        # Step 2: Generate answer with LLM
        # TỐI ƯU HÓA: Bọc data vào XML để kích hoạt Recency Bias ở cuối
        user_message_content = f"""
            <context>
            {context_text}
            </context>

            <user_question>
            {question}
            </user_question>
            """
        messages = [
            {"role": "system", "content": _STRICT_PROMPT.strip()},
            {"role": "user", "content": user_message_content.strip()},
        ]

        response = await self._client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=messages,  # type: ignore[arg-type]
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
            "contexts": [r["text"] for r in retrieved],
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
            resp = await agent.query("What is the largest religious group in Canada?")
            print("Answer:", resp["answer"])
            print("Retrieved:", resp["retrieved_ids"])
            print("Cost: $", resp["metadata"]["cost_usd"])
            print("Latency:", resp["metadata"]["latency_ms"], "ms")

    asyncio.run(test())
