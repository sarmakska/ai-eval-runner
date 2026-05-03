import os
import httpx


async def openai_completion(prompt: str, model: str = "gpt-4o-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return f"[no OPENAI_API_KEY] echo: {prompt[:80]}"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
