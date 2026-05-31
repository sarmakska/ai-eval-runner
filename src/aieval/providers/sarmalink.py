import os

import httpx

BASE_URL = os.getenv("SARMALINK_BASE_URL", "https://api.sarmalink.ai/v1")


async def sarmalink_completion(prompt: str, model: str = "smart") -> str:
    api_key = os.getenv("SARMALINK_API_KEY", "")
    if not api_key:
        return f"[no SARMALINK_API_KEY] echo: {prompt[:80]}"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
