"""Bounded raw Qwen prompt; no client-controlled Ollama options."""
from pathlib import Path

import httpx
from tokenizers import Tokenizer

MODEL = "qwen3:1.7b"
CONTEXT = 2048
OUTPUT = 256
GUARD = 32
SYSTEM = "Ты полезный помощник. Отвечай по-русски, кратко и по существу."


class PromptBuilder:
    def __init__(self, path: Path):
        self.tokenizer = Tokenizer.from_file(str(path))
        self.control_tokens = {
            token for token in self.tokenizer.get_added_tokens_decoder().values()
            if token.special
        }

    def validate(self, text: str):
        if any(token.content in text for token in self.control_tokens):
            raise ValueError("Служебные токены модели нельзя использовать в сообщениях.")

    def build(self, messages: list[dict], text: str) -> tuple[str, int]:
        self.validate(text)
        turns = [{"role": "system", "content": SYSTEM}, *messages,
                 {"role": "user", "content": text}]
        prompt = "".join(
            f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n"
            for turn in turns
        ) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return prompt, len(self.tokenizer.encode(prompt, add_special_tokens=False).ids)


class Ollama:
    def __init__(self, url: str):
        self.client = httpx.AsyncClient(base_url=url, timeout=120)

    async def generate(self, prompt: str) -> dict:
        response = await self.client.post("/api/generate", json={
            "model": MODEL, "prompt": prompt, "raw": True, "stream": False,
            "think": False, "keep_alive": "30m",
            "options": {"num_ctx": CONTEXT, "num_predict": OUTPUT,
                        "num_thread": 2, "temperature": 0.2},
        })
        response.raise_for_status()
        data = response.json()
        if not data.get("done") or not data.get("response", "").strip():
            raise ValueError("Empty or incomplete model response")
        if data.get("prompt_eval_count", 0) > CONTEXT - OUTPUT:
            raise ValueError("Backend context exceeded input budget")
        return {"text": data["response"].strip(),
                "input_tokens": data.get("prompt_eval_count"),
                "output_tokens": data.get("eval_count"),
                "truncated": data.get("done_reason") == "length"}

    async def close(self):
        await self.client.aclose()
