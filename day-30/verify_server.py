"""Run three short real local generations and save a diagnostic report."""
import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:11434"


def call(path, body=None):
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=300) as response:
        return json.load(response)


def generate(label, messages):
    started = time.monotonic()
    data = call("/api/chat", {
        "model": "qwen3:1.7b", "messages": messages,
        "think": False, "stream": False, "keep_alive": "30m",
        "options": {"num_ctx": 2048, "num_predict": 128,
                    "num_thread": 2, "temperature": 0},
    })
    answer = data.get("message", {}).get("content", "")
    result = {
        "label": label, "seconds": round(time.monotonic() - started, 3),
        "completed": data.get("done") is True,
        "done_reason": data.get("done_reason"),
        "answer": answer, "prompt_tokens": data.get("prompt_eval_count"),
        "generated_tokens": data.get("eval_count"),
        "generation_seconds": data.get("eval_duration", 0) / 1e9,
        "load_seconds": data.get("load_duration", 0) / 1e9,
    }
    if not result["completed"] or not answer.strip() or data.get("error"):
        raise RuntimeError(f"Incomplete generation: {label}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"version": call("/api/version"), "models": call("/api/tags")}
    report["model_details"] = call("/api/show", {"model": "qwen3:1.7b"})
    warmup = generate("first-request", [{"role": "user", "content": "Сколько будет 2 + 2? Ответь одним числом."}])
    report["first_request"] = warmup
    print(json.dumps(warmup, ensure_ascii=False), flush=True)
    # Two independent HTTP connections, with processing serialized by Ollama.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(generate, "client-a", [
                {"role": "user", "content": "Мой кодовый цвет — синий."},
                {"role": "assistant", "content": "Запомнил: синий."},
                {"role": "user", "content": "Назови мой кодовый цвет одним словом."},
            ]),
            pool.submit(generate, "client-b", [
                {"role": "user", "content": "Мой кодовый цвет — зелёный."},
                {"role": "assistant", "content": "Запомнил: зелёный."},
                {"role": "user", "content": "Назови мой кодовый цвет одним словом."},
            ]),
        ]
        report["concurrent_requests"] = [future.result() for future in futures]
    report["loaded_models"] = call("/api/ps")
    report["reachable_after_requests"] = call("/api/version")
    report["quantization"] = report["model_details"].get("details", {}).get("quantization_level")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for result in report["concurrent_requests"]:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if report["quantization"] != "Q4_K_M":
        raise RuntimeError("Unexpected model quantization")
    if "4" not in warmup["answer"]:
        raise RuntimeError("Incorrect arithmetic answer")
    for result, expected in zip(report["concurrent_requests"], ("синий", "зелёный")):
        if expected not in result["answer"].lower():
            raise RuntimeError(f"Unexpected conversation answer: {result['label']}")


if __name__ == "__main__":
    main()
