"""Real local retrieval with synthetic generation; not a quality benchmark."""
import json
from datetime import datetime, timezone

from pipeline import DAY, Runner, load_cases, save_report


def synthetic(messages, key, model):
    payload = json.loads(messages[1]["content"])
    if "response" in payload:
        response = {"verdict": "fail", "checks": {"support": True, "coverage": False, "abstention": True},
                    "explanation": "Синтетическая оценка: тестовый ответ не раскрывает вопрос; это не реальная LLM-проверка."}
    elif "sources" not in payload:
        response = {"query": payload["question"]}
    else:
        source = payload["sources"][0]
        response = {"status": "answered", "answer": [{"text": "Тестовая выдержка из найденного фрагмента; это не ответ на вопрос.",
            "citations": [{"source_id": source["source_id"], "quote": source["text"][:240]}]}],
            "sources": [{k: source[k] for k in ("source_id", "source", "section", "chunk_id")}], "clarification": ""}
    return {"answer": json.dumps(response, ensure_ascii=False), "usage": {}, "llm_seconds": 0}


if __name__ == "__main__":
    path = DAY / "resources" / ("local-check-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    def mark(report):
        report["validation_only"] = True
        save_report(path, report)
    report = Runner(key="offline-placeholder", completion=synthetic).run(load_cases(), path, notify=mark)
    print(f"{report['status']}: {len(report['items'])} вопросов; синтетические ответы; {path}")
    if report.get("error"):
        print(report["error"])
    raise SystemExit(0 if report["status"] == "complete" else 1)
