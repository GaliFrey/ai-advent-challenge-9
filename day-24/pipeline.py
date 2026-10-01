"""One grounded RAG chain with incremental results and deterministic abstention."""
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from evidence import messages, refusal, validate
from judge import messages as judge_messages, validate as validate_judge
from retrieval import DEFAULT_INDEX, RERANKER_ID, RERANKER_REVISION, Retriever, Reranker
from session_log import SessionLog
from shared import load_day23, rag

_helpers = load_day23("pipeline")
rewrite_question = _helpers.rewrite_question
select_chunks = _helpers.select_chunks
save_report = _helpers.save_report
DAY = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    top_k_before: int = 20
    top_k_after: int = 5
    rerank_threshold: float = 0.1

    def __post_init__(self):
        if type(self.top_k_before) is not int or type(self.top_k_after) is not int or not 1 <= self.top_k_after <= self.top_k_before <= 100:
            raise ValueError("Нужно 1 ≤ итоговый K ≤ кандидаты K ≤ 100")
        if not math.isfinite(self.rerank_threshold) or not 0 <= self.rerank_threshold <= 1:
            raise ValueError("Порог reranker должен быть от 0 до 1")


def load_cases():
    return json.loads((DAY / "questions.json").read_text(encoding="utf-8"))


class Runner:
    def __init__(self, *, key, model="deepseek-flash", settings=None, index=DEFAULT_INDEX,
                 completion=rag.complete, retriever_factory=Retriever, reranker_factory=Reranker):
        self.key, self.model = key, model
        self.settings = settings or Settings()
        self.index = index
        self.completion = completion
        self.retriever_factory, self.reranker_factory = retriever_factory, reranker_factory

    def run(self, cases, path, log=None, notify=lambda report: None, cancelled=lambda: False):
        if not self.key.strip():
            raise ValueError("Задайте DEEPSEEK_API_KEY в day-24/.env")
        if not cases or any(not case["question"].strip() for case in cases):
            raise ValueError("Нужен непустой вопрос")
        log = log or SessionLog()
        report = {"schema_version": 2, "day": 24, "created_at": datetime.now(timezone.utc).isoformat(),
                  "model": self.model, "parameters": rag.PARAMETERS, "settings": asdict(self.settings),
                  "reranker": {"model": RERANKER_ID, "revision": RERANKER_REVISION},
                  "status": "running", "setup": [], "items": [{**case, "status": "pending",
                  "steps": [], "judge": {"status": "pending"}} for case in cases]}
        active, item, stage, number = report["setup"], None, "index", None

        def publish():
            save_report(path, report)
            notify(report)

        def step(name, operation):
            nonlocal stage
            stage = name
            if cancelled():
                raise InterruptedError
            entry = {"stage": name, "status": "running"}
            active.append(entry)
            publish()
            started = time.perf_counter()
            try:
                with log.stage(name, question_number=number):
                    result = operation()
            except Exception as error:
                entry.update(status="failed", error_type=type(error).__name__)
                raise
            finally:
                entry["seconds"] = round(time.perf_counter() - started, 3)
            entry["status"] = "complete"
            return result

        try:
            retriever = step("index", lambda: self.retriever_factory(self.index))
            report["index"] = retriever.metadata
            step("embedding_model", retriever.load)
            reranker = step("reranker_model", self.reranker_factory)
            for number, item in enumerate(report["items"], 1):
                active = item["steps"]
                item["status"] = "running"
                item["rewrite"] = step("rewrite", lambda: rewrite_question(item["question"], key=self.key,
                    model=self.model, completion=self.completion))
                chunks = step("search", lambda: retriever.search(item["rewrite"]["query"], self.settings.top_k_before))
                candidates = [{**chunk, "rank_before": i} for i, chunk in enumerate(chunks, 1)]
                item["candidates"] = candidates
                ranked = step("rerank", lambda: reranker.rank(item["question"], candidates))
                annotated, selected = step("filter", lambda: select_chunks(ranked, self.settings.top_k_after,
                    self.settings.rerank_threshold, "rerank_score"))
                item.update(candidates=annotated, chunks=selected)
                publish()
                if not selected:
                    item["origin"] = "threshold"
                    raw = json.dumps(refusal(), ensure_ascii=False)
                    item["usage"] = {}
                    item["llm_seconds"] = 0
                else:
                    item["origin"] = "model"
                    request = messages(item["question"], selected)
                    item["messages"] = request
                    output = step("answer", lambda: self.completion(request, self.key, self.model))
                    raw = output["answer"]
                    item.update({k: v for k, v in output.items() if k != "answer"})
                item["raw_answer"] = raw
                data, checks = step("validate", lambda: validate(raw, selected))
                item.update(response=data if checks["passed"] else None, checks=checks,
                            status="running" if checks["passed"] else "invalid")
                if checks["passed"]:
                    request = judge_messages(item["question"], data, selected, item["origin"])
                    item["judge"] = {"status": "running", "messages": request}
                    publish()
                    output = step("judge", lambda: self.completion(request, self.key, self.model))
                    assessment, errors = validate_judge(output["answer"])
                    item["judge"].update(output)
                    item["judge"].update(status="invalid" if errors else "complete", assessment=assessment, errors=errors)
                    item["status"] = "invalid" if errors else "complete"
                else:
                    item["judge"] = {"status": "skipped"}
                publish()
            report["status"] = "complete" if all(i["status"] == "complete" for i in report["items"]) else "invalid"
        except InterruptedError:
            report["status"] = "cancelled"
            if item and item["status"] == "running":
                item["status"] = "cancelled"
                if item.get("judge", {}).get("status") == "running":
                    item["judge"]["status"] = "cancelled"
        except Exception as error:
            report.update(status="failed", error={"stage": stage, "type": type(error).__name__})
            if item:
                item["status"] = "failed"
                if stage == "judge":
                    item["judge"]["status"] = "failed"
        publish()
        return report
