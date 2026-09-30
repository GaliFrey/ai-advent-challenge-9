"""Read-only index adapter and local multilingual cross-encoder."""
from contextlib import closing
from pathlib import Path
import math
import sqlite3

from shared import ROOT, retrieval

DEFAULT_INDEX = retrieval.DEFAULT_INDEX
RERANKER_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
RERANKER_CACHE = Path(__file__).resolve().parent / "models" / "reranker"


def quiet_model_progress():
    from huggingface_hub.utils import disable_progress_bars
    from transformers.utils.logging import disable_progress_bar
    disable_progress_bars()
    disable_progress_bar()


class Retriever:
    def __init__(self, path=DEFAULT_INDEX):
        self.path = Path(path).resolve()
        self.indexer = retrieval.indexer_module()
        with closing(self.connect()) as connection:
            self.metadata = retrieval.validate_index(connection, self.indexer)
        self.model = None

    def connect(self):
        return sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)

    def load(self):
        quiet_model_progress()
        from sentence_transformers import SentenceTransformer
        cache = ROOT / "day-21/indexes/hf-cache/hub"
        snapshot = cache / ("models--" + self.indexer.MODEL_ID.replace("/", "--")) / "snapshots" / self.indexer.MODEL_REVISION
        if (snapshot / "model.safetensors").is_file():
            self.model = SentenceTransformer(str(snapshot), local_files_only=True, device="cpu")
        else:
            self.model = SentenceTransformer(self.indexer.MODEL_ID, revision=self.indexer.MODEL_REVISION,
                                            cache_folder=str(cache), device="cpu")

    def search(self, query, top_k):
        if self.model is None:
            raise RuntimeError("Embedding model has not been loaded")
        with closing(self.connect()) as connection:
            return self.indexer.search(connection, query, self.model, retrieval.STRATEGY, top_k)


class Reranker:
    def __init__(self):
        quiet_model_progress()
        from huggingface_hub import snapshot_download
        from sentence_transformers import CrossEncoder
        from torch import nn
        snapshot = RERANKER_CACHE / ("models--" + RERANKER_ID.replace("/", "--")) / "snapshots" / RERANKER_REVISION
        cached = all((snapshot / name).is_file() for name in
                     ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"))
        if not cached:
            snapshot = Path(snapshot_download(RERANKER_ID, revision=RERANKER_REVISION,
                                              cache_dir=str(RERANKER_CACHE),
                                              allow_patterns=["*.json", "*.model", "model.safetensors"]))
        self.model = CrossEncoder(str(snapshot), local_files_only=True,
                                  device="cpu", max_length=512, activation_fn=nn.Sigmoid())

    def rank(self, question, chunks):
        if not chunks:
            return []
        scores = self.model.predict([(question, chunk["text"]) for chunk in chunks],
                                    batch_size=8, show_progress_bar=False)
        result = []
        for chunk, score in zip(chunks, scores, strict=True):
            value = float(score)
            if not math.isfinite(value):
                raise ValueError("Reranker returned a nonfinite score")
            result.append({**chunk, "rerank_score": value})
        return sorted(result, key=lambda chunk: (-chunk["rerank_score"], chunk["chunk_id"]))
