"""Reuse read-only retrieval and the pinned CPU reranker from day 23."""
from shared import load_day23

_impl = load_day23("retrieval")
DEFAULT_INDEX = _impl.DEFAULT_INDEX
RERANKER_ID = _impl.RERANKER_ID
RERANKER_REVISION = _impl.RERANKER_REVISION
Retriever = _impl.Retriever
Reranker = _impl.Reranker
