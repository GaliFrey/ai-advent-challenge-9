"""Scoped read-only retrieval and compact multi-query reranking."""
from contextlib import closing
from itertools import zip_longest
from math import ceil
from shared import load_day23

_impl = load_day23('retrieval')
DEFAULT_INDEX = _impl.DEFAULT_INDEX
RERANKER_ID = _impl.RERANKER_ID
RERANKER_REVISION = _impl.RERANKER_REVISION
Reranker = _impl.Reranker


class Retriever(_impl.Retriever):
    def search(self, query, top_k, source=None):
        if source is None:
            return super().search(query, top_k)
        if self.model is None:
            raise RuntimeError('Embedding model has not been loaded')
        # The shared search already scans all vectors. Keep its ranking, then
        # limit within the requested source, rather than filtering a global top-K.
        with closing(self.connect()) as connection:
            strategy = _impl.retrieval.STRATEGY
            count = connection.execute('SELECT COUNT(*) FROM chunks WHERE strategy = ?',
                                       (strategy,)).fetchone()[0]
            rows = self.indexer.search(connection, query, self.model, strategy, max(1, count))
        return [row for row in rows if row['source'] == source][:top_k]


def search_candidates(retriever, searches, top_k):
    groups = []
    per_query = max(1, ceil(top_k / len(searches)))
    for spec in searches:
        kwargs = {'source': spec['source']} if spec['source'] is not None else {}
        groups.append(retriever.search(spec['query'], per_query, **kwargs))
    by_id = {}
    # Round-robin preserves both sides of a comparison under a shared budget.
    for row in zip_longest(*groups):
        for query_index, chunk in enumerate(row):
            if chunk is None:
                continue
            identity = chunk['chunk_id']
            if identity not in by_id:
                if len(by_id) >= top_k:
                    continue
                by_id[identity] = {**chunk, 'rank_before': len(by_id) + 1, 'query_ids': []}
            if query_index not in by_id[identity]['query_ids']:
                by_id[identity]['query_ids'].append(query_index)
    return list(by_id.values())


def rerank_candidates(reranker, searches, candidates):
    merged = {c['chunk_id']: {**c, 'rerank_scores': [], 'rerank_score': 0.0} for c in candidates}
    for index, spec in enumerate(searches):
        group = [c for c in candidates if index in c['query_ids']]
        if not group:
            continue
        for ranked in reranker.rank(spec['query'], group):
            row = merged[ranked['chunk_id']]
            row['rerank_scores'].append({'query': index, 'score': ranked['rerank_score']})
            row['rerank_score'] = max(row['rerank_score'], ranked['rerank_score'])
    return sorted(merged.values(), key=lambda c: (-c['rerank_score'], c['chunk_id']))


def select_context(ranked, top_k, threshold, query_count):
    chosen = set()
    if top_k >= query_count:
        for index in range(query_count):
            eligible = next((c for c in ranked if any(
                score['query'] == index and score['score'] >= threshold
                for score in c['rerank_scores'])), None)
            if eligible:
                chosen.add(eligible['chunk_id'])
    for chunk in ranked:
        if len(chosen) >= top_k:
            break
        if chunk['rerank_score'] >= threshold:
            chosen.add(chunk['chunk_id'])
    annotated, selected = [], []
    for rank, chunk in enumerate(ranked, 1):
        included = chunk['chunk_id'] in chosen
        decision = 'В контексте' if included else (
            'Ниже порога' if chunk['rerank_score'] < threshold else 'За пределами итогового top-K')
        row = {**chunk, 'rank_after': rank, 'decision': decision,
               'source_id': f'S{len(selected) + 1}' if included else None}
        annotated.append(row)
        if included:
            selected.append(row)
    return annotated, selected
