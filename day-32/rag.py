"""In-memory semantic index, rebuilt from the exact head snapshot on every run."""
from dataclasses import replace
from pathlib import PurePosixPath
import re

MODEL_ID = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MODEL_REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
MAX_TOKENS = 112
OVERLAP = 16


def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device='cpu')


def slices(text, tokenizer):
    # Same offset-window approach as day-21/indexer.py; standalone for CI.
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)['offset_mapping']
    offsets = [(a, b) for a, b in offsets if b > a]
    result, start = [], 0
    while start < len(offsets):
        end = min(start + MAX_TOKENS, len(offsets))
        left, right = offsets[start][0], offsets[end - 1][1]
        while len(tokenizer(text[left:right], add_special_tokens=False,
                            return_offsets_mapping=True, truncation=False,
                            verbose=False)['offset_mapping']) > MAX_TOKENS:
            end -= 1
            if end <= start:
                raise ValueError('Не удалось разбить текст в пределах окна модели.')
            right = offsets[end - 1][1]
        result.append((left, right, text[left:right]))
        if end == len(offsets):
            break
        start = max(start + 1, end - OVERLAP)
    return result


def retrieve(snapshot, model):
    import numpy as np
    chunks = []
    originals = {s.path: s for s in snapshot['corpus']}
    for source in snapshot['corpus']:
        for left, right, text in slices(source.text, model.tokenizer):
            chunks.append(replace(source, start=source.text.count('\n', 0, left) + 1,
                                  end=source.text.count('\n', 0, right - 1) + 1, text=text))
    if not chunks:
        raise ValueError('RAG-корпус пуст.')
    vectors = model.encode([s.text for s in chunks], normalize_embeddings=True,
                           batch_size=32, show_progress_bar=False)
    chosen, associations = {}, {}
    for change in snapshot['changes']:
        # Embed all changed lines, including removed lines, rather than truncating
        # a long query in the sentence-transformer. Average normalized windows.
        changed = '\n'.join(line for line in change['diff'].splitlines()
                            if line.startswith(('+', '-')) and not line.startswith(('+++', '---')))
        query = change['path'] + '\n' + changed
        windows = [part for _, _, part in slices(query, model.tokenizer)]
        query_vectors = model.encode(windows, normalize_embeddings=True, show_progress_bar=False)
        query_vector = np.mean(query_vectors, axis=0)
        norm = np.linalg.norm(query_vector)
        if norm == 0:
            raise ValueError('Нулевой embedding запроса.')
        scores = np.asarray(vectors) @ (query_vector / norm)
        # Exact API names complement multilingual semantic similarity; short
        # code-only queries otherwise often miss their Russian documentation.
        direct = next((s for s in snapshot['direct'] if s.path == change['path']
                       and s.kind == 'head'), None)
        if direct is None:
            direct = next((s for s in snapshot['direct'] if s.path == change['path']), None)
        names = set(re.findall(r'export\s+(?:const|function|class)\s+(\w+)', direct.text)) if direct else set()
        basename = PurePosixPath(change['path']).name
        for i, chunk in enumerate(chunks):
            scores[i] += 0.4 * (basename in chunk.text)
            if names:
                scores[i] += 0.3 * sum(name in chunk.text for name in names) / len(names)
        indices = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), i))
        identities = []
        for kind in ('docs', 'code'):
            selected = 0
            for i in indices:
                source = chunks[i]
                if source.kind != kind or source.path == change['path']:
                    continue
                # Give generation complete short paragraphs, keeping embedding
                # windows small. Quotes and line ranges still refer to head.
                original = originals[source.path]
                if kind == 'docs':
                    lines = original.text.splitlines(keepends=True)
                    left, right = source.start - 1, source.end
                    while left > 0 and lines[left - 1].strip():
                        left -= 1
                    while right < len(lines) and lines[right].strip():
                        right += 1
                    paragraph = ''.join(lines[left:right])
                    if len(paragraph) <= 1800:
                        source = replace(source, start=left + 1, end=right, text=paragraph)
                key = (source.path, source.start, source.end, source.text)
                # Neighboring overlapping windows add little context.
                if any(chosen[k].path == source.path and
                       max(chosen[k].start, source.start) <= min(chosen[k].end, source.end)
                       for k in chosen if chosen[k].id in identities):
                    continue
                if key not in chosen:
                    chosen[key] = replace(source, id=f'R{len(chosen) + 1}')
                identities.append(chosen[key].id)
                selected += 1
                if selected == 2:
                    break
        associations[change['path']] = identities
    if len(chosen) > 80:
        raise ValueError('Слишком много RAG-фрагментов.')
    return list(chosen.values()), associations, len(chunks)
