import io
import re
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

import indexer


class FakeTokenizer:
    is_fast = True

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping, truncation, verbose):
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}

    def num_special_tokens_to_add(self, pair=False):
        return 2


class FakeModel:
    tokenizer = FakeTokenizer()
    max_seq_length = 128

    def encode(self, texts, **_kwargs):
        def vector(text):
            return [float("кошка" in text.lower()), float("собака" in text.lower())]
        return [vector(text) for text in texts] if isinstance(texts, list) else vector(texts)


class IndexerTests(unittest.TestCase):
    def test_fixed_chunks_cover_tokens_and_respect_limit(self):
        body = "# Тема\n\n" + " ".join(f"слово{i}" for i in range(31))
        chunks = indexer.make_chunks("day-01/README.md", body, FakeTokenizer(), "fixed", 20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.token_count <= 20 for chunk in chunks))
        self.assertEqual("Тема", chunks[0].title)
        self.assertIn("слово30", chunks[-1].text)
        self.assertEqual(len(chunks), len({chunk.chunk_id for chunk in chunks}))

    def test_structure_preserves_section_metadata_and_long_paragraph(self):
        body = "# Курс\n\n## Кошки\n\n" + " ".join("кошка" for _ in range(30))
        body += "\n\n## Собаки\n\nсобака живёт дома"
        chunks = indexer.make_chunks("day-01/README.md", body, FakeTokenizer(), "structure", 20)
        self.assertTrue(any("Кошки" in chunk.section for chunk in chunks))
        self.assertTrue(any("Собаки" in chunk.section for chunk in chunks))
        self.assertTrue(all(chunk.token_count <= 20 for chunk in chunks))
        self.assertTrue(all("Собаки" not in chunk.section or "кошка" not in chunk.text
                            for chunk in chunks))

    def test_section_windows_overlap_without_crossing_sections(self):
        body = "# Курс\n\n## Кошки\n\n" + " ".join(f"слово{i}" for i in range(35))
        body += "\n\n## Собаки\n\nсобака живёт дома"
        chunks = indexer.make_chunks("day-01/README.md", body, FakeTokenizer(),
                                     "section_windows", 20)
        cats = [chunk for chunk in chunks if "Кошки" in chunk.section]
        dogs = [chunk for chunk in chunks if "Собаки" in chunk.section]
        self.assertGreater(len(cats), 1)
        self.assertEqual(1, len(dogs))
        self.assertTrue(all(chunk.token_count <= 20 for chunk in chunks))
        self.assertTrue(all(chunk.text.startswith("Раздел: Курс / Кошки") for chunk in cats))
        self.assertTrue(all("слово" not in chunk.text for chunk in dogs))
        self.assertTrue(set(re.findall(r"слово\d+", cats[0].text)) &
                        set(re.findall(r"слово\d+", cats[1].text)))
        self.assertEqual({f"слово{i}" for i in range(35)},
                         set().union(*(set(re.findall(r"слово\d+", chunk.text)) for chunk in cats)))
        self.assertFalse(any(chunk.text.strip() == "# Курс" for chunk in chunks))

    def test_truncated_tokenizer_output_is_rejected(self):
        class TruncatingTokenizer(FakeTokenizer):
            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                result["offset_mapping"] = result["offset_mapping"][:5]
                return result

        with self.assertRaisesRegex(ValueError, "full input"):
            indexer.token_windows(" ".join(f"слово{i}" for i in range(20)),
                                  TruncatingTokenizer(), 10)

    def test_build_search_and_replace_index(self):
        docs = [("day-01/README.md", "# Кошки\n\nкошка спит"),
                ("day-02/README.md", "# Собаки\n\nсобака гуляет")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "readmes.sqlite3"
            with patch.object(indexer, "read_documents", return_value=docs):
                result = indexer.build(path, FakeModel())
                indexer.build(path, FakeModel())
            self.assertEqual(2, result["documents"])
            with closing(sqlite3.connect(path)) as connection:
                self.assertEqual(6, connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
                for strategy in indexer.STRATEGIES:
                    hits = indexer.search(connection, "кошка", FakeModel(), strategy)
                    self.assertEqual("day-01/README.md", hits[0]["source"])
                    self.assertEqual("Кошки", hits[0]["title"])
                    self.assertTrue(hits[0]["chunk_id"].startswith(strategy))

    def test_model_mismatch_is_rejected(self):
        with closing(sqlite3.connect(":memory:")) as connection:
            indexer.create_schema(connection)
            connection.execute("INSERT INTO meta VALUES ('model_id', 'other-model')")
            with self.assertRaisesRegex(ValueError, "model differs"):
                indexer.search(connection, "кошка", FakeModel(), "fixed")

    def test_demo_shows_three_strategies_metadata_and_ranks(self):
        docs = [("day-01/README.md", "# Кошки\n\nкошка спит"),
                ("day-02/README.md", "# Собаки\n\nсобака гуляет")]
        with tempfile.TemporaryDirectory() as directory:
            output_buffer = io.StringIO()
            console = Console(file=output_buffer, width=160, force_terminal=False)
            with patch.object(indexer, "read_documents", return_value=docs):
                indexer.demo(Path(directory) / "index.sqlite3", FakeModel(), console)
            output = output_buffer.getvalue()
        for label in ("fixed", "structure", "section_windows", "Как отличаются стратегии",
                      "source:", "title:", "section:",
                      "chunk_id:", "hit@5", "MRR@5", "место нужного README",
                      "Вывод", "не полноту ответа"):
            self.assertIn(label, output)


if __name__ == "__main__":
    unittest.main()
