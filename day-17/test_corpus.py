from __future__ import annotations

import unittest

import corpus


class CorpusSearchTests(unittest.TestCase):
    def test_array_sort_query_prioritizes_array_functions(self) -> None:
        results = corpus.search("сортировка массива sort", limit=3)["results"]
        titles = {item["title"] for item in results}
        self.assertIn("ArraySort()", titles)
        self.assertIn("ArraySortExt()", titles)

    def test_exact_xml_sort_stays_first(self) -> None:
        results = corpus.search("XmlElem.Sort", limit=3)["results"]
        self.assertEqual(results[0]["title"], "XmlElem.Sort()")


if __name__ == "__main__":
    unittest.main()
