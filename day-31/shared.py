"""Reuse the Day 21 chunking and vector format without its fixed corpus."""
import importlib.util
from pathlib import Path
import sys

NAME = "day31_indexer21"
if NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        NAME, Path(__file__).resolve().parent.parent / "day-21" / "indexer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = module
    spec.loader.exec_module(module)
indexer = sys.modules[NAME]
