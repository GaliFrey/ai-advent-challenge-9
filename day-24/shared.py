"""Load stable day-22 helpers under unique module names."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_day22(name):
    key = f"day24_shared_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "day-22" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


rag = load_day22("rag")
retrieval = load_day22("retrieval")


def load_day23(name):
    key = f"day24_shared23_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "day-23" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]
