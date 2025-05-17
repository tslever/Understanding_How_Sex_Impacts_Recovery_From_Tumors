"""
CD8 Analysis package initialization
"""

from importlib import import_module
from types import ModuleType
from typing import Any, Final

__all__: Final = ["CD8Analysis"]


def __getattr__(name: str) -> Any:  # PEP 562
    if name == "CD8Analysis":
        mod: ModuleType = import_module(".cd8_analysis", __name__)
        attr = getattr(mod, "CD8Analysis")
        globals()[name] = attr          # cache for future look-ups
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")