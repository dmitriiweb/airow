"""airow package public API and version information.

Exposes `Airow` for batched, row-wise LLM inference over pandas DataFrames
and `OutputColumn` for declaring structured outputs.
"""
__version__ = "0.1.0"

from .airow import Airow
from .schemas import OutputColumn

__all__ = ["Airow", "OutputColumn"]
