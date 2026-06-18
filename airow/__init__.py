"""airow package public API and version information.

Exposes `Airow` for batched, row-wise LLM inference over DataFrames and
`OutputColumn` for declaring structured outputs.
"""

__version__ = "0.2.0"

from .airow import Airow
from .backends import DataFrameBackend, PandasBackend, PolarsBackend
from .schemas import OutputColumn

__all__ = [
    "Airow",
    "DataFrameBackend",
    "OutputColumn",
    "PandasBackend",
    "PolarsBackend",
]
