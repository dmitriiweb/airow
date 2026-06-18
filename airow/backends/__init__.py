"""Built-in and extensible dataframe backends."""

from typing import Any

from .base import DataFrameBackend
from .pandas import PandasBackend
from .polars import PolarsBackend

BUILTIN_BACKENDS: tuple[type[DataFrameBackend], ...] = (
    PandasBackend,
    PolarsBackend,
)


def resolve_backend(dataframe: Any) -> DataFrameBackend:
    """Return the installed built-in backend supporting ``dataframe``."""
    for backend_type in BUILTIN_BACKENDS:
        backend = backend_type()
        if backend.supports(dataframe):
            return backend

    dataframe_type = type(dataframe)
    raise TypeError(
        "No dataframe backend supports "
        f"{dataframe_type.__module__}.{dataframe_type.__qualname__}. "
        "Install airow[pandas] or airow[polars], or pass a custom "
        "DataFrameBackend to Airow(backend=...)."
    )


__all__ = [
    "DataFrameBackend",
    "PandasBackend",
    "PolarsBackend",
    "resolve_backend",
]
