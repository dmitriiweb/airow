"""Abstract interface for dataframe backends."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any


class DataFrameBackend(ABC):
    """Operations required by :class:`airow.Airow` to process a dataframe.

    Third-party backends can implement this class and pass an instance to
    ``Airow(backend=...)``.
    """

    @abstractmethod
    def supports(self, dataframe: object) -> bool:
        """Return whether this backend can process ``dataframe``."""

    @abstractmethod
    def copy(self, dataframe: Any) -> Any:
        """Return an independent dataframe suitable for result writes."""

    @abstractmethod
    def row_count(self, dataframe: Any) -> int:
        """Return the number of rows in ``dataframe``."""

    @abstractmethod
    def read_row(
        self,
        dataframe: Any,
        position: int,
        columns: Iterable[str],
    ) -> Mapping[str, object]:
        """Read named values from a row selected by integer position."""

    @abstractmethod
    def write_batch(
        self,
        dataframe: Any,
        updates: Mapping[int, Mapping[str, object]],
    ) -> Any:
        """Apply position-keyed row updates and return the resulting dataframe."""
