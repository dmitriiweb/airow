"""Polars dataframe backend."""

from collections.abc import Iterable, Mapping
from typing import Any

from .base import DataFrameBackend


class PolarsBackend(DataFrameBackend):
    """Backend for eager Polars ``DataFrame`` objects."""

    def supports(self, dataframe: object) -> bool:
        try:
            import polars as pl
        except ImportError:
            return False
        return isinstance(dataframe, pl.DataFrame)

    def copy(self, dataframe: Any) -> Any:
        return dataframe.clone()

    def row_count(self, dataframe: Any) -> int:
        return dataframe.height

    def read_row(
        self,
        dataframe: Any,
        position: int,
        columns: Iterable[str],
    ) -> Mapping[str, object]:
        return dataframe.select(list(columns)).row(position, named=True)

    def write_batch(
        self,
        dataframe: Any,
        updates: Mapping[int, Mapping[str, object]],
    ) -> Any:
        import polars as pl

        output_columns = dict.fromkeys(
            column for result in updates.values() for column in result
        )
        expressions = []
        for column in output_columns:
            if column in dataframe.columns:
                values = dataframe.get_column(column).to_list()
            else:
                values = [None] * dataframe.height

            for position, result in updates.items():
                if column in result:
                    values[position] = result[column]
            expressions.append(pl.Series(column, values, strict=False))

        if expressions:
            dataframe = dataframe.with_columns(expressions)
        return dataframe
