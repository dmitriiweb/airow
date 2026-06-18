"""pandas dataframe backend."""

from collections.abc import Iterable, Mapping
from typing import Any

from .base import DataFrameBackend


class PandasBackend(DataFrameBackend):
    """Backend for pandas ``DataFrame`` objects."""

    def supports(self, dataframe: object) -> bool:
        try:
            import pandas as pd
        except ImportError:
            return False
        return isinstance(dataframe, pd.DataFrame)

    def copy(self, dataframe: Any) -> Any:
        return dataframe.copy()

    def row_count(self, dataframe: Any) -> int:
        return len(dataframe)

    def read_row(
        self,
        dataframe: Any,
        position: int,
        columns: Iterable[str],
    ) -> Mapping[str, object]:
        row = dataframe.iloc[position]
        return {column: row[column] for column in columns}

    def write_batch(
        self,
        dataframe: Any,
        updates: Mapping[int, Mapping[str, object]],
    ) -> Any:
        output_columns = dict.fromkeys(
            column for result in updates.values() for column in result
        )
        for column in output_columns:
            if column not in dataframe.columns:
                dataframe[column] = None

        for position, result in updates.items():
            for column, value in result.items():
                column_position = dataframe.columns.get_loc(column)
                dataframe.iat[position, column_position] = value
        return dataframe
