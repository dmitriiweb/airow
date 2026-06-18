from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd
import polars as pl
import pytest

from airow.backends import (
    DataFrameBackend,
    PandasBackend,
    PolarsBackend,
    resolve_backend,
)


@pytest.mark.parametrize(
    ("dataframe", "backend_type"),
    [
        (pd.DataFrame({"value": [1]}), PandasBackend),
        (pl.DataFrame({"value": [1]}), PolarsBackend),
    ],
)
def test_resolve_backend_known_dataframe_returns_matching_backend(
    dataframe: object,
    backend_type: type[DataFrameBackend],
):
    backend = resolve_backend(dataframe)

    assert isinstance(backend, backend_type)


def test_resolve_backend_unknown_dataframe_raises_actionable_error():
    with pytest.raises(TypeError, match=r"airow\[pandas\].*airow\[polars\]"):
        resolve_backend(object())


def test_pandas_backend_writes_by_position_with_non_default_index():
    dataframe = pd.DataFrame({"input": [10, 20]}, index=[5, 9])
    backend = PandasBackend()

    result = backend.write_batch(dataframe, {1: {"output": "done"}})

    assert result.loc[9, "output"] == "done"
    assert pd.isna(result.loc[5, "output"])


def test_pandas_backend_copy_is_independent():
    dataframe = pd.DataFrame({"input": [10]})
    backend = PandasBackend()

    result = backend.write_batch(backend.copy(dataframe), {0: {"output": "done"}})

    assert "output" not in dataframe.columns
    assert result.loc[0, "output"] == "done"


def test_polars_backend_adds_output_column_without_mutating_input():
    dataframe = pl.DataFrame({"input": [10, 20]})
    backend = PolarsBackend()

    result = backend.write_batch(
        backend.copy(dataframe),
        {0: {"output": "first"}, 1: {"output": "second"}},
    )

    assert dataframe.columns == ["input"]
    assert result["output"].to_list() == ["first", "second"]


def test_polars_backend_preserves_unmodified_values_when_replacing_column():
    dataframe = pl.DataFrame({"input": [10, 20], "output": ["old", "keep"]})
    backend = PolarsBackend()

    result = backend.write_batch(dataframe, {0: {"output": "new"}})

    assert result["output"].to_list() == ["new", "keep"]


def test_backends_read_selected_columns_by_position():
    pandas_row = PandasBackend().read_row(
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        1,
        ["b"],
    )
    polars_row = PolarsBackend().read_row(
        pl.DataFrame({"a": [1, 2], "b": [3, 4]}),
        1,
        ["b"],
    )

    assert pandas_row == {"b": 4}
    assert polars_row == {"b": 4}


@pytest.mark.parametrize(
    ("dataframe", "backend"),
    [
        (pd.DataFrame({"input": [1]}), PandasBackend()),
        (pl.DataFrame({"input": [1]}), PolarsBackend()),
    ],
)
def test_backend_adds_output_columns_in_result_order(
    dataframe: object,
    backend: DataFrameBackend,
):
    result = backend.write_batch(
        dataframe,
        {0: {"second": 2, "first": 1}},
    )

    assert list(result.columns) == ["input", "second", "first"]


class ListBackend(DataFrameBackend):
    def supports(self, dataframe: object) -> bool:
        return isinstance(dataframe, list)

    def copy(self, dataframe: Any) -> Any:
        return [row.copy() for row in dataframe]

    def row_count(self, dataframe: Any) -> int:
        return len(dataframe)

    def read_row(
        self,
        dataframe: Any,
        position: int,
        columns: Iterable[str],
    ) -> Mapping[str, object]:
        return {column: dataframe[position][column] for column in columns}

    def write_batch(
        self,
        dataframe: Any,
        updates: Mapping[int, Mapping[str, object]],
    ) -> Any:
        for position, values in updates.items():
            dataframe[position].update(values)
        return dataframe
