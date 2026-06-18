from typing import Any

import pandas as pd
import polars as pl
import pytest

from airow import Airow, OutputColumn
from tests.test_backends import ListBackend


class StubAgent:
    def __init__(self, results: list[dict[str, object]]):
        self.results = iter(results)
        self.prompts: list[str] = []

    async def run(self, prompt: str, output_columns: Any) -> dict[str, object]:
        self.prompts.append(prompt)
        return next(self.results)


def make_airow(
    results: list[dict[str, object]],
    *,
    backend: ListBackend | None = None,
    batch_size: int = 1,
) -> tuple[Airow, StubAgent]:
    airow = Airow(
        model=None,  # type: ignore[arg-type]
        system_prompt="test",
        backend=backend,
        batch_size=batch_size,
    )
    agent = StubAgent(results)
    airow.agent = agent  # type: ignore[assignment]
    return airow, agent


@pytest.mark.asyncio
async def test_run_pandas_returns_copy_with_results_for_non_default_index():
    dataframe = pd.DataFrame({"input": ["a", "b"]}, index=[10, 20])
    airow, _ = make_airow([{"output": "A"}, {"output": "B"}], batch_size=2)

    result = await airow.run(
        dataframe,
        prompt="Transform",
        input_columns=["input"],
        output_columns=OutputColumn("output", str, "Result"),
    )

    assert result["output"].to_list() == ["A", "B"]
    assert "output" not in dataframe.columns


@pytest.mark.asyncio
async def test_run_polars_returns_frame_with_results():
    dataframe = pl.DataFrame({"input": ["a", "b"]})
    airow, _ = make_airow([{"output": "A"}, {"output": "B"}], batch_size=2)

    result = await airow.run(
        dataframe,
        prompt="Transform",
        input_columns=["input"],
        output_columns=OutputColumn("output", str, "Result"),
    )

    assert result["output"].to_list() == ["A", "B"]
    assert dataframe.columns == ["input"]


@pytest.mark.asyncio
async def test_run_formats_selected_values_in_prompt():
    airow, agent = make_airow([{"output": "A"}], backend=ListBackend())

    await airow.run(
        [{"name": "Ada", "ignored": 1}],
        prompt="Describe",
        input_columns=["name"],
        output_columns=OutputColumn("output", str, "Result"),
    )

    assert agent.prompts == ["Describe\n\nColumn: name, Value: Ada"]


@pytest.mark.asyncio
async def test_run_accepts_custom_backend():
    dataframe = [{"input": "a"}]
    airow, _ = make_airow([{"output": "A"}], backend=ListBackend())

    result = await airow.run(
        dataframe,
        prompt="Transform",
        input_columns=["input"],
        output_columns=OutputColumn("output", str, "Result"),
    )

    assert result == [{"input": "a", "output": "A"}]
    assert dataframe == [{"input": "a"}]


@pytest.mark.asyncio
async def test_run_rejects_incompatible_explicit_backend():
    airow, _ = make_airow([], backend=ListBackend())

    with pytest.raises(TypeError, match="ListBackend does not support"):
        await airow.run(
            pd.DataFrame({"input": ["a"]}),
            prompt="Transform",
            input_columns=["input"],
            output_columns=OutputColumn("output", str, "Result"),
        )
