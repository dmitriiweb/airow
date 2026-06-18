"""High-level API to run LLMs over DataFrame rows in batches."""

import asyncio
from typing import Any, Iterable, TypeVar, cast

from pydantic_ai.models import Model
from tqdm import tqdm

from . import schemas
from .agent import AirowAgent
from .backends import DataFrameBackend, resolve_backend

DataFrameT = TypeVar("DataFrameT")


class Airow:
    """Apply an LLM to each row of a DataFrame and write results.

    Uses `AirowAgent` internally and supports parallel row processing per batch.
    """

    def __init__(
        self,
        *,
        model: Model | str,
        system_prompt: str,
        batch_size: int = 1,
        retries: int = 3,
        backend: DataFrameBackend | None = None,
    ):
        """Configure the runner.

        Args:
            model: The pydantic-ai model to use.
            system_prompt: System prompt applied to each request.
            batch_size: Number of DataFrame rows to process concurrently.
            retries: Number of retries for a run.
            backend: Optional custom dataframe backend. When omitted, a built-in
                backend is selected from the dataframe passed to :meth:`run`.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.batch_size = batch_size
        self.agent = AirowAgent(self.model, self.system_prompt, retries)
        self.backend = backend

    async def run(
        self,
        df: DataFrameT,
        *,
        prompt: str,
        input_columns: Iterable[str],
        output_columns: schemas.OutputColumn | Iterable[schemas.OutputColumn],
        show_progress: bool = False,
    ) -> DataFrameT:
        """Run the model across the DataFrame and return results.

        For each row, the values from `input_columns` are appended to `prompt`
        as labeled lines and the model is asked to produce `output_columns`.
        Results are written into the original DataFrame.

        Args:
            df: Input DataFrame.
            prompt: Base prompt text provided to the model.
            input_columns: Columns whose values are passed as context to the model.
            output_columns: One or more output column specifications.
            show_progress: Whether to display a progress bar.

        Returns:
            The input DataFrame with new output columns populated.
        """
        if isinstance(output_columns, schemas.OutputColumn):
            output_columns = [output_columns]

        input_columns = list(input_columns)
        backend = self.backend or resolve_backend(df)
        if self.backend is not None and not backend.supports(df):
            raise TypeError(
                f"{type(backend).__name__} does not support "
                f"{type(df).__module__}.{type(df).__qualname__}."
            )

        result_df: Any = backend.copy(df)

        total_rows = backend.row_count(result_df)
        batch_ranges: Iterable[tuple[int, int]] = [
            (i, i + self.batch_size) for i in range(0, total_rows, self.batch_size)
        ]
        if show_progress:
            batch_ranges = tqdm(batch_ranges)

        for batch_range in batch_ranges:
            tasks = []
            row_indices = []

            start_idx = batch_range[0]
            end_idx = min(batch_range[1], total_rows)

            for row_idx in range(start_idx, end_idx):
                input_data = backend.read_row(
                    result_df,
                    row_idx,
                    input_columns,
                )
                input_data_str = "\n".join(
                    f"Column: {key}, Value: {value}"
                    for key, value in input_data.items()
                )
                full_prompt = f"{prompt}\n\n{input_data_str}"
                task = self.agent.run(full_prompt, output_columns)
                tasks.append(task)
                row_indices.append(row_idx)

            results = await asyncio.gather(*tasks)
            updates = dict(zip(row_indices, results))
            result_df = backend.write_batch(result_df, updates)

        return cast(DataFrameT, result_df)
