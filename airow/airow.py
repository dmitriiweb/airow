import asyncio
from typing import Iterable

import pandas as pd
from pydantic_ai.models import Model
from tqdm import tqdm

from . import schemas
from .agent import AirowAgent


class Airow:
    def __init__(
        self,
        *,
        model: Model,
        system_prompt: str,
        batch_size: int = 1,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.batch_size = batch_size
        self.agent = AirowAgent(self.model, self.system_prompt)

    async def run(
        self,
        df: pd.DataFrame,
        *,
        prompt: str,
        input_columns: Iterable[str],
        output_columns: schemas.OutputColumn | Iterable[schemas.OutputColumn],
        show_progress: bool = False,
    ) -> pd.DataFrame:
        if isinstance(output_columns, schemas.OutputColumn):
            output_columns = [output_columns]

        # Convert to list for easier handling
        input_columns = list(input_columns)

        # Split dataframe into batches
        total_rows = df.shape[0]
        batche_ranges = [
            (i, i + self.batch_size)
            for i in range(0, total_rows, self.batch_size)
        ]
        if show_progress:
            batche_ranges = tqdm(batche_ranges)

        for batch_range in batche_ranges:
            # Process each row in the batch in parallel
            tasks = []
            row_indices = []
            batch = df.iloc[batch_range[0] : batch_range[1]]

            for idx, row in batch.iterrows():
                input_data = {col: row[col] for col in input_columns}
                input_data_str = "\n".join([f"Column: {k}, Value: {v}" for k, v in input_data.items()])
                prompt = f"{prompt}\n\n{input_data_str}"
                task = self.agent.run(prompt, output_columns)
                tasks.append(task)
                row_indices.append(idx)

            # Run all tasks in parallel
            results = await asyncio.gather(*tasks)

            # Add results to dataframe
            for i, result in enumerate(results):
                row_idx = row_indices[i]
                for col_name, value in result.items():
                    df.loc[row_idx, col_name] = value

        return df
