# Airow

**AI-powered DataFrame processing made simple**

Airow is a Python library that combines pandas or Polars DataFrames with AI models to process structured data at scale. Built on top of `pydantic-ai`, it provides type-safe, async processing of DataFrames using any AI model.

## Features

- 🚀 **Async processing** with batch support for high performance
- 🔒 **Type-safe outputs** using Pydantic models
- 📊 **Progress tracking** with built-in progress bars
- 🔄 **Automatic retries** with configurable retry logic
- 🤖 **Flexible AI models** - works with OpenAI, Ollama, Anthropic, and more
- ⚡ **Parallel processing** within batches for maximum throughput
- 📝 **Structured outputs** with defined schemas and validation

## Installation

```bash
# Core library without a DataFrame backend
pip install airow

# pandas only
pip install "airow[pandas]"

# Polars only
pip install "airow[polars]"

# pandas and Polars
pip install "airow[all]"
```

## Pandas example

Install Airow with the pandas backend:

```bash
pip install "airow[pandas]"
```

The examples use Pydantic AI's `openai:gpt-5` model string, so configure the
corresponding provider credentials before running them.

```python
import asyncio

import pandas as pd

from airow import Airow, OutputColumn


async def main():
    df = pd.DataFrame(
        {
            "description": [
                "Bright citrus flavors with a crisp finish.",
                "Rich dark fruit with firm tannins.",
            ]
        }
    )

    airow = Airow(
        model="openai:gpt-5",
        system_prompt="You are an expert in wine tasting and selection.",
        batch_size=2,
    )

    output_columns = [
        OutputColumn(
            name="summary",
            type=str,
            description="A concise summary of the wine",
        ),
        OutputColumn(
            name="style",
            type=str,
            description="The inferred wine style",
        ),
    ]

    result_df = await airow.run(
        df,
        prompt="Analyze this wine description.",
        input_columns=["description"],
        output_columns=output_columns,
        show_progress=True,
    )

    # result_df is a pandas.DataFrame; df is unchanged.
    print(result_df.head())


if __name__ == "__main__":
    asyncio.run(main())
```

Airow detects pandas automatically and returns a new `pandas.DataFrame`.

## Polars example

Install Airow with the Polars backend:

```bash
pip install "airow[polars]"
```

```python
import asyncio

import polars as pl

from airow import Airow, OutputColumn


async def main():
    df = pl.DataFrame(
        {
            "description": [
                "Bright citrus flavors with a crisp finish.",
                "Rich dark fruit with firm tannins.",
            ]
        }
    )

    airow = Airow(
        model="openai:gpt-5",
        system_prompt="You are an expert in wine tasting and selection.",
        batch_size=2,
    )

    output_columns = [
        OutputColumn(
            name="summary",
            type=str,
            description="A concise summary of the wine",
        ),
        OutputColumn(
            name="style",
            type=str,
            description="The inferred wine style",
        ),
    ]

    result_df = await airow.run(
        df,
        prompt="Analyze this wine description.",
        input_columns=["description"],
        output_columns=output_columns,
        show_progress=True,
    )

    # result_df is a polars.DataFrame; df is unchanged.
    print(result_df.head())


if __name__ == "__main__":
    asyncio.run(main())
```

Airow detects eager Polars DataFrames automatically and returns a new
`polars.DataFrame`. LazyFrames are not currently supported.

## Custom backends

Custom dataframe implementations can subclass `DataFrameBackend` and pass an
instance explicitly:

```python
from airow import Airow, DataFrameBackend

backend: DataFrameBackend = MyDataFrameBackend()
airow = Airow(
    model="openai:gpt-5",
    system_prompt="You are a data processing assistant.",
    backend=backend,
)
```
