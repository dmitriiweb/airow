# Airow

**AI-powered DataFrame processing made simple**

Airow is a Python library that combines the power of pandas DataFrames with AI models to process structured data at scale. Built on top of `pydantic-ai`, it provides type-safe, async processing of DataFrames using any AI model.

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
# Using pip
pip install airow

# Using uv (recommended)
uv add airow

# Using conda
conda install -c conda-forge airow
```

## Quick Start

```python
import pandas as pd
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from airow import Airow, OutputColumn
import asyncio

async def main():
    # Setup your AI model
    model = OpenAIChatModel(
        model_name="llama3.2:latest",
        provider=OllamaProvider(base_url="http://localhost:11434/v1"),
    )
    # or use strings:
    model = "openai:gpt-5"
    model = "anthropic:claude-sonnet-4-0"
    
    # Create Airow instance
    airow = Airow(
        model=model,
        system_prompt="You are an expert in wine tasting and selection.",
    )
    
    # Load your data
    df = pd.read_csv("wine_data.csv")

    output_columns = [
        OutputColumn(name="sentiment", type=str, description="Positive, negative, or neutral sentiment"),
        OutputColumn(name="confidence", type=float, description="Confidence score between 0 and 1"),
        OutputColumn(name="keywords", type=list, description="List of key terms extracted"),
    ]
    
    # Process with AI
    result_df = await airow.run(
        df,
        prompt="Analyze the wine description and provide sentiment analysis, confidence score, and extract key terms.",
        input_columns=["description"],
        output_columns=output_columns,
        show_progress=True,
    )
    
    print(result_df.head())

if __name__ == "__main__":
    asyncio.run(main())
```
