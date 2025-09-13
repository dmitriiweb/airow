"""Agent utilities for executing pydantic-ai models with structured outputs."""

from typing import Iterable

from pydantic import BaseModel, Field, create_model
from pydantic_ai import Agent
from pydantic_ai.models import Model
from loguru import logger

from . import schemas


class AirowAgent:
    """Wrapper around `pydantic_ai.Agent` that builds structured output models."""
    def __init__(
        self,
        model: Model | str,
        system_prompt: str,
        retries: int = 3,
    ):
        """Initialize the agent.

        Args:
            model: The underlying model used by pydantic-ai.
            system_prompt: System prompt applied to all runs.
            retries: Number of retries for a run.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.agent = Agent(model=model, system_prompt=self.system_prompt, retries=retries)

    async def run(
        self,
        prompt: str,
        output_columns: Iterable[schemas.OutputColumn],
    ) -> dict[str, object]:
        """Run the agent with the given prompt and expected outputs.

        Args:
            prompt: User prompt to pass to the model.
            output_columns: Iterable of expected output columns specifications.

        Returns:
            A dictionary mapping output column names to parsed values. Returns
            an empty dictionary when the underlying model call fails.
        """
        output_columns_fields = self.build_agent_output_type(output_columns)
        try:
            result = await self.agent.run(prompt, output_type=output_columns_fields)
        except Exception as e:
            logger.error(f"{e=}")
            return {}
        return result.output.model_dump()

    def build_agent_output_type(
        self,
        output_columns: Iterable[schemas.OutputColumn],
    ) -> type[BaseModel]:
        """Create a `pydantic.BaseModel` for the requested output columns.

        Args:
            output_columns: Iterable of output column specifications.

        Returns:
            A dynamically created `BaseModel` subclass with fields per column.
        """
        fields = {
            col.name: (col.type, Field(..., description=col.description))
            for col in output_columns
        }
        return create_model("OutputColumns", **fields)
