from typing import Iterable

from pydantic import BaseModel, Field, create_model
from pydantic_ai import Agent
from pydantic_ai.models import Model
from loguru import logger

from . import schemas


class AirowAgent:
    def __init__(
        self,
        model: Model,
        system_prompt: str,
        retries: int = 3,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.agent = Agent(model=model, system_prompt=self.system_prompt, retries=retries)

    async def run(
        self,
        prompt: str,
        output_columns: Iterable[schemas.OutputColumn],
    ) -> dict[str, object]:
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
        fields = {
            col.name: (col.type, Field(..., description=col.description))
            for col in output_columns
        }
        return create_model("OutputColumns", **fields)
