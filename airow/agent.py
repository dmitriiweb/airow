from typing import Iterable

import pandas as pd
from pydantic import BaseModel, Field, create_model
from pydantic_ai import Agent
from pydantic_ai.models import Model

from . import schemas


class AirowAgent:
    def __init__(
        self,
        model: Model,
        system_prompt: str,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.agent = Agent(model=model, system_prompt=self.system_prompt)

    async def run(
        self,
        input_data: dict[str, object],
        prompt: str,
        output_columns: Iterable[schemas.OutputColumn],
    ) -> dict[str, object]:
        output_columns_fields = self.build_agent_output_type(output_columns)
        return output_columns_fields.model_dump()

    def build_agent_output_type(
        self,
        output_columns: Iterable[schemas.OutputColumn],
    ) -> type[BaseModel]:
        fields = {
            col.name: (col.type, Field(..., description=col.description))
            for col in output_columns
        }
        return create_model("OutputColumns", **fields)
