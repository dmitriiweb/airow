"""Data structures for declaring model outputs."""

from dataclasses import dataclass
from typing import Any, Type

from pydantic import BaseModel


@dataclass
class OutputColumn:
    """
    Output column for the AI model.

    Args:
        name: Name of the output column.
        type: Type of the output column.

    Examples:
        >>> OutputColumn(name="output_column", type=str)
        >>> OutputColumn(name="output_column", type=int)
    """

    name: str
    type: Type[Any]
    description: str
