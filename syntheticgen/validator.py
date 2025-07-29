# Add to validator.py
from pydantic import BaseModel, Field
from typing import Literal, Union

class FieldConstraint(BaseModel):
    min: Union[int, float, None] = None
    max: Union[int, float, None] = None
    options: list[str] = []
    pattern: str = ""
    date_direction: Literal["past", "future", "any"] = "any"
    date_format: str = "%Y-%m-%d"

class FieldDefinition(BaseModel):
    name: str
    type: Literal["int", "float", "category", "date", "string", "bool"]
    constraints: FieldConstraint = Field(default_factory=FieldConstraint)

def parse_schema(raw_schema: dict) -> list[FieldDefinition]:
    # Implementation that converts user input to structured format