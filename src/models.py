from pydantic import BaseModel
from enum import Enum


class ParamType(str, Enum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"


class ParameterSchema(BaseModel):
    type: ParamType


class FunctionDefinition(BaseModel):

    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ParameterSchema


class PromptEntry(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, bool | int | float | str]
