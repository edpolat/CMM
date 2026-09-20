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


if __name__ == "__main__":
    good = {"name": "fn_x", "description": "d",
            "parameters": {"a": {"type": "number"}},
            "returns": {"type": "string"}}
    fd = FunctionDefinition.model_validate(good)

    from pydantic import ValidationError
    try:
        FunctionDefinition.model_validate({"name": "fn_x"})
    except ValidationError:
        print("eksik alan yakalandı")

    r = FunctionCallResult(prompt="p", name="fn_x", parameters={"a": 2.0})
    print("models ok")