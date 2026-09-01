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
    parameters: dict[str, bool | int | str]

"""ParamType — izin verilen tip isimleri (number/string/boolean)
ParameterSchema — {"type": ...} şeklini temsil ediyor, hem parameters değerleri hem de returns için kullanılıyor
FunctionDefinition — functions_definition.json'daki her fonksiyon kaydı
PromptEntry — function_calling_tests.json'daki her girdi
FunctionCallResult — senin üreteceğin çıktının şekli"""