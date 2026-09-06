from llm_sdk import Small_LLM_Model

from .constrained import (
    TrieNode,
    insert,
    generate_function_name,
    generate_number,
    generate_string,
    generate_boolean,
)
from .models import (
    FunctionCallResult,
    FunctionDefinition,
    ParamType,
    PromptEntry,
)
from .tokenizer_utils import encode_prompt


def build_prompt(func_defs: list[FunctionDefinition], user_prompt: str
                 ) -> str:
    lines = []
    for definition in func_defs:
        params = ", ".join(
            f"{name}: {schema.type.value}"
            for name, schema in definition.parameters.items()
        )
        lines.append(
            f"{definition.name}({params}): {definition.description}")
    text = "\n".join(lines)
    text += "\n" + user_prompt + "\nFunction name:"
    return text


def build_function_trie(func_defs: list[FunctionDefinition]) -> TrieNode:
    root = TrieNode()
    for definition in func_defs:
        insert(root, definition.name)
    return root


def find_function(func_defs: list[FunctionDefinition], name: str
                  ) -> FunctionDefinition:
    for definition in func_defs:
        if definition.name == name:
            return definition
    raise ValueError(f"Model generated an unknown function name: {name!r}")


def generate_call(
    model: Small_LLM_Model,
    func_defs: list[FunctionDefinition],
    id_to_token: dict[int, str],
    entry: PromptEntry,
) -> FunctionCallResult:
    root: TrieNode = build_function_trie(func_defs)
    promted = build_prompt(func_defs, entry.prompt)
    input_ids = encode_prompt(model, promted)
    func_name, dec = generate_function_name(
        model, input_ids, id_to_token, root)
    func_is: FunctionDefinition = find_function(func_defs, func_name)
    prompt_text = promted + " " + func_is.name
    parameters: dict[str, bool | int | float | str] = {}
    for param_name, schema in func_is.parameters.items():
        if schema.type == ParamType.STRING:
            param_prompt = (
                f'{prompt_text}\nRequest: "{entry.prompt}"'
                f'\nParameter {param_name} (string) value: "'
            )
            param_ids = encode_prompt(model, param_prompt)
            text, _ = generate_string(model, param_ids, id_to_token)
            value: bool | int | float | str = text

        elif schema.type == ParamType.NUMBER:
            param_prompt = (
                f'{prompt_text}\nRequest: "{entry.prompt}"'
                f'\nParameter {param_name} (number) value:'
            )
            param_ids = encode_prompt(model, param_prompt)
            text, _ = generate_number(model, param_ids, id_to_token)
            value = float(text)

        elif schema.type == ParamType.BOOLEAN:
            param_prompt = (
                f'{prompt_text}\nRequest: "{entry.prompt}"'
                f'\nParameter {param_name} (boolean) value:'
            )
            param_ids = encode_prompt(model, param_prompt)
            text, _ = generate_boolean(model, param_ids, id_to_token)
            value = text == "true"
        parameters[param_name] = value
        prompt_text += f"\nParameter {param_name}: {value}"
    return FunctionCallResult(prompt=entry.prompt, name=func_is.name,
                              parameters=parameters)
