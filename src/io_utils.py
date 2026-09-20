import json

from pydantic import ValidationError

from .models import FunctionDefinition, PromptEntry


def func_def_reader(file_name: str) -> list[FunctionDefinition]:
    try:
        with open(file_name) as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise ValueError(f"File not found: {file_name}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format ({file_name}): {e}") from e

    function = []
    try:
        for func in data:
            transformation_data = FunctionDefinition.model_validate(func)
            function.append(transformation_data)
    except ValidationError as e:
        raise ValueError(
            f"Function definition does not match schema ({file_name}): {e}"
            ) from e

    return function


def prompt_reader(file_name: str) -> list[PromptEntry]:
    try:
        with open(file_name) as f:
            prompt_list = json.load(f)
    except FileNotFoundError as e:
        raise ValueError(f"File not found: {file_name}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format ({file_name}): {e}") from e

    prompts = []
    try:
        for prompt in prompt_list:
            transformation_pmt = PromptEntry.model_validate(prompt)
            prompts.append(transformation_pmt)
    except ValidationError as e:
        raise ValueError(
            f"Prompt entry does not match schema ({file_name}): {e}") from e

    return prompts

if __name__ == "__main__":
    import json, tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([{"prompt": "Greet john"}], f)
        path = f.name

    for bad in ["yok.json"]:
        try:
            prompt_reader(bad)
        except ValueError as e:
            print("beklenen hata:", e)

    with open(path, "w") as f:
        f.write("{bozuk json")
    try:
        prompt_reader(path)
    except ValueError as e:
        print("beklenen hata:", e)
    os.remove(path)
    print("io_utils ok")