import argparse
import json
import sys
from llm_sdk import Small_LLM_Model
from .generator import generate_call
from .io_utils import func_def_reader, prompt_reader
from .tokenizer_utils import build_id_to_token
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        model = Small_LLM_Model()
    except Exception as e:
        print(f"Failed to load the model: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        func_defs = func_def_reader(args.functions_definition)
        prompts = prompt_reader(args.input)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    id_to_token = build_id_to_token(model)
    results = []
    for entry in prompts:
        try:
            result = generate_call(model, func_defs, id_to_token, entry)
        except Exception as e:
            print(f"Skipping prompt {entry.prompt!r}: {e}", file=sys.stderr)
            continue
        results.append(result.model_dump())
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
