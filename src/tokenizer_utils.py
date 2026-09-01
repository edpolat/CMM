from llm_sdk import Small_LLM_Model
import json

def build_id_to_token(model: Small_LLM_Model) -> dict[int, str]:
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path) as token_file:
        token_id_dict = json.load(token_file)
    id_token_dict: dict[int, str] = {}
    for key, value in token_id_dict.items():
        id_token_dict[value] = key
    return id_token_dict


def encode_prompt(model: Small_LLM_Model, text: str) -> list[int]:
    tensor_list = model.encode(text)
    tensor_to_list: list[int] = [int(x) for x in tensor_list[0]]
    return tensor_to_list


def decode_ids(id_to_token: dict[int, str], token_ids: list[int]) -> str:
    decoded_str: str = ""
    for id in token_ids:
        decoded_str += id_to_token[id]
    decoded_str = decoded_str.replace("Ġ", " ")
    return decoded_str