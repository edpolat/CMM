from pydantic import BaseModel, Field
from llm_sdk import Small_LLM_Model


class TrieNode(BaseModel):
    children: dict[str, "TrieNode"] = Field(default_factory=dict)
    is_end: bool = False


def insert(root: TrieNode, word: str) -> None:
    current = root
    for c in word:
        if c not in current.children:
            current.children[c] = TrieNode()
        current = current.children[c]
    current.is_end = True


def is_valid_prefix(root: TrieNode, text: str) -> bool:
    current = root
    for c in text:
        if c not in current.children:
            return False
        current = current.children[c]
    return True


def is_complete_word(root: TrieNode, text: str) -> bool:
    current = root
    for c in text:
        if c not in current.children:
            return False
        current = current.children[c]
    return current.is_end


def has_children(root: TrieNode, text: str) -> bool:
    current = root
    for c in text:
        if c not in current.children:
            return False
        current = current.children[c]
    return len(current.children) > 0


def valid_token_ids(root: TrieNode, text: str, id_to_token: dict[int, str]
                    ) -> list[int]:
    valid = []
    for token_id, token_str in id_to_token.items():
        candidate = text + token_str
        if is_valid_prefix(root, candidate):
            valid.append(token_id)
    return valid


def mask_logits(logits: list[float], valid_ids: list[int]) -> list[float]:
    valid_set = set(valid_ids)
    valid_logit: list[float] = []
    for i, score in enumerate(logits):
        if i in valid_set:
            valid_logit.append(score)
        else:
            valid_logit.append(float("-inf"))
    return valid_logit


def generate_function_name(
    model: Small_LLM_Model,
    input_ids: list[int],
    id_to_token: dict[int, str],
    root: TrieNode,
) -> tuple[str, list[int]]:
    text: str = ""
    generated_ids: list[int] = []
    while True:
        if is_complete_word(root, text) and not has_children(root, text):
            break
        logit = model.get_logits_from_input_ids(input_ids + generated_ids)
        if is_complete_word(root, text):
            raw_best = logit.index(max(logit))
            raw_best_str = id_to_token.get(raw_best, "")
            if not is_valid_prefix(root, text + raw_best_str):
                break
        valid_tok_id = valid_token_ids(root, text, id_to_token)
        valid_logit = mask_logits(logit, valid_tok_id)
        if not valid_tok_id or max(valid_logit) == float("-inf"):
            break
        best = valid_logit.index(max(valid_logit))
        generated_ids.append(best)
        text += id_to_token[best]
    return text, generated_ids


def is_valid_number_prefix(text: str) -> bool:
    valid_nb = "0123456789-."
    dot_flag = 0
    minus_flag = 0
    i = 0
    for c in text:
        if c == "." and (i == 0 or not text[i - 1].isdigit()):
            return False
        if c not in valid_nb:
            return False
        if c == "-":
            minus_flag += 1
        if c == ".":
            dot_flag += 1
        i += 1
    if dot_flag > 1 or minus_flag > 1:
        return False
    if minus_flag == 1 and not text.startswith("-"):
        return False
    return True


def strip_leading_marker(token_str: str) -> str:
    return token_str[1:] if token_str.startswith("Ġ") else token_str


def number_candidate_ids(id_to_token: dict[int, str]) -> list[int]:
    allowed = set("0123456789-.")
    candidates = []
    for t_id, t_str in id_to_token.items():
        cleaned = strip_leading_marker(t_str)
        if cleaned and all(c in allowed for c in cleaned):
            candidates.append(t_id)
    return candidates


def valid_number_token_ids(
    text: str, id_to_token: dict[int, str], candidates: list[int]
) -> list[int]:
    valid = []
    for t_id in candidates:
        cleaned = strip_leading_marker(id_to_token[t_id])
        if is_valid_number_prefix(text + cleaned):
            valid.append(t_id)
    return valid


def generate_number(
    model: Small_LLM_Model,
    input_ids: list[int],
    id_to_token: dict[int, str],
    max_tokens: int = 10,
) -> tuple[str, list[int]]:
    text: str = ""
    generated_ids: list[int] = []
    candidates = number_candidate_ids(id_to_token)
    while len(generated_ids) < max_tokens:
        raw_logits = model.get_logits_from_input_ids(input_ids + generated_ids)
        raw_best = raw_logits.index(max(raw_logits))
        raw_best_str = id_to_token.get(raw_best, "")
        has_digit = any(c.isdigit() for c in text)
        if has_digit and not is_valid_number_prefix(text + raw_best_str):
            break

        valid_ids = valid_number_token_ids(text, id_to_token, candidates)
        masked = mask_logits(raw_logits, valid_ids)
        if not valid_ids or max(masked) == float("-inf"):
            break
        best = masked.index(max(masked))
    generated_ids.append(best)
    text += strip_leading_marker(id_to_token[best])
    text = text.rstrip(".")
    if text in ("", "-"):
        text = "0"
    return text, generated_ids


def is_valid_string_token(token_str: str) -> bool:
    for c in token_str:
        if not c.isprintable():
            return False
    return True


def valid_string_token_ids(id_to_token: dict[int, str]) -> list[int]:
    valid = []
    for t_id, t_str in id_to_token.items():
        if is_valid_string_token(t_str):
            valid.append(t_id)
    return valid


def generate_string(
    model: Small_LLM_Model,
    input_ids: list[int],
    id_to_token: dict[int, str],
    max_tokens: int = 30,
) -> tuple[str, list[int]]:
    text: str = ""
    generated_ids: list[int] = []
    valid_ids = valid_string_token_ids(id_to_token)
    while len(generated_ids) < max_tokens:
        raw_logits = model.get_logits_from_input_ids(input_ids + generated_ids)
        masked = mask_logits(raw_logits, valid_ids)
        if max(masked) == float("-inf"):
            break
        best = masked.index(max(masked))
        best_str = id_to_token[best].replace("Ġ", " ")
        generated_ids.append(best)
        if '"' in best_str:
            text += best_str.split('"')[0]
            break
        text += best_str
    return text.strip(), generated_ids


def generate_boolean(
    model: Small_LLM_Model,
    input_ids: list[int],
    id_to_token: dict[int, str],
) -> tuple[str, list[int]]:
    root = TrieNode()
    insert(root, "true")
    insert(root, "false")
    return generate_function_name(model, input_ids, id_to_token, root)
