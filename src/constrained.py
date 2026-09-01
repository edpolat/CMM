from pydantic import BaseModel, Field


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


def valid_token_ids(root: TrieNode, so_far: str, id_to_token: dict[int, str]
                    ) -> list[int]:
    valid = []
    for token_id, token_str in id_to_token.items():
        candidate = so_far + token_str
        if is_valid_prefix(root, candidate):
            valid.append(token_id)
    return valid