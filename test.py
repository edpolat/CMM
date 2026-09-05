"""from llm_sdk import Small_LLM_Model
model = Small_LLM_Model()
def unconstrained_generate(prompt, n=200):
    ids = model.encode(prompt)[0].tolist()
    out = []
    for _ in range(n):
        logits = model.get_logits_from_input_ids(ids + out)
        out.append(logits.index(max(logits)))
    return model.decode(out)
prompt = "<|im_start|>user\nWhat is the sum of 2 and 3?<|im_end|>\n<|im_start|>assistant\n"
print(unconstrained_generate(prompt, n=15))
"""


"""doğru çalışan test-func için
from src.io_utils import func_def_reader
functions = func_def_reader("data/input/functions_definition.json")
print(len(functions))       # 5 olmalı
print(functions[0].name)    # "fn_add_numbers" olmalı
print(functions[0].parameters["a"].type)  # ParamType.NUMBER olmalı
"""



"""doğru çalışan test-promt için
from src.io_utils import prompt_reader
prompts = prompt_reader("data/input/function_calling_tests.json")
print(len(prompts))
print(prompts[0].prompt)
"""



"""hata testi
from src.io_utils import func_def_reader
func_def_reader("data/input/nonexistent.json")
"""

"""
LLM'in token to id olan listesini ters çevirebilmiş miyim testi.
from llm_sdk import Small_LLM_Model
from src.tokenizer_utils import build_id_to_token
import json
model = Small_LLM_Model()
id_to_token = build_id_to_token(model)
path = model.get_path_to_vocab_file()
f = open(path)
g = json.load(f)
print(len(g))
print(g["!"])
print(g[","])
print(len(id_to_token))     # 151643 olmalı
print(id_to_token[0])       # '!' olmalı
print(id_to_token[11])      # ',' olmalı
"""

""""
encode and decode functions tester
from llm_sdk import Small_LLM_Model
from src.tokenizer_utils import build_id_to_token
import json
from src.tokenizer_utils import encode_prompt, decode_ids
model = Small_LLM_Model()
prompt = "What is the sum of 2 and 3?"
id_to_token = build_id_to_token(model)
input_ids = encode_prompt(model, prompt)
print(input_ids)
print(type(input_ids))
print(type(input_ids[0]))
prompt = "What is the sum of 2 and 3?"
input_ids = encode_prompt(model, prompt)

for tid in input_ids:
    print(tid, repr(id_to_token[tid]))
decoded = decode_ids(id_to_token, input_ids)
print(repr(decoded))
"""

"""
TrieNode tester
from src.constrained import TrieNode, insert

root = TrieNode()
insert(root, "fn_add_numbers")
insert(root, "fn_greet")
insert(root, "fn_reverse_string")

print(root.children.keys())              # sadece "f" olmalı
print(root.children["f"].children.keys())  # sadece "n" olmalı
print(root.children["f"].children["n"].children.keys())  # sadece "_" olmalı
print(root.children["f"].children["n"].children["_"].children.keys())
# "a", "g", "r" olmalı
"""

"""
valid token test
from src.constrained import TrieNode, insert, valid_token_ids
from llm_sdk import Small_LLM_Model
from src.tokenizer_utils import build_id_to_token

model = Small_LLM_Model()
id_to_token = build_id_to_token(model)

root = TrieNode()
insert(root, "fn_add_numbers")
insert(root, "fn_greet")
insert(root, "reverse_string")

valid = valid_token_ids(root, "", id_to_token)
print(len(valid))
for tid in valid[:10]:
    print(tid, repr(id_to_token[tid]))
"""

"""
from llm_sdk import Small_LLM_Model
from src.tokenizer_utils import build_id_to_token, encode_prompt
from src.constrained import TrieNode, insert, valid_token_ids, mask_logits

model = Small_LLM_Model()
id_to_token = build_id_to_token(model)

root = TrieNode()
insert(root, "fn_add_numbers")
insert(root, "fn_greet")
insert(root, "fn_reverse_string")

prompt = "What is the sum of 2 and 3?"
input_ids = encode_prompt(model, prompt)

valid_ids = valid_token_ids(root, "", id_to_token)
logits = model.get_logits_from_input_ids(input_ids)
masked = mask_logits(logits, valid_ids)

print(len(masked))
best = masked.index(max(masked))
print(best, repr(id_to_token[best]))
"""


"""
fonksiyon ismi doğru seçiliyor mu testi.
from llm_sdk import Small_LLM_Model
from src.tokenizer_utils import build_id_to_token, encode_prompt
from src.constrained import TrieNode, insert, generate_function_name

model = Small_LLM_Model()
id_to_token = build_id_to_token(model)

root = TrieNode()
insert(root, "fn_add_numbers")
insert(root, "fn_greet")
insert(root, "fn_reverse_string")

prompt = "Reverse the string 'hello'"
input_ids = encode_prompt(model, prompt)

name, generated_ids = generate_function_name(model, input_ids, id_to_token, root)
print(name)
print(generated_ids)
"""

"""

from llm_sdk import Small_LLM_Model
from src.tokenizer_utils import build_id_to_token
from src.constrained import valid_number_token_ids

model = Small_LLM_Model()
id_to_token = build_id_to_token(model)

valid = valid_number_token_ids("", id_to_token)
print(len(valid))
for tid in valid[:15]:
    print(tid, repr(id_to_token[tid]))
"""

"""
parametre üretim testi
from llm_sdk import Small_LLM_Model
from src.tokenizer_utils import build_id_to_token, encode_prompt
from src.constrained import generate_number

model = Small_LLM_Model()
id_to_token = build_id_to_token(model)

prompt = "What is the sum of 2 and 3? The first number is"
input_ids = encode_prompt(model, prompt)

number, generated_ids = generate_number(model, input_ids, id_to_token)
print(repr(number))
print(generated_ids)
"""

"""from llm_sdk import Small_LLM_Model
from src.tokenizer_utils import build_id_to_token, encode_prompt
from src.constrained import generate_string

model = Small_LLM_Model()
id_to_token = build_id_to_token(model)

prompt = 'Greet Eda\n{"name": "fn_greet", "parameters": {"name": "'
input_ids = encode_prompt(model, prompt)

s, ids = generate_string(model, input_ids, id_to_token)
print(repr(s))
print(ids)"""