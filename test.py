from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()


def unconstrained_generate(prompt, n=6):
    ids = model.encode(prompt)[0].tolist()
    out = []
    for _ in range(n):
        logits = model.get_logits_from_input_ids(ids + out)
        out.append(logits.index(max(logits)))
    return model.decode(out)


print(unconstrained_generate("What is the sum of 2 and 3?"))