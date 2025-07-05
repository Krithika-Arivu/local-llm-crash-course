from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")


def get_prompt(instruction: str) -> str:
    system = "give one word answer . use only minimal words."
    prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt


question = "the capital of india is:"

for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
