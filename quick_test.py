import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-3-4b-it"

print("torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
model.eval()

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful research assistant."}],
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "Explain what RAG is in exactly 3 sentences."}],
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = {k: v.to(device) for k, v in inputs.items()}
input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=96,
        do_sample=False,
        use_cache=True,
    )

generated = outputs[0][input_len:]
text = tokenizer.decode(generated, skip_special_tokens=True)

print("\n=== MODEL OUTPUT ===\n")
print(text)