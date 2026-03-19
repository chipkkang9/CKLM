from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GemmaChatModel:
    def __init__(self, model_id: str = "google/gemma-3-4b-it") -> None:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
        )

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 256,
        do_sample: bool = False,
    ) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=True,
            )

        generated = outputs[0][input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
