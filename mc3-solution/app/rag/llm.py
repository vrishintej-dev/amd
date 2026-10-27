"""One vision-language model, resident on the GPU, used for both OCR and answering.

Default: Qwen3-VL-4B-Instruct (bf16, ~9 GB). Any Hugging Face image-text-to-text
chat model with a chat template should work; set MODEL_DIR to switch.
"""

from __future__ import annotations

import os
import time

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/vlm")
MAX_IMAGE_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1600"))
MIN_IMAGE_SIDE = 640

OCR_PROMPT = """Transcribe ALL text visible in this {hint}, exactly as printed (keep codes,
part numbers, prefixes, units and punctuation).
Then preserve the meaning of the layout:
- For diagrams, pinouts, tables or charts where a label is attached to a box, pin, cell or
  line, write one line per association as "<box/pin/cell>: <label>".
- For labels, forms and nameplates write "<FIELD>: <VALUE>" lines.
Output only the transcription, no commentary."""


class VLM:
    def __init__(self, model_dir: str = MODEL_DIR, log=print):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoM
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as AutoM

        if not torch.cuda.is_available():
            raise RuntimeError("no GPU visible to torch (ROCm) - refusing to run on CPU")
        t0 = time.time()
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        kw = dict(local_files_only=True, attn_implementation="sdpa")
        try:
            self.model = AutoM.from_pretrained(model_dir, dtype=torch.bfloat16, **kw)
        except TypeError:
            self.model = AutoM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, **kw)
        self.model.to("cuda").eval()
        self.log = log
        log(f"[vlm] loaded {model_dir} in {time.time()-t0:.1f}s on {torch.cuda.get_device_name(0)}; "
            f"VRAM {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    @staticmethod
    def _fit(img):
        w, h = img.size
        scale = 1.0
        if max(w, h) > MAX_IMAGE_SIDE:
            scale = MAX_IMAGE_SIDE / max(w, h)
        elif max(w, h) < MIN_IMAGE_SIDE:
            scale = MIN_IMAGE_SIDE / max(w, h)
        if scale != 1.0:
            img = img.resize((max(28, int(w * scale)), max(28, int(h * scale))))
        return img

    def chat(self, system: str, user: str, images=None, max_new_tokens: int = 128) -> str:
        torch = self.torch
        images = [self._fit(i) for i in (images or [])]
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": user}]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": content},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images or None, return_tensors="pt")
        inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen, skip_special_tokens=True).strip()

    def ocr(self, img, hint: str = "image") -> str:
        return self.chat("You are an exact OCR engine.", OCR_PROMPT.format(hint=hint), [img], 1024)
