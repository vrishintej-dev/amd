#!/usr/bin/env python3
"""MC-2 resident OCR worker: loads the vision-language model ONCE, serves over a unix socket."""

from __future__ import annotations

import json
import os
import re
import socketserver
import threading
import time
import traceback
from pathlib import Path

SOCK = os.environ.get("MC2_SOCK", "/tmp/mc2.sock")
PIDFILE = os.environ.get("MC2_PIDFILE", "/tmp/mc2_server.pid")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/vlm")
MAX_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1280"))
MIN_SIDE = 448

PROMPT = """Read the text in this image. It shows a vehicle licence plate or a road sign,
possibly blurred, noisy, dark, glary or photographed at an angle.

Reply with ONLY the transcription, following these rules:
- LICENCE PLATE: give only the registration number, exactly as printed, in reading order.
  Leave out text printed AROUND the number: state or country names (CALIFORNIA, TEXAS,
  New York), slogans, websites, dealer frames and month/year stickers.
  On Chinese plates the leading province character and letter ARE part of the number:
  write e.g. 京A·12345 (do not drop 京A).
- ROAD SIGN: give every word and number printed on the sign, in reading order, top to
  bottom, with lines joined by single spaces, e.g. "SPEED LIMIT 65", "ROAD WORK AHEAD",
  "STOP". An advisory plaque that only shows a number: give just the number, no unit.
- No labels, no quotes, no explanation, no extra words."""

T0 = time.time()
LOCK = threading.Lock()
STATE = {"model": None, "error": None}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class VLM:
    def __init__(self):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoM
        if not torch.cuda.is_available():
            raise RuntimeError("no ROCm GPU visible")
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        kw = dict(local_files_only=True, attn_implementation="sdpa")
        try:
            self.model = AutoM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, **kw)
        except TypeError:
            self.model = AutoM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, **kw)
        self.model.to("cuda").eval()
        log(f"[vlm] loaded in {time.time()-T0:.1f}s, VRAM {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    def read(self, img, max_new_tokens=48) -> str:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        prompt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[prompt], images=[img], return_tensors="pt")
        inp = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inp.items()}
        with self.torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
        return self.proc.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)


def load_image(path: str):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im.load()
        try:
            im.seek(0)          # multi-page TIFF: first frame
        except Exception:
            pass
        if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):   # 16-bit / float TIFFs
            import numpy as np
            a = np.asarray(im, dtype="float32")
            lo, hi = float(a.min()), float(a.max())
            a = ((a - lo) / (hi - lo + 1e-6) * 255).clip(0, 255).astype("uint8")
            im = Image.fromarray(a)
        im = ImageOps.exif_transpose(im)
        img = im.convert("RGB")
    w, h = img.size
    s = 1.0
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
    elif max(w, h) < MIN_SIDE:
        s = MIN_SIDE / max(w, h)
    if s != 1.0:
        img = img.resize((max(28, round(w * s)), max(28, round(h * s))), Image.LANCZOS)
    return img


PREFIX = re.compile(r"^(the )?(text|plate( number)?|licen[cs]e plate( number)?|sign|answer|transcription)\s*(is\s*:?|:)\s*", re.I)


def clean(raw: str) -> str:
    t = (raw or "").strip().strip("`").strip()
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    t = " ".join(lines)
    t = PREFIX.sub("", t).strip().strip('"').strip("'").strip()
    t = re.sub(r"\s+", " ", t)
    return t.rstrip(".")


def ocr(path: str) -> dict:
    t0 = time.time()
    m = STATE["model"]
    if m is None:
        return {"text": "", "confidence": 0.0}
    img = load_image(path)
    text = clean(m.read(img))
    if not text and time.time() - t0 < 12:        # dark / low-contrast: retry enhanced
        from PIL import ImageOps
        text = clean(m.read(ImageOps.autocontrast(img, cutoff=1)))
    log(f"[ocr] {path} -> {text!r} ({time.time()-t0:.1f}s)")
    return {"text": text, "confidence": 0.9 if text else 0.0}


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            req = json.loads(self.rfile.readline())
            if req.get("op") == "ping":
                resp = {"ok": True, "model": STATE["model"] is not None, "error": STATE["error"]}
            else:
                with LOCK:
                    resp = ocr(req["image"])
        except Exception as e:
            log("[server] error:", traceback.format_exc())
            resp = {"text": "", "confidence": 0.0, "error": repr(e)}
        try:
            self.wfile.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
        except Exception:
            pass


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def main():
    Path(PIDFILE).write_text(str(os.getpid()))
    try:
        STATE["model"] = VLM()
        # warm-up so the first graded image doesn't pay kernel compilation
        from PIL import Image
        STATE["model"].read(Image.new("RGB", (448, 224), "white"), 4)
        log(f"[server] warm-up done at {time.time()-T0:.1f}s")
    except Exception as e:
        STATE["error"] = repr(e)
        log("[server] MODEL FAILED:", traceback.format_exc())
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = Server(SOCK, Handler)
    os.chmod(SOCK, 0o777)
    log(f"[server] ready on {SOCK} after {time.time()-T0:.1f}s")
    srv.serve_forever()


if __name__ == "__main__":
    main()
