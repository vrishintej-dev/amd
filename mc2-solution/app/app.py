#!/usr/bin/env python3
"""AMD Mini-Challenge 2 (OCR) - thin client.

    python3 /app/app.py --input-image /app/input/image_01.png
    -> /app/output/image_01_output.json  {"text": "...", "confidence": 0.9}

The model lives in /app/server.py (started by the container CMD), so each image
costs one socket round-trip instead of a model load. A valid JSON file is ALWAYS
written, and the script always exits well inside the 30 s per-image budget.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", "/app/output"))
SOCK = os.environ.get("MC2_SOCK", "/tmp/mc2.sock")
PIDFILE = os.environ.get("MC2_PIDFILE", "/tmp/mc2_server.pid")
LOG = os.environ.get("MC2_SERVER_LOG", "/tmp/mc2_server.log")
BUDGET = float(os.environ.get("MC2_BUDGET_S", "26"))


def server_alive() -> bool:
    try:
        pid = int(Path(PIDFILE).read_text())
        os.kill(pid, 0)
        st = Path(f"/proc/{pid}/status").read_text()
        return "\nState:\tZ" not in st and "server.py" in Path(f"/proc/{pid}/cmdline").read_text()
    except Exception:
        return False


def ensure_server():
    if not server_alive():
        with open(LOG, "ab") as fh:
            subprocess.Popen([sys.executable, str(HERE / "server.py")], stdout=fh, stderr=fh,
                             stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(HERE))


def ask(image: str) -> dict:
    deadline = T0 + BUDGET
    while True:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(max(0.5, deadline - time.time()))
            s.connect(SOCK)
            break
        except OSError:
            s.close()
            if time.time() > deadline:
                raise TimeoutError("server not ready")
            time.sleep(0.3)
    with s:
        s.sendall((json.dumps({"op": "ocr", "image": image}) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            s.settimeout(max(0.3, deadline - time.time()))
            part = s.recv(65536)
            if not part:
                break
            buf += part
    return json.loads(buf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-image", required=True)
    a = ap.parse_args()
    text, conf = "", 0.0
    try:
        ensure_server()
        r = ask(str(Path(a.input_image).resolve()))
        text, conf = r.get("text", "") or "", float(r.get("confidence", 0.0) or 0.0)
    except Exception as e:
        print(f"ocr failed: {e!r}", file=sys.stderr)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / (Path(a.input_image).stem + "_output.json")
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps({"text": text, "confidence": conf}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
