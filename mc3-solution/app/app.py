#!/usr/bin/env python3
"""AMD Mini-Challenge 3 (RAG) - thin client.

    python3 /app/app.py --index /app/corpus
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."

All the work happens in /app/server.py, a long-running process started by the
container CMD that keeps the model on the GPU and the index in memory. This
script only talks to it over a unix socket, so each question costs one socket
round-trip, not a model load. If anything goes wrong, a well-formed refusal is
still written: a missing output file scores zero, an empty answer might not.
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
SOCK = os.environ.get("MC3_SOCK", "/tmp/mc3.sock")
PIDFILE = os.environ.get("MC3_PIDFILE", "/tmp/mc3_server.pid")
SERVER_LOG = os.environ.get("MC3_SERVER_LOG", "/tmp/mc3_server.log")
QUERY_BUDGET_S = float(os.environ.get("MC3_QUERY_BUDGET_S", "27"))
INDEX_WAIT_S = float(os.environ.get("MC3_INDEX_WAIT_S", "560"))


def server_alive() -> bool:
    try:
        pid = int(Path(PIDFILE).read_text().strip())
        os.kill(pid, 0)
        try:  # a crashed server reparented to a non-reaping PID 1 lingers as a zombie
            state = Path(f"/proc/{pid}/status").read_text()
            if "\nState:\tZ" in state or "server.py" not in Path(f"/proc/{pid}/cmdline").read_text():
                return False
        except FileNotFoundError:
            return False
        except Exception:
            pass
        return True
    except Exception:
        return False


def ensure_server():
    if server_alive():
        return
    with open(SERVER_LOG, "ab") as fh:
        subprocess.Popen([sys.executable, str(HERE / "server.py")], stdout=fh, stderr=fh,
                         stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(HERE))
    time.sleep(0.5)


def request(payload: dict, deadline: float) -> dict:
    """Wait for the socket (server may still be loading), then do one round-trip."""
    last = None
    while time.time() < deadline:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(max(1.0, deadline - time.time()))
            s.connect(SOCK)
            break
        except (FileNotFoundError, ConnectionRefusedError, socket.timeout, OSError) as e:
            last = e
            try:
                s.close()
            except Exception:
                pass
            time.sleep(0.5)
    else:
        raise TimeoutError(f"server not reachable: {last!r}")
    with s:
        s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        buf = b""
        while not buf.endswith(b"\n"):
            s.settimeout(max(0.5, deadline - time.time()))
            part = s.recv(65536)
            if not part:
                break
            buf += part
    return json.loads(buf.decode("utf-8"))


def write_output(query_id: str, answer: str, citations, confidence: float):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / (query_id + "_output.json")
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps({"answer": answer or "", "citations": list(citations or []),
                               "confidence": float(confidence or 0.0)}, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, out)   # atomic: the harness never reads a half-written file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, help="build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem the harness assigns, e.g. query_01")
    ap.add_argument("--query", help="the question to answer")
    args = ap.parse_args()

    if args.index is not None:
        ensure_server()
        try:
            resp = request({"op": "index", "corpus": str(args.index.resolve())}, T0 + INDEX_WAIT_S)
            print(json.dumps(resp, ensure_ascii=False, indent=1))
        except Exception as e:
            print(f"index request failed: {e!r}", file=sys.stderr)
        return 0

    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    answer, cites, conf = "", [], 0.0
    try:
        ensure_server()
        resp = request({"op": "query", "corpus": str(args.corpus.resolve()), "query": args.query},
                       T0 + QUERY_BUDGET_S)
        answer = resp.get("answer", "") or ""
        cites = resp.get("citations", []) or []
        conf = resp.get("confidence", 0.0) or 0.0
        if not answer:
            cites = []
    except Exception as e:
        print(f"query failed, writing refusal: {e!r}", file=sys.stderr)
    write_output(args.query_id, answer, cites, conf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
