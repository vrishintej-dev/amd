#!/usr/bin/env python3
"""AMD AI League Match 4 (Intelligent Web Information Retrieval) - thin client.

The official Match 4 harness was not published when this was written, so this
follows the container contract of Matches 2-3 and accepts every likely spelling:

    python3 /app/app.py --query-id q01 --query "What is the price of ...?" --url https://site/
    python3 /app/app.py --id q01 --question "..." --start-url https://site/
    python3 /app/app.py --index /app/corpus                    # saved pages (optional)
    python3 /app/app.py --query-id q01 --query "..." --corpus /app/corpus
    python3 /app/app.py --input /app/input/q01.json            # {"id","query"/"question","url"}

Writes /app/output/<id>_output.json:
    {"answer": "...", "citations": [...], "sources": [...], "source_url": "...", "confidence": 0.9}
The answer is the specific value only; "" when the pages do not contain it.
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
OUTPUT_DIR = Path(os.environ.get("MC4_OUTPUT_DIR", "/app/output"))
SOCK = os.environ.get("MC4_SOCK", "/tmp/mc4.sock")
PIDFILE = os.environ.get("MC4_PIDFILE", "/tmp/mc4_server.pid")
SERVER_LOG = os.environ.get("MC4_SERVER_LOG", "/tmp/mc4_server.log")
QUERY_BUDGET_S = float(os.environ.get("MC4_QUERY_BUDGET_S", "27"))
INDEX_WAIT_S = float(os.environ.get("MC4_INDEX_WAIT_S", "560"))


def server_alive() -> bool:
    try:
        pid = int(Path(PIDFILE).read_text().strip())
        os.kill(pid, 0)
        st = Path(f"/proc/{pid}/status").read_text()
        return "\nState:\tZ" not in st and "server.py" in Path(f"/proc/{pid}/cmdline").read_text()
    except Exception:
        return False


def ensure_server():
    if not server_alive():
        with open(SERVER_LOG, "ab") as fh:
            subprocess.Popen([sys.executable, str(HERE / "server.py")], stdout=fh, stderr=fh,
                             stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(HERE))
        time.sleep(0.5)


def request(payload: dict, deadline: float) -> dict:
    last = None
    while time.time() < deadline:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(max(1.0, deadline - time.time()))
            s.connect(SOCK)
            break
        except OSError as e:
            last = e
            s.close()
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


def write_output(qid: str, answer: str, cites, conf: float):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cites = list(cites or [])
    out = OUTPUT_DIR / (qid + "_output.json")
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps({"answer": answer or "", "citations": cites, "sources": cites,
                               "source_url": cites[0] if cites else "",
                               "confidence": float(conf or 0.0)}, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path)
    ap.add_argument("--query", "--question", "--q", "--request", "--task", dest="query")
    ap.add_argument("--query-id", "--id", "--qid", "--task-id", dest="qid")
    ap.add_argument("--url", "--start-url", "--website", "--site", "--start", dest="url")
    ap.add_argument("--corpus", "--pages", "--site-dir", dest="corpus", type=Path)
    ap.add_argument("--input", "--input-file", dest="input", type=Path)
    args, _unknown = ap.parse_known_args()

    if args.index is not None:
        ensure_server()
        try:
            print(json.dumps(request({"op": "index", "corpus": str(args.index.resolve())},
                                     T0 + INDEX_WAIT_S), indent=1))
        except Exception as e:
            print(f"index failed: {e!r}", file=sys.stderr)
        return 0

    query, qid, url = args.query, args.qid, args.url
    corpus = str(args.corpus.resolve()) if args.corpus else None
    if args.input:
        try:
            spec = json.loads(args.input.read_text(encoding="utf-8"))
            query = query or spec.get("query") or spec.get("question") or spec.get("task")
            url = url or spec.get("url") or spec.get("start_url") or spec.get("website")
            qid = qid or spec.get("id") or spec.get("query_id")
        except Exception as e:
            print(f"bad --input: {e!r}", file=sys.stderr)
        qid = qid or args.input.stem
    qid = qid or "query"
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url

    answer, cites, conf = "", [], 0.0
    if query:
        try:
            ensure_server()
            r = request({"op": "web", "query": query, "url": url, "corpus": corpus},
                        T0 + QUERY_BUDGET_S)
            answer = r.get("answer", "") or ""
            cites = (r.get("citations") or []) if answer else []
            conf = r.get("confidence", 0.0) or 0.0
        except Exception as e:
            print(f"request failed, writing empty answer: {e!r}", file=sys.stderr)
    write_output(qid, answer, cites, conf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
