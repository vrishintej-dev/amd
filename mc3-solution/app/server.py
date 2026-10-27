#!/usr/bin/env python3
"""Long-running worker: holds the model and the index, answers over a unix socket.

Started by the container CMD. /app/app.py is a thin client that talks to it, so
the model is loaded ONCE per container, not once per question.
"""

from __future__ import annotations

import json
import os
import socketserver
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag import corpus as C          # noqa: E402
from rag.engine import Engine        # noqa: E402

SOCK = os.environ.get("MC3_SOCK", "/tmp/mc3.sock")
PIDFILE = os.environ.get("MC3_PIDFILE", "/tmp/mc3_server.pid")
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
INDEX_FILE = INDEX_DIR / "index.json"
# Startup budget is 600 s for container start + model load + --index. Stop OCR-ing
# new images after this many seconds since the server started.
OCR_CUTOFF_S = float(os.environ.get("MC3_OCR_CUTOFF_S", "470"))

T_START = time.time()
STATE = {"llm": None, "engine": None, "llm_error": None, "ready": False}
LOCK = threading.Lock()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_llm():
    from rag.llm import VLM
    return VLM(log=log)


def load_engine_from_disk():
    if INDEX_FILE.exists():
        try:
            idx = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            STATE["engine"] = Engine(idx, STATE["llm"], idx.get("corpus", "/app/corpus"), log)
            log(f"[server] loaded index from disk: {len(idx['docs'])} docs")
        except Exception:
            log("[server] could not load index:", traceback.format_exc())


def do_index(corpus: str) -> dict:
    llm = STATE["llm"]
    ocr = llm.ocr if llm is not None else None
    deadline = T_START + OCR_CUTOFF_S
    t0 = time.time()
    docs, report = C.build_docs(Path(corpus), ocr=ocr, ocr_deadline=deadline, log=log)
    chunks, nid = [], 0
    for d in docs:
        cs = C.chunk_doc(d, nid)
        chunks += cs
        nid += len(cs)
    idx = C.to_jsonable(docs, chunks, report)
    idx["corpus"] = corpus
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, INDEX_FILE)
    STATE["engine"] = Engine(idx, llm, corpus, log)
    log(f"[index] done in {time.time()-t0:.1f}s: {len(docs)} docs, {len(chunks)} chunks, "
        f"{len(report.skipped)} skipped")
    return {"ok": True, "docs": len(docs), "chunks": len(chunks), "skipped": report.skipped}


def do_query(corpus: str, query: str) -> dict:
    eng = STATE["engine"]
    if eng is None:
        load_engine_from_disk()
        eng = STATE["engine"]
    if eng is None:  # never indexed: emergency text-only index (no OCR)
        log("[server] no index yet - building a quick one")
        global OCR_CUTOFF_S
        OCR_CUTOFF_S = 0
        do_index(corpus)
        eng = STATE["engine"]
    eng.corpus_root = corpus or eng.corpus_root
    t0 = time.time()
    ans, cites, conf = eng.answer(query)
    log(f"[query] {query!r} -> {ans!r} {cites} ({time.time()-t0:.1f}s)")
    return {"answer": ans, "citations": cites, "confidence": conf}


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            req = json.loads(self.rfile.readline().decode("utf-8"))
            op = req.get("op")
            if op == "ping":
                resp = {"ok": True, "ready": STATE["ready"], "llm": STATE["llm"] is not None,
                        "llm_error": STATE["llm_error"]}
            else:
                with LOCK:
                    if op == "index":
                        resp = do_index(req["corpus"])
                    elif op == "query":
                        resp = do_query(req.get("corpus", "/app/corpus"), req["query"])
                    else:
                        resp = {"ok": False, "error": f"unknown op {op}"}
        except Exception as e:
            log("[server] request failed:", traceback.format_exc())
            resp = {"ok": False, "error": repr(e), "answer": "", "citations": [], "confidence": 0.0}
        try:
            self.wfile.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
        except Exception:
            pass


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def main():
    Path(PIDFILE).write_text(str(os.getpid()))
    try:
        STATE["llm"] = load_llm()
    except Exception as e:
        STATE["llm_error"] = repr(e)
        log("[server] MODEL FAILED TO LOAD:", traceback.format_exc())
    load_engine_from_disk()
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = Server(SOCK, Handler)
    os.chmod(SOCK, 0o777)
    STATE["ready"] = True
    log(f"[server] listening on {SOCK} ({time.time()-T_START:.1f}s after start)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
