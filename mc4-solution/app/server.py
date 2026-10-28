#!/usr/bin/env python3
"""Match 4 long-running worker: holds the model; answers web-retrieval requests over a unix socket.

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
from rag import web as W             # noqa: E402

SOCK = os.environ.get("MC4_SOCK", "/tmp/mc4.sock")
PIDFILE = os.environ.get("MC4_PIDFILE", "/tmp/mc4_server.pid")
INDEX_DIR = Path(os.environ.get("MC4_INDEX_DIR", "/app/index"))
INDEX_FILE = INDEX_DIR / "index.json"
OCR_CUTOFF_S = float(os.environ.get("MC4_OCR_CUTOFF_S", "470"))
CRAWL_BUDGET_S = float(os.environ.get("MC4_CRAWL_BUDGET_S", "12"))

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
    t0 = time.time()
    docs, report = C.build_docs(Path(corpus), ocr=ocr, ocr_deadline=T_START + OCR_CUTOFF_S, log=log)
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
    log(f"[index] done in {time.time()-t0:.1f}s: {len(docs)} docs, {len(chunks)} chunks")
    return {"ok": True, "docs": len(docs), "chunks": len(chunks), "skipped": report.skipped}


def do_web(query: str, url: str | None, corpus: str | None) -> dict:
    """Answer one natural-language request from a live site (url) or saved pages (corpus)."""
    t0 = time.time()
    if url:
        pages = W.crawl(url, query, budget_s=CRAWL_BUDGET_S, log=log)
        docs = [C.Doc(path=u, kind="web", text=t) for u, t in pages if t.strip()]
        chunks, nid = [], 0
        for d in docs:
            cs = C.chunk_doc(d, nid)
            chunks += cs
            nid += len(cs)
        if not docs:
            return {"answer": "", "citations": [], "confidence": 0.0}
        eng = Engine(C.to_jsonable(docs, chunks, C.IndexReport()), STATE["llm"], "", log)
    else:
        eng = STATE["engine"]
        if eng is None or (corpus and eng.corpus_root != corpus):
            do_index(corpus or "/app/corpus")
            eng = STATE["engine"]
    eng.soft_deadline = max(4.0, 22.0 - (time.time() - t0))   # crawl + answer < 30 s
    ans, cites, conf = eng.answer(query)
    log(f"[web] {query!r} -> {ans!r} {cites} ({time.time()-t0:.1f}s)")
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
                    elif op == "web":
                        resp = do_web(req["query"], req.get("url"), req.get("corpus"))
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
