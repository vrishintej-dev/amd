"""Retrieval, prompting and citation post-processing.

The LLM backend is anything with
    chat(system: str, user: str, images: list | None, max_new_tokens: int) -> str
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .search import BM25, identifiers, tokenize

FULL_CONTEXT_CHARS = 36_000     # whole corpus fits -> give the model everything
CONTEXT_CHARS = 22_000          # otherwise: top chunks up to this budget
EXPANSION_CHARS = 5_000         # extra room for identifier-linked chunks
SOFT_DEADLINE_S = 19.0          # stop optional passes after this (budget is 30s)

SYSTEM = """You are a precise question-answering engine over a private document collection.
You are given numbered FILES and a QUESTION. Answer ONLY from the files. The products and
companies in the files are fictional: anything you think you know about them is wrong.

Reply with ONE JSON object and nothing else:
{"answer": "<value>", "sources": ["F#", ...], "lookup": "<identifier or empty>"}

Rules for "answer":
- The value only, copied as written in the file: a number, part number, version, code, pin,
  date, quarter or name. No sentence, no explanation, no restating the question.
- Keep qualifiers that identify the value: "Q3 FY27" not "Q3"; "REV-C2" not "C2"; full part
  numbers with all suffixes; a version like "4.3.2" exactly.
- If a file says a value was changed, raised or corrected, give the CURRENT value, not the old one.
- Files marked WITHDRAWN / SUPERSEDED are obsolete; use the current document instead.
- If the files do not state the answer to exactly this question (right product, right
  quantity, right condition), answer "" with sources []. A similar value for a different
  product, part, volume or field is NOT an answer. Never guess. Never compute or estimate.

Rules for "sources":
- List only the files that are NECESSARY: a file belongs only if removing it would make the
  answer impossible. Usually that is one file.
- Chains: if one file gives an identifier (ticket id, error code, part number) and another
  file gives the value for that identifier, cite BOTH.
- Never cite a file just because it mentions the same topic.

Rules for "lookup":
- If the files give you an identifier that leads to the answer, but the value for that
  identifier is not in the files shown, put the identifier in "lookup", answer "" and list the
  file where you found the identifier in sources. Otherwise "lookup" is "".

Examples of correct replies:
{"answer": "94", "sources": ["F2"], "lookup": ""}
{"answer": "Q3 FY27", "sources": ["F1"], "lookup": ""}
{"answer": "4.3.2", "sources": ["F1", "F3"], "lookup": ""}
{"answer": "", "sources": [], "lookup": ""}
{"answer": "", "sources": ["F2"], "lookup": "ORR-1847"}"""

IMAGE_SYSTEM = """You read text in images precisely. Answer the question using ONLY what is
visible in this image. Reply with ONE JSON object: {"answer": "<value>"}.
The answer is the value only, exactly as printed (keep prefixes such as REV-, pin names such
as B14, full codes). If the image does not show the answer, reply {"answer": ""}."""

REFUSALS = {"", "NONE", "NULL", "N/A", "NA", "UNKNOWN", "NOTFOUND", "NOTSTATED",
            "NOTAVAILABLE", "NOTINTHEFILES", "NOANSWER", "IDONTKNOW", "NOTSPECIFIED"}

UNIT_WORDS = r"(seconds?|secs?|s|minutes?|mins?|hours?|days?|ms|c|°c|degrees?( c| celsius)?|celsius|" \
             r"w|watts?|v|volts?|gib|gb|mb|mib|tb/s|gb/s|usd|\$|%|percent|units?|pins?)"


def norm(s: str) -> str:
    """The grader's normalisation: upper, drop whitespace and - . · _"""
    return re.sub(r"[\s\-\.·_]", "", (s or "").upper())


def loose(s: str) -> str:
    return re.sub(r"[^0-9A-Z一-鿿]", "", (s or "").upper())


class Engine:
    def __init__(self, index: dict, llm=None, corpus_root: str = "/app/corpus", log=print):
        self.docs = {d["path"]: d for d in index["docs"]}
        self.order = [d["path"] for d in index["docs"]]
        self.chunks = index["chunks"]
        self.llm = llm
        self.corpus_root = corpus_root
        self.log = log
        self.bm25 = BM25([f"{c['path']}\n{c['text']}" for c in self.chunks])
        self.total_chars = sum(len(d["text"]) for d in index["docs"])
        # identifier -> set(files), for "specific" (rare) identifiers
        self.id_files: dict[str, set] = {}
        for p, d in self.docs.items():
            for ident in identifiers(d["text"]):
                self.id_files.setdefault(ident, set()).add(p)
        self.loose_text = {p: loose(d["text"]) for p, d in self.docs.items()}

    # ------------------------------------------------------------------ retrieval
    def rank(self, query: str, boost: dict | None = None):
        s = self.bm25.scores(query, boost)
        for i, c in enumerate(self.chunks):
            if self.docs[c["path"]].get("obsolete"):
                s[i] *= 0.6
        file_score: dict[str, float] = {}
        for i, c in enumerate(self.chunks):
            file_score[c["path"]] = max(file_score.get(c["path"], 0.0), s[i])
        files = sorted(file_score, key=lambda p: -file_score[p])
        return s, files, file_score

    def specific(self, ident: str) -> bool:
        return 0 < len(self.id_files.get(ident, ())) <= 3

    def select(self, query: str, must_idents: list[str] | None = None,
               must_files: list[str] | None = None):
        """Return an ordered list of (path, text) to show the model."""
        boost = {i: 3.0 for i in (must_idents or [])}
        scores, files, fscore = self.rank(query, boost)
        if self.total_chars <= FULL_CONTEXT_CHARS:
            ordered = list(dict.fromkeys((must_files or []) + files))
            return [(p, self.docs[p]["text"]) for p in ordered], files
        picked: list[int] = []
        used = 0
        # 1) everything mentioning a must-identifier, and chunks of must-files
        for i, c in enumerate(self.chunks):
            up = c["text"].upper()
            if any(ident.upper() in up for ident in (must_idents or [])) or \
                    c["path"] in (must_files or []) and scores[i] > 0:
                picked.append(i)
                used += len(c["text"])
        # 2) best chunks by score
        for i in sorted(range(len(self.chunks)), key=lambda i: -scores[i]):
            if scores[i] <= 0 or used > CONTEXT_CHARS:
                break
            if i in picked:
                continue
            picked.append(i)
            used += len(self.chunks[i]["text"])
        # 3) identifier expansion: rare identifiers in what we picked -> their other files
        extra = 0
        qid = identifiers(query)
        for i in list(picked[:8]):
            for ident in identifiers(self.chunks[i]["text"]):
                if ident in qid or not self.specific(ident):
                    continue
                for j, c in enumerate(self.chunks):
                    if j in picked:
                        continue
                    if c["path"] != self.chunks[i]["path"] and ident in c["text"].upper():
                        if extra + len(c["text"]) > EXPANSION_CHARS:
                            break
                        picked.append(j)
                        extra += len(c["text"])
        # group per file, files ordered by score, chunks in document order
        by_file: dict[str, list[int]] = {}
        for i in picked:
            by_file.setdefault(self.chunks[i]["path"], []).append(i)
        ordered = sorted(by_file, key=lambda p: -fscore.get(p, 0.0))
        out = []
        for p in ordered:
            ids = sorted(set(by_file[p]))
            out.append((p, "\n...\n".join(self.chunks[i]["text"] for i in ids)))
        return out, files

    # ------------------------------------------------------------------ prompting
    def render(self, ctx, note: str = ""):
        labels, parts = {}, []
        for n, (p, text) in enumerate(ctx, 1):
            lab = f"F{n}"
            labels[lab] = p
            d = self.docs[p]
            tags = [d["kind"]]
            if d["kind"] == "image":
                tags.append("text transcribed from an image")
            if d.get("obsolete"):
                tags.append("WITHDRAWN/SUPERSEDED - obsolete")
            parts.append(f"=== [{lab}] {p} ({'; '.join(tags)}) ===\n{text.strip()}\n")
        return labels, "\n".join(parts) + (f"\n{note}\n" if note else "")

    def ask(self, ctx, question, note=""):
        labels, files_txt = self.render(ctx, note)
        user = f"FILES:\n\n{files_txt}\nQUESTION: {question}\n\nReply with the JSON object only."
        raw = self.llm.chat(SYSTEM, user, None, 120)
        obj = parse_json(raw)
        ans = clean_answer(obj.get("answer", ""))
        srcs = []
        for s in obj.get("sources", []) or []:
            s = str(s).strip().strip("[]")
            if s in labels:
                srcs.append(labels[s])
            elif s in self.docs:
                srcs.append(s)
        lookup = str(obj.get("lookup", "") or "").strip()
        self.log(f"[llm] raw={raw!r}")
        return ans, list(dict.fromkeys(srcs)), lookup

    # ------------------------------------------------------------------ main entry
    def answer(self, question: str):
        t0 = time.time()
        if self.llm is None or not self.docs:
            return "", [], 0.0
        ctx, ranked = self.select(question)
        ans, srcs, lookup = self.ask(ctx, question)
        hop_files: list[str] = []

        # second hop: the model found an identifier whose value lives elsewhere
        if not ans and lookup and time.time() - t0 < SOFT_DEADLINE_S:
            hop_files = [s for s in srcs if s in self.docs]
            look_ids = [lookup] + [i for i in identifiers(lookup)]
            ctx2, _ = self.select(question + " " + lookup, must_idents=look_ids,
                                  must_files=hop_files)
            note = f"NOTE: identifier {lookup} was found in {', '.join(hop_files) or 'the files'}."
            ans, srcs2, _ = self.ask(ctx2, question, note)
            srcs = list(dict.fromkeys(hop_files + srcs2))

        if not ans:
            return "", [], 0.0

        cites = self.fix_citations(question, ans, srcs, [p for p, _ in ctx], ranked,
                                   [lookup] + list(identifiers(lookup)) if lookup else [])
        if not cites:
            self.log(f"[verify] answer {ans!r} not found in any file -> refusing")
            return "", [], 0.0

        # image answers: look at the picture again with the question in hand
        if len(cites) == 1 and self.docs[cites[0]]["kind"] == "image" and \
                time.time() - t0 < SOFT_DEADLINE_S and hasattr(self.llm, "chat"):
            a2 = self.reread_image(cites[0], question)
            if a2 and a2 != ans:
                self.log(f"[image] re-read {cites[0]}: {ans!r} -> {a2!r}")
                ans = a2
        conf = 0.9 if len(cites) == 1 else 0.75
        return ans, cites, conf

    def reread_image(self, rel: str, question: str) -> str:
        try:
            from PIL import Image
            with Image.open(Path(self.corpus_root) / rel) as im:
                im.load()
                img = im.convert("RGB")
            raw = self.llm.chat(IMAGE_SYSTEM, f"QUESTION: {question}", [img], 60)
            return clean_answer(parse_json(raw).get("answer", ""))
        except Exception as e:
            self.log(f"[image] re-read failed: {e!r}")
            return ""

    # ------------------------------------------------------------------ citations
    def contains(self, path: str, ans: str) -> bool:
        txt = self.loose_text.get(path, "")
        for cand in answer_candidates(ans):
            if cand and cand in txt:
                return True
        return False

    def answer_lines(self, path: str, ans: str) -> list[str]:
        cands = [c for c in answer_candidates(ans) if c]
        return [l for l in self.docs[path]["text"].splitlines()
                if any(c in loose(l) for c in cands)]

    def fix_citations(self, question, ans, srcs, ctx_files, ranked, lookup_ids):
        srcs = [s for s in srcs if s in self.docs]
        holders = [p for p in ctx_files if self.contains(p, ans)]
        if not holders:
            # maybe outside the shown context (should not happen) - check whole corpus
            holders = [p for p in self.order if self.contains(p, ans)]
            if not holders:
                return []
        cited_holders = [s for s in srcs if s in holders]
        if not cited_holders:
            # model cited the wrong file: take the best-ranked current file holding the value
            pool = [p for p in holders if not self.docs[p].get("obsolete")] or holders
            pool.sort(key=lambda p: ranked.index(p) if p in ranked else 999)
            cited_holders = [pool[0]]
        # drop obsolete holders when a current holder is also cited
        if any(not self.docs[p].get("obsolete") for p in cited_holders):
            cited_holders = [p for p in cited_holders if not self.docs[p].get("obsolete")]
        ans_ids = {i for i in identifiers(ans)}
        q_ids = identifiers(question)

        def link_ids(p):
            ids = set()
            for line in self.answer_lines(p, ans):
                for i in identifiers(line):
                    if i not in ans_ids and not any(i in a or a in i for a in ans_ids) \
                            and i not in q_ids and self.specific(i):
                        ids.add(i)
            return ids

        links = set()
        for h in cited_holders:
            links |= link_ids(h)
        keep = list(cited_holders)
        for i in lookup_ids:
            if any(i.upper() in self.docs[h]["text"].upper() for h in cited_holders):
                links.add(i.upper())
        # non-holder files the model cited: keep only if they carry a linking identifier
        for s in srcs:
            if s not in keep and any(i in self.docs[s]["text"].upper() for i in links):
                keep.append(s)
        # chain add-back: single holder whose answer row is keyed by an identifier that
        # the question does not mention, and a top-ranked file carries that identifier
        if len(keep) == 1 and links:
            top = [p for p in ranked[:3] if p not in keep]
            for p in top:
                up = self.docs[p]["text"].upper()
                if any(i in up for i in links):
                    keep.append(p)
                    break
        # stable order: as ranked
        return sorted(dict.fromkeys(keep), key=lambda p: ranked.index(p) if p in ranked else 999)


# ---------------------------------------------------------------------- helpers

def parse_json(raw: str) -> dict:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(raw[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    m = re.search(r'"answer"\s*:\s*"([^"]*)"', raw)
    srcs = re.findall(r'"(F\d+)"', raw)
    return {"answer": m.group(1) if m else "", "sources": srcs, "lookup": ""}


def clean_answer(a) -> str:
    if a is None:
        return ""
    if isinstance(a, (int, float)):
        a = str(int(a)) if float(a).is_integer() else str(a)
    a = str(a).strip().strip('"').strip("'").strip()
    a = a.rstrip(".").strip()
    if norm(a).replace("'", "") in REFUSALS or re.match(r"(?i)^(i (don't|do not)|not (found|stated|available|in)|no (answer|information))", a):
        return ""
    return a


def answer_candidates(ans: str) -> list[str]:
    a = ans.strip()
    c = [loose(a)]
    stripped = re.sub(rf"(?i)\s*{UNIT_WORDS}\s*$", "", a).strip()
    c.append(loose(stripped))
    m = re.match(r"^\s*([\d.,]+)\s*[^\d]*$", a)
    if m:
        c.append(loose(m.group(1)))
    return [x for x in dict.fromkeys(c) if x]
