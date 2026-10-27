# AMD AI League: Mini Challenges 2 & 3

Two containerised AI pipelines. They run entirely on an AMD GPU with ROCm and PyTorch, using the open-source **Qwen3-VL-4B-Instruct** vision-language model. No external APIs are called and no network is needed at inference time.

| Challenge | What it does |
|---|---|
| Mini Challenge 2: OCR | Reads licence plates (keeping the province character on Chinese plates and dropping state banners) and road signs (multi-line, top to bottom) from PNG, JPEG or TIFF images. |
| Mini Challenge 3: RAG | Indexes a mixed corpus (PDF, DOCX tables, multi-sheet XLSX, CSV, logs, source code, and images read by the VLM) and answers each question with the value plus the exact set of files it came from. Encrypted, unreadable and unknown files are skipped safely. |

## Design
- **Resident GPU worker.** `app/server.py` is started by the container CMD and loads the model once. `app/app.py` is a thin client that talks to it over a Unix socket, so each request costs a round-trip rather than a model reload. A valid JSON output is always written, even on errors.
- **OCR.** A rules-following prompt plus image normalisation: resizing, 16-bit TIFF handling and a contrast-boosted retry for dark images.
- **RAG.**
  - Fault-tolerant per-file parsing (pypdf, python-docx, openpyxl, csv), with superseded and withdrawn documents flagged.
  - Retrieval: dependency-free BM25 with identifier-aware tokenisation, identifier expansion, and a second retrieval hop for chained evidence (log, then ticket, then fix version).
  - A citation verifier keeps only the files that are strictly necessary, and refuses rather than guesses when the answer is not in a readable document.

## Results
Tested on an AMD Radeon Pro W7900D (ROCm 10, torch 2.13, transformers 5.21.0):
- **MC3:** 200/200 on the 10 official sample questions. Indexing takes 15 s and each question about 1 s.
- **MC2:** about 0.3 s per image after a 25 s model load.

Both images are built on the mandated `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0` base, are about 38 GiB each, and ship their weights inside the image.

## Layout
- `mc2-solution/`: OCR container (Dockerfile, app/app.py, app/server.py)
- `mc3-solution/`: RAG container (Dockerfile, app/app.py, app/server.py, app/rag/*)
